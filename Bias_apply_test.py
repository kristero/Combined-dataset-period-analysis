from __future__ import annotations

import argparse
import ast
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks


REPO_DIR = Path(__file__).resolve().parent
DEFAULT_CORRECTION_DIR = REPO_DIR / "correction_files"
DEFAULT_OUTPUT_DIR = REPO_DIR / "bias_apply_test_outputs"


@dataclass(frozen=True)
class CorrectionOption:
    observatory: str
    code: str
    band: str
    catalog: str
    offset: float
    source_files: tuple[str, ...]
    confidence_tags: tuple[str, ...]
    max_confidence_rank: int

    @property
    def label(self) -> str:
        catalog_txt = self.catalog if self.catalog else "generic"
        return f"{self.observatory}|catalog={catalog_txt}|d={self.offset:+.6f}"


@dataclass
class AnalysisResult:
    variant: str
    offset: float
    catalog: str
    n_points: int
    removed_points: int
    primary_period_hours: float
    primary_power: float
    selected_period_hours: float
    selected_power: float
    selection_reason: str
    frequency: np.ndarray
    power: np.ndarray
    best_frequency: float
    t_rel: np.ndarray
    mags: np.ndarray
    phase_deg: np.ndarray
    model_phase_deg: np.ndarray
    model_mag: np.ndarray
    top_peaks: pd.DataFrame
    sheet_labels: np.ndarray | None = None
    point_offsets: np.ndarray | None = None


def slugify(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._+-]+", "_", text.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "variant"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a single-observatory Lomb-Scargle comparison with and without "
            "DePhOCUS correction-file offsets."
        )
    )
    parser.add_argument("--workbook", type=Path, default=None, help="Excel workbook to analyse.")
    parser.add_argument(
        "--mode",
        default="single",
        choices=["single", "combined_corrected_only"],
        help="Analysis mode.",
    )
    parser.add_argument("--sheet", default="T08o", help="Preferred observatory sheet.")
    parser.add_argument(
        "--mag-column",
        default="magph7",
        help="Magnitude column to use when present. Falls back automatically if unavailable.",
    )
    parser.add_argument("--min-period-hours", type=float, default=0.5)
    parser.add_argument("--max-period-hours", type=float, default=50.0)
    parser.add_argument("--nterms", type=int, default=2)
    parser.add_argument("--nfreq", type=int, default=30000)
    parser.add_argument("--outlier-iqr", type=float, default=1.8)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def discover_default_workbook() -> Path:
    preferred = REPO_DIR.parent / "Asteroid_data"
    candidate_names = ["4303-excelBF.xlsx", "*-excelBF.xlsx"]
    search_roots = [preferred, REPO_DIR.parent, REPO_DIR]
    for pattern in candidate_names:
        for root in search_roots:
            if not root.exists():
                continue
            matches = sorted(root.rglob(pattern))
            if matches:
                return matches[0]
    raise FileNotFoundError("Could not discover a default *-excelBF.xlsx workbook.")


def asteroid_number_from_path(path: Path) -> int | None:
    match = re.search(r"(\d+)-excel", path.name)
    return int(match.group(1)) if match else None


def confidence_rank_from_name(name: str) -> int:
    match = re.search(r"_g(\d+)\.txt$", name)
    return int(match.group(1)) if match else -1


def confidence_tag_from_name(name: str) -> str:
    match = re.search(r"_(g\d+)\.txt$", name)
    return match.group(1) if match else "g?"


def parse_sheet_identity(sheet_name: str) -> tuple[str, str]:
    if len(sheet_name) < 4:
        raise ValueError(f"Sheet name '{sheet_name}' is too short to parse as code+band.")
    return sheet_name[:3], sheet_name[3]


def load_correction_records(correction_dir: Path) -> list[dict]:
    records: list[dict] = []
    for path in sorted(correction_dir.glob("corr_DePhOCUS_*.txt")):
        rank = confidence_rank_from_name(path.name)
        tag = confidence_tag_from_name(path.name)
        for raw_line in path.read_text().splitlines():
            line = raw_line.strip()
            if not line:
                continue
            row = ast.literal_eval(line)
            row["source_file"] = path.name
            row["confidence_rank"] = rank
            row["confidence_tag"] = tag
            records.append(row)
    return records


def list_primary_sheets(workbook: Path) -> list[str]:
    sheet_names = pd.ExcelFile(workbook).sheet_names
    primary = []
    for sheet in sheet_names:
        if sheet == "mpj" or sheet.endswith("a") or sheet.endswith("fa"):
            continue
        if len(sheet) < 4:
            continue
        try:
            cols = pd.read_excel(workbook, sheet_name=sheet, nrows=0).columns.tolist()
        except Exception:
            continue
        has_time = ("JDc2" in cols) or ("epoch" in cols)
        has_mag = any(col in cols for col in ["magph7", "magred", "mag"])
        if "Ph" in cols and has_time and has_mag:
            primary.append(sheet)
    return primary


def build_correction_inventory(
    sheet_names: list[str], correction_records: list[dict]
) -> tuple[dict[str, list[CorrectionOption]], pd.DataFrame]:
    inventory: dict[str, list[CorrectionOption]] = {}
    rows: list[dict] = []

    for sheet in sheet_names:
        code, band = parse_sheet_identity(sheet)
        grouped: dict[tuple[str, float], dict] = {}

        for row in correction_records:
            if row.get("code") != code or row.get("band") != band:
                continue

            catalog = (row.get("catalog") or "").strip()
            offset = float(row["d"])
            key = (catalog, offset)
            if key not in grouped:
                grouped[key] = {
                    "source_files": set(),
                    "confidence_tags": set(),
                    "max_confidence_rank": -1,
                }
            grouped[key]["source_files"].add(str(row["source_file"]))
            grouped[key]["confidence_tags"].add(str(row["confidence_tag"]))
            grouped[key]["max_confidence_rank"] = max(
                grouped[key]["max_confidence_rank"], int(row["confidence_rank"])
            )

        options = []
        for (catalog, offset), meta in grouped.items():
            option = CorrectionOption(
                observatory=sheet,
                code=code,
                band=band,
                catalog=catalog,
                offset=offset,
                source_files=tuple(sorted(meta["source_files"])),
                confidence_tags=tuple(sorted(meta["confidence_tags"])),
                max_confidence_rank=int(meta["max_confidence_rank"]),
            )
            options.append(option)
            rows.append(
                {
                    "observatory": sheet,
                    "code": code,
                    "band": band,
                    "catalog": catalog,
                    "offset": offset,
                    "max_confidence_rank": option.max_confidence_rank,
                    "confidence_tags": ",".join(option.confidence_tags),
                    "source_files": ",".join(option.source_files),
                }
            )

        options.sort(
            key=lambda opt: (
                -opt.max_confidence_rank,
                -len(opt.source_files),
                opt.catalog == "",
                opt.catalog,
                opt.offset,
            )
        )
        inventory[sheet] = options

    detail_df = pd.DataFrame(rows).sort_values(
        ["observatory", "max_confidence_rank", "catalog", "offset"],
        ascending=[True, False, True, True],
    )
    return inventory, detail_df


def choose_target_sheet(
    preferred_sheet: str, sheet_names: list[str], inventory: dict[str, list[CorrectionOption]]
) -> tuple[str, str]:
    if preferred_sheet in sheet_names:
        reason = "preferred sheet found in workbook"
        return preferred_sheet, reason

    with_corrections = [sheet for sheet in sheet_names if inventory.get(sheet)]
    if with_corrections:
        fallback = sorted(
            with_corrections,
            key=lambda name: (-len(inventory[name]), name),
        )[0]
        return fallback, f"preferred sheet missing; fell back to '{fallback}'"

    fallback = sheet_names[0]
    return fallback, f"preferred sheet missing and no corrected sheet found; fell back to '{fallback}'"


def select_representative_correction(options: list[CorrectionOption]) -> CorrectionOption | None:
    if not options:
        return None
    return sorted(
        options,
        key=lambda opt: (
            -opt.max_confidence_rank,
            -len(opt.source_files),
            opt.catalog == "",
            opt.catalog,
            abs(opt.offset),
        ),
    )[0]


def load_observatory_series(workbook: Path, sheet_name: str, preferred_mag_column: str) -> pd.DataFrame:
    df = pd.read_excel(workbook, sheet_name=sheet_name)
    time_column = "JDc2" if "JDc2" in df.columns else "epoch"

    if preferred_mag_column in df.columns:
        mag = pd.to_numeric(df[preferred_mag_column], errors="coerce")
        mag_column = preferred_mag_column
    elif "magred" in df.columns:
        mag = pd.to_numeric(df["magred"], errors="coerce")
        mag_column = "magred"
    else:
        mag = pd.to_numeric(df["mag"], errors="coerce") - 5.0 * np.log10(
            pd.to_numeric(df.iloc[:, 3], errors="coerce") * pd.to_numeric(df.iloc[:, 4], errors="coerce")
        )
        mag_column = "computed_magred"

    out = pd.DataFrame(
        {
            "time_jd": pd.to_numeric(df[time_column], errors="coerce"),
            "phase_angle_deg": pd.to_numeric(df["Ph"], errors="coerce"),
            "mag": mag,
        }
    ).dropna()
    out.attrs["mag_column_used"] = mag_column
    out.attrs["time_column_used"] = time_column
    return out


def assemble_combined_dataset(
    workbook: Path,
    sheet_names: list[str],
    preferred_mag_column: str,
    correction_by_sheet: dict[str, CorrectionOption] | None = None,
    prefilter_outlier_iqr: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    time_parts = []
    mag_parts = []
    sheet_parts = []
    offset_parts = []
    meta: dict[str, str] = {}

    for sheet in sheet_names:
        df = load_observatory_series(workbook, sheet, preferred_mag_column=preferred_mag_column)
        option = correction_by_sheet.get(sheet) if correction_by_sheet is not None else None
        offset = float(option.offset) if option is not None else 0.0

        time_vals = df["time_jd"].to_numpy(dtype=float)
        raw_mag_vals = df["mag"].to_numpy(dtype=float)
        if prefilter_outlier_iqr is not None:
            _, _, keep = remove_iqr_outliers(time_vals, raw_mag_vals, outlier_iqr=float(prefilter_outlier_iqr))
            time_vals = time_vals[keep]
            raw_mag_vals = raw_mag_vals[keep]

        mag_vals = raw_mag_vals + offset
        n = len(mag_vals)
        if n == 0:
            continue

        time_parts.append(time_vals)
        mag_parts.append(mag_vals)
        sheet_parts.append(np.array([sheet] * n, dtype=object))
        offset_parts.append(np.full(n, offset, dtype=float))
        meta[f"{sheet}.mag_column_used"] = str(df.attrs["mag_column_used"])
        meta[f"{sheet}.time_column_used"] = str(df.attrs["time_column_used"])

    if not time_parts:
        raise ValueError("No valid observatory data loaded for combined dataset.")

    return (
        np.concatenate(time_parts),
        np.concatenate(mag_parts),
        np.concatenate(sheet_parts),
        np.concatenate(offset_parts),
        meta,
    )


def remove_iqr_outliers(times_jd: np.ndarray, mags: np.ndarray, outlier_iqr: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_rel = np.asarray(times_jd, dtype=float) - float(np.min(times_jd))
    y = np.asarray(mags, dtype=float)
    q1, q3 = np.percentile(y, [25, 75])
    iqr = q3 - q1
    lower = q1 - float(outlier_iqr) * iqr
    upper = q3 + float(outlier_iqr) * iqr
    keep = ((y >= lower) & (y <= upper)) | (t_rel < 5.0)
    return t_rel[keep], y[keep], keep


def count_model_extrema(ls: LombScargle, frequency: float, n_grid: int = 800) -> tuple[int, int]:
    phase = np.linspace(0.0, 1.0, int(n_grid), endpoint=False)
    y = ls.model(phase / frequency, frequency)
    return len(find_peaks(y)[0]), len(find_peaks(-y)[0])


def build_peak_table(period_hours: np.ndarray, power: np.ndarray, peak_indices: np.ndarray, top_n: int = 20) -> pd.DataFrame:
    peak_df = pd.DataFrame(
        {
            "peak_index": peak_indices,
            "period_hours": period_hours[peak_indices],
            "power": power[peak_indices],
        }
    ).sort_values("power", ascending=False, ignore_index=True)
    return peak_df.head(top_n)


def select_best_peak(
    ls: LombScargle,
    frequency: np.ndarray,
    power: np.ndarray,
    peak_indices: np.ndarray,
    double_period_tol: float = 0.12,
    search_top_n: int = 25,
    min_relative_power: float = 0.25,
) -> tuple[int, int, str]:
    if len(peak_indices) == 0:
        global_idx = int(np.argmax(power))
        return global_idx, global_idx, "no local peaks found; used global maximum"

    local_order = peak_indices[np.argsort(power[peak_indices])[::-1]]
    primary_idx = int(local_order[0])
    primary_frequency = float(frequency[primary_idx])
    primary_period = 24.0 / primary_frequency
    n_max, n_min = count_model_extrema(ls, primary_frequency)

    if not ((n_max <= 1) and (n_min <= 1)):
        return primary_idx, primary_idx, f"kept strongest local peak (n_max={n_max}, n_min={n_min})"

    primary_power = float(power[primary_idx])
    target_period = 2.0 * primary_period
    candidates: list[tuple[float, float, int]] = []
    for cand_idx in local_order[1 : 1 + int(search_top_n)]:
        cand_period = 24.0 / float(frequency[cand_idx])
        cand_power = float(power[cand_idx])
        if cand_power < min_relative_power * primary_power:
            continue
        rel_err = abs(cand_period - target_period) / target_period
        if rel_err <= float(double_period_tol):
            candidates.append((rel_err, -cand_power, int(cand_idx)))

    if not candidates:
        return primary_idx, primary_idx, "primary peak is single-peaked; no suitable 2x candidate found"

    chosen_idx = sorted(candidates)[0][2]
    chosen_period = 24.0 / float(frequency[chosen_idx])
    reason = (
        "primary peak is single-peaked; switched to closest strong local peak near 2x period "
        f"(P1={primary_period:.6f} h, P2={chosen_period:.6f} h)"
    )
    return primary_idx, chosen_idx, reason


def run_single_variant(
    times_jd: np.ndarray,
    mags: np.ndarray,
    variant: str,
    offset: float,
    catalog: str,
    outlier_iqr: float,
    min_period_hours: float,
    max_period_hours: float,
    nterms: int,
    nfreq: int,
) -> AnalysisResult:
    y = np.asarray(mags, dtype=float) + float(offset)
    t_rel, y_kept, keep = remove_iqr_outliers(times_jd, y, outlier_iqr=outlier_iqr)
    ls = LombScargle(t_rel, y_kept, nterms=nterms)

    min_frequency = 24.0 / float(max_period_hours)
    max_frequency = 24.0 / float(min_period_hours)
    frequency = np.linspace(min_frequency, max_frequency, int(nfreq), dtype=float)
    power = ls.power(frequency, method="fastchi2")

    peak_indices, _ = find_peaks(power, distance=max(50, int(nfreq) // 800))
    if len(peak_indices) == 0:
        peak_indices = np.array([int(np.argmax(power))], dtype=int)

    primary_idx, selected_idx, reason = select_best_peak(ls, frequency, power, peak_indices)
    top_peaks = build_peak_table(24.0 / frequency, power, peak_indices)

    best_frequency = float(frequency[selected_idx])
    best_period_hours = float(24.0 / best_frequency)
    phase = (t_rel * best_frequency) % 1.0
    phase_deg = phase * 360.0
    model_phase = np.linspace(0.0, 1.0, 600, endpoint=False)
    model_mag = ls.model(model_phase / best_frequency, best_frequency)

    return AnalysisResult(
        variant=variant,
        offset=float(offset),
        catalog=catalog,
        n_points=int(len(y_kept)),
        removed_points=int(np.size(y) - len(y_kept)),
        primary_period_hours=float(24.0 / float(frequency[primary_idx])),
        primary_power=float(power[primary_idx]),
        selected_period_hours=best_period_hours,
        selected_power=float(power[selected_idx]),
        selection_reason=reason,
        frequency=frequency,
        power=power,
        best_frequency=best_frequency,
        t_rel=t_rel,
        mags=y_kept,
        phase_deg=phase_deg,
        model_phase_deg=model_phase * 360.0,
        model_mag=model_mag,
        top_peaks=top_peaks,
        sheet_labels=np.array([variant] * len(y_kept), dtype=object),
        point_offsets=np.full(len(y_kept), float(offset), dtype=float),
    )


def run_combined_variant(
    times_jd: np.ndarray,
    mags: np.ndarray,
    sheet_labels: np.ndarray,
    point_offsets: np.ndarray,
    variant: str,
    outlier_iqr: float,
    min_period_hours: float,
    max_period_hours: float,
    nterms: int,
    nfreq: int,
    apply_combined_outlier_filter: bool = True,
) -> AnalysisResult:
    if apply_combined_outlier_filter:
        t_rel, y_kept, keep = remove_iqr_outliers(times_jd, mags, outlier_iqr=outlier_iqr)
        kept_sheets = np.asarray(sheet_labels, dtype=object)[keep]
        kept_offsets = np.asarray(point_offsets, dtype=float)[keep]
    else:
        t_rel = np.asarray(times_jd, dtype=float) - float(np.min(times_jd))
        y_kept = np.asarray(mags, dtype=float)
        kept_sheets = np.asarray(sheet_labels, dtype=object)
        kept_offsets = np.asarray(point_offsets, dtype=float)
    ls = LombScargle(t_rel, y_kept, nterms=nterms)

    min_frequency = 24.0 / float(max_period_hours)
    max_frequency = 24.0 / float(min_period_hours)
    frequency = np.linspace(min_frequency, max_frequency, int(nfreq), dtype=float)
    power = ls.power(frequency, method="fastchi2")

    peak_indices, _ = find_peaks(power, distance=max(50, int(nfreq) // 800))
    if len(peak_indices) == 0:
        peak_indices = np.array([int(np.argmax(power))], dtype=int)

    primary_idx, selected_idx, reason = select_best_peak(ls, frequency, power, peak_indices)
    top_peaks = build_peak_table(24.0 / frequency, power, peak_indices)

    best_frequency = float(frequency[selected_idx])
    best_period_hours = float(24.0 / best_frequency)
    phase = (t_rel * best_frequency) % 1.0
    phase_deg = phase * 360.0
    model_phase = np.linspace(0.0, 1.0, 600, endpoint=False)
    model_mag = ls.model(model_phase / best_frequency, best_frequency)

    return AnalysisResult(
        variant=variant,
        offset=float(np.mean(kept_offsets)) if len(kept_offsets) else 0.0,
        catalog="combined",
        n_points=int(len(y_kept)),
        removed_points=int(np.size(mags) - len(y_kept)),
        primary_period_hours=float(24.0 / float(frequency[primary_idx])),
        primary_power=float(power[primary_idx]),
        selected_period_hours=best_period_hours,
        selected_power=float(power[selected_idx]),
        selection_reason=reason,
        frequency=frequency,
        power=power,
        best_frequency=best_frequency,
        t_rel=t_rel,
        mags=y_kept,
        phase_deg=phase_deg,
        model_phase_deg=model_phase * 360.0,
        model_mag=model_mag,
        top_peaks=top_peaks,
        sheet_labels=kept_sheets,
        point_offsets=kept_offsets,
    )


def save_inventory(
    out_dir: Path,
    asteroid_number: int | None,
    detail_df: pd.DataFrame,
    inventory: dict[str, list[CorrectionOption]],
) -> None:
    stem = f"{asteroid_number}_observatory_corrections" if asteroid_number is not None else "observatory_corrections"
    detail_df.to_csv(out_dir / f"{stem}.csv", index=False)

    json_payload = {}
    for sheet, options in inventory.items():
        json_payload[sheet] = [
            {
                "catalog": option.catalog,
                "offset": option.offset,
                "source_files": list(option.source_files),
                "confidence_tags": list(option.confidence_tags),
                "max_confidence_rank": option.max_confidence_rank,
            }
            for option in options
        ]
    (out_dir / f"{stem}.json").write_text(json.dumps(json_payload, indent=2))


def save_selected_corrections(
    out_dir: Path,
    asteroid_number: int | None,
    correction_by_sheet: dict[str, CorrectionOption],
    mode: str,
) -> Path:
    rows = []
    for sheet, option in sorted(correction_by_sheet.items()):
        rows.append(
            {
                "observatory": sheet,
                "catalog": option.catalog if option.catalog else "generic",
                "offset": option.offset,
                "max_confidence_rank": option.max_confidence_rank,
                "confidence_tags": ",".join(option.confidence_tags),
                "source_files": ",".join(option.source_files),
            }
        )
    df = pd.DataFrame(rows)
    stem = f"{asteroid_number}_{mode}_selected_corrections" if asteroid_number is not None else f"{mode}_selected_corrections"
    out_path = out_dir / f"{stem}.csv"
    df.to_csv(out_path, index=False)
    return out_path


def save_results_table(
    out_dir: Path,
    asteroid_number: int | None,
    target_sheet: str,
    results: list[AnalysisResult],
    reference_period_hours: float | None,
) -> Path:
    rows = []
    for res in results:
        row = {
            "variant": res.variant,
            "catalog": res.catalog,
            "offset": res.offset,
            "n_points": res.n_points,
            "removed_points": res.removed_points,
            "primary_period_hours": res.primary_period_hours,
            "primary_power": res.primary_power,
            "selected_period_hours": res.selected_period_hours,
            "selected_power": res.selected_power,
            "selection_reason": res.selection_reason,
        }
        if reference_period_hours is not None:
            row["combined_reference_period_hours"] = reference_period_hours
            row["delta_vs_combined_reference_seconds"] = (
                (res.selected_period_hours - reference_period_hours) * 3600.0
            )
        rows.append(row)

    df = pd.DataFrame(rows)
    stem = f"{asteroid_number}_{target_sheet}_bias_apply_results" if asteroid_number is not None else f"{target_sheet}_bias_apply_results"
    out_path = out_dir / f"{stem}.csv"
    df.to_csv(out_path, index=False)
    return out_path


def load_combined_reference_period(asteroid_number: int | None) -> float | None:
    if asteroid_number is None:
        return None
    ls_path = REPO_DIR / "LS_data" / f"{asteroid_number}_LS_results_n=2.pkl"
    if not ls_path.exists():
        return None
    with ls_path.open("rb") as handle:
        frequency, power = pickle.load(handle)
    frequency = np.asarray(frequency, dtype=float)
    power = np.asarray(power, dtype=float)
    if frequency.size == 0 or power.size == 0:
        return None
    return float(24.0 / frequency[int(np.argmax(power))])


def plot_power_comparison(out_dir: Path, asteroid_number: int | None, target_sheet: str, baseline: AnalysisResult, corrected: AnalysisResult) -> Path:
    period_base = 24.0 / baseline.frequency
    period_corr = 24.0 / corrected.frequency

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=200)
    ax.plot(period_base, baseline.power, color="black", lw=1.0, label=f"{baseline.variant}")
    ax.plot(period_corr, corrected.power, color="#c36a05", lw=1.0, alpha=0.95, label=f"{corrected.variant}")
    ax.axvline(baseline.selected_period_hours, color="black", ls="--", lw=0.8)
    ax.axvline(corrected.selected_period_hours, color="#c36a05", ls="--", lw=0.8)
    ax.set_xlabel("Rotational period (hours)")
    ax.set_ylabel("Power")
    ax.set_title(f"{target_sheet}: Lomb-Scargle power comparison")
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(frameon=False)
    plt.tight_layout()

    stem = f"{asteroid_number}_{target_sheet}_power_compare" if asteroid_number is not None else f"{target_sheet}_power_compare"
    out_path = out_dir / f"{stem}.png"
    fig.savefig(out_path, dpi=250)
    plt.close(fig)
    return out_path


def plot_phase_comparison(out_dir: Path, asteroid_number: int | None, target_sheet: str, baseline: AnalysisResult, corrected: AnalysisResult) -> Path:
    def _display_name(res: AnalysisResult) -> str:
        if res.variant == "no_correction":
            return "No correction"
        if res.variant == "bias_applied":
            return "Bias applied"
        return "Representative correction"

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), dpi=200, sharey=True)
    panel_defs = [
        (axes[0], baseline, _display_name(baseline)),
        (axes[1], corrected, _display_name(corrected)),
    ]

    for ax, res, title in panel_defs:
        ax.scatter(res.phase_deg, res.mags, s=12, alpha=0.7, color="#9c5d00", edgecolors="none")
        ax.plot(res.model_phase_deg, res.model_mag, "--", lw=1.0, color="black")
        ax.set_xlim(0, 360)
        ax.set_xlabel("Rotational phase (deg)")
        ax.set_title(
            f"{title}\nP={res.selected_period_hours:.6f} h, offset={res.offset:+.6f}"
        )
        ax.grid(alpha=0.25, linestyle=":")
        ax.invert_yaxis()

    axes[0].set_ylabel("Magnitude")
    fig.suptitle(f"{target_sheet}: phase-folded light curves", y=1.02)
    plt.tight_layout()

    stem = f"{asteroid_number}_{target_sheet}_phase_compare" if asteroid_number is not None else f"{target_sheet}_phase_compare"
    out_path = out_dir / f"{stem}.png"
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return out_path


def export_single_light_curve(
    out_dir: Path,
    asteroid_number: int | None,
    target_sheet: str,
    result: AnalysisResult,
    label: str,
) -> tuple[Path, Path]:
    stem_prefix = f"{asteroid_number}_{target_sheet}" if asteroid_number is not None else target_sheet
    variant_slug = slugify(label)

    data_df = pd.DataFrame(
        {
            "observatory": result.sheet_labels if result.sheet_labels is not None else np.array([target_sheet] * len(result.mags), dtype=object),
            "time_rel_days": result.t_rel,
            "phase_deg": result.phase_deg,
            "magnitude": result.mags,
            "point_offset_mag": result.point_offsets if result.point_offsets is not None else np.full(len(result.mags), result.offset, dtype=float),
        }
    ).sort_values(["observatory", "phase_deg"], ignore_index=True)
    data_df["selected_period_hours"] = result.selected_period_hours
    data_df["offset_applied_mag"] = result.offset
    data_df["catalog"] = result.catalog if result.catalog else "generic"

    model_df = pd.DataFrame(
        {
            "model_phase_deg": result.model_phase_deg,
            "model_magnitude": result.model_mag,
        }
    )

    max_len = max(len(data_df), len(model_df))
    export_df = pd.DataFrame(index=np.arange(max_len))
    for col in data_df.columns:
        export_df[col] = pd.Series(data_df[col].to_numpy())
    for col in model_df.columns:
        export_df[col] = pd.Series(model_df[col].to_numpy())

    csv_path = out_dir / f"{stem_prefix}_{variant_slug}_light_curve.csv"
    export_df.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(5.6, 4.6), dpi=200)
    if result.sheet_labels is not None and len(np.unique(result.sheet_labels)) > 1:
        unique_sheets = list(dict.fromkeys(result.sheet_labels.tolist()))
        cmap = plt.get_cmap("tab20")
        colors = cmap(np.linspace(0, 1, max(len(unique_sheets), 2)))
        for idx, sheet in enumerate(unique_sheets):
            mask = result.sheet_labels == sheet
            ax.scatter(
                result.phase_deg[mask],
                result.mags[mask],
                s=12,
                alpha=0.72,
                color=colors[idx],
                edgecolors="none",
                label=sheet,
            )
        ax.legend(frameon=False, fontsize=6, ncol=2)
    else:
        ax.scatter(result.phase_deg, result.mags, s=12, alpha=0.72, color="#9c5d00", edgecolors="none")
    ax.plot(result.model_phase_deg, result.model_mag, "--", lw=1.0, color="black")
    ax.set_xlim(0, 360)
    ax.set_xlabel("Rotational phase (deg)")
    ax.set_ylabel("Magnitude")
    ax.set_title(
        f"{target_sheet}: {label}\nP={result.selected_period_hours:.6f} h, offset={result.offset:+.6f}"
    )
    ax.grid(alpha=0.25, linestyle=":")
    ax.invert_yaxis()
    plt.tight_layout()

    png_path = out_dir / f"{stem_prefix}_{variant_slug}_light_curve.png"
    fig.savefig(png_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return csv_path, png_path


def main() -> None:
    args = parse_args()

    workbook = args.workbook or discover_default_workbook()
    asteroid_number = asteroid_number_from_path(workbook)
    output_name = (
        f"{asteroid_number}_{args.sheet}" if args.mode == "single" and asteroid_number is not None
        else args.sheet if args.mode == "single"
        else f"{asteroid_number}_{args.mode}" if asteroid_number is not None
        else args.mode
    )
    output_dir = args.output_dir / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    correction_records = load_correction_records(DEFAULT_CORRECTION_DIR)
    sheet_names = list_primary_sheets(workbook)
    inventory, detail_df = build_correction_inventory(sheet_names, correction_records)
    save_inventory(output_dir, asteroid_number, detail_df, inventory)

    results: list[AnalysisResult] = []
    metadata_note = {}

    if args.mode == "single":
        target_sheet, selection_note = choose_target_sheet(args.sheet, sheet_names, inventory)
        correction_options = inventory.get(target_sheet, [])
        representative = select_representative_correction(correction_options)

        data = load_observatory_series(workbook, target_sheet, preferred_mag_column=args.mag_column)
        times_jd = data["time_jd"].to_numpy(dtype=float)
        mags = data["mag"].to_numpy(dtype=float)
        metadata_note["mag_column_used"] = data.attrs["mag_column_used"]
        metadata_note["time_column_used"] = data.attrs["time_column_used"]

        results.append(
            run_single_variant(
                times_jd=times_jd,
                mags=mags,
                variant="no_correction",
                offset=0.0,
                catalog="",
                outlier_iqr=args.outlier_iqr,
                min_period_hours=args.min_period_hours,
                max_period_hours=args.max_period_hours,
                nterms=args.nterms,
                nfreq=args.nfreq,
            )
        )

        for option in correction_options:
            results.append(
                run_single_variant(
                    times_jd=times_jd,
                    mags=mags,
                    variant=option.label,
                    offset=option.offset,
                    catalog=option.catalog,
                    outlier_iqr=args.outlier_iqr,
                    min_period_hours=args.min_period_hours,
                    max_period_hours=args.max_period_hours,
                    nterms=args.nterms,
                    nfreq=args.nfreq,
                )
            )
        selected_corrections_path = None
    else:
        corrected_sheets = [sheet for sheet in sheet_names if inventory.get(sheet)]
        if not corrected_sheets:
            raise ValueError("No observatory sheets with available corrections were found.")

        selection_note = (
            "combined dataset built only from observatories that have at least one available correction"
        )
        target_sheet = "combined_corrected_only"
        representative = None
        correction_by_sheet = {
            sheet: select_representative_correction(inventory[sheet])
            for sheet in corrected_sheets
        }
        correction_by_sheet = {k: v for k, v in correction_by_sheet.items() if v is not None}
        selected_corrections_path = save_selected_corrections(
            output_dir, asteroid_number, correction_by_sheet, args.mode
        )

        no_bias_times, no_bias_mags, no_bias_sheets, no_bias_offsets, load_meta = assemble_combined_dataset(
            workbook=workbook,
            sheet_names=sorted(correction_by_sheet),
            preferred_mag_column=args.mag_column,
            correction_by_sheet=None,
            prefilter_outlier_iqr=args.outlier_iqr,
        )
        bias_times, bias_mags, bias_sheets, bias_offsets, _ = assemble_combined_dataset(
            workbook=workbook,
            sheet_names=sorted(correction_by_sheet),
            preferred_mag_column=args.mag_column,
            correction_by_sheet=correction_by_sheet,
            prefilter_outlier_iqr=args.outlier_iqr,
        )
        metadata_note.update(load_meta)
        metadata_note["observatories_used"] = ",".join(sorted(correction_by_sheet))

        results.append(
            run_combined_variant(
                times_jd=no_bias_times,
                mags=no_bias_mags,
                sheet_labels=no_bias_sheets,
                point_offsets=no_bias_offsets,
                variant="no_correction",
                outlier_iqr=args.outlier_iqr,
                min_period_hours=args.min_period_hours,
                max_period_hours=args.max_period_hours,
                nterms=args.nterms,
                nfreq=args.nfreq,
                apply_combined_outlier_filter=False,
            )
        )
        results.append(
            run_combined_variant(
                times_jd=bias_times,
                mags=bias_mags,
                sheet_labels=bias_sheets,
                point_offsets=bias_offsets,
                variant="bias_applied",
                outlier_iqr=args.outlier_iqr,
                min_period_hours=args.min_period_hours,
                max_period_hours=args.max_period_hours,
                nterms=args.nterms,
                nfreq=args.nfreq,
                apply_combined_outlier_filter=False,
            )
        )

    reference_period_hours = load_combined_reference_period(asteroid_number)
    results_path = save_results_table(output_dir, asteroid_number, target_sheet, results, reference_period_hours)

    baseline = results[0]
    visual_compare = (
        representative.label if (args.mode == "single" and representative is not None) else "bias_applied"
    )
    corrected = next((res for res in results if res.variant == visual_compare), results[min(1, len(results) - 1)])
    power_plot = plot_power_comparison(output_dir, asteroid_number, target_sheet, baseline, corrected)
    phase_plot = plot_phase_comparison(output_dir, asteroid_number, target_sheet, baseline, corrected)
    baseline_csv, baseline_png = export_single_light_curve(
        output_dir,
        asteroid_number,
        target_sheet,
        baseline,
        label="no_correction",
    )
    corrected_csv, corrected_png = export_single_light_curve(
        output_dir,
        asteroid_number,
        target_sheet,
        corrected,
        label="bias_applied",
    )

    peak_rows = []
    for res in results:
        peak_df = res.top_peaks.copy()
        peak_df.insert(0, "variant", res.variant)
        peak_rows.append(peak_df)
    if peak_rows:
        peaks_path = output_dir / (
            f"{asteroid_number}_{target_sheet}_top_peaks.csv"
            if asteroid_number is not None
            else f"{target_sheet}_top_peaks.csv"
        )
        pd.concat(peak_rows, ignore_index=True).to_csv(peaks_path, index=False)

    summary = {
        "workbook": str(workbook),
        "mode": args.mode,
        "selection_note": selection_note,
        "target_sheet": target_sheet,
        "metadata": metadata_note,
        "n_unique_observatory_corrections": (
            len(correction_options) if args.mode == "single" else len(correction_by_sheet)
        ),
        "representative_correction": representative.label if representative is not None else None,
        "selected_corrections_csv": str(selected_corrections_path) if selected_corrections_path is not None else None,
        "combined_reference_period_hours": reference_period_hours,
        "results_csv": str(results_path),
        "power_plot": str(power_plot),
        "phase_plot": str(phase_plot),
        "light_curve_no_bias_csv": str(baseline_csv),
        "light_curve_no_bias_plot": str(baseline_png),
        "light_curve_bias_csv": str(corrected_csv),
        "light_curve_bias_plot": str(corrected_png),
    }
    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print("\nSelected-period summary:")
    for res in results:
        print(
            f"  {res.variant}: primary={res.primary_period_hours:.6f} h, "
            f"selected={res.selected_period_hours:.6f} h, power={res.selected_power:.6f}"
        )


if __name__ == "__main__":
    main()
