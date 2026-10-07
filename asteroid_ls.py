# asteroid_ls.py
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import pickle
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional

from astropy.timeseries import LombScargle
from astropy.time import Time
from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import uniform_filter1d
from scipy.optimize import curve_fit
import pandas as pd
from matplotlib import colormaps
from matplotlib.ticker import FuncFormatter, NullFormatter
import os
from astroquery.jplhorizons import Horizons

class AsteroidLSPipeline():
    """
    Reproduces your notebook's flow:

    - load all_sheets from pickle: {sheet_name: (H_s, time_s, ph_s)}
    - flatten to arrays t, y, ph
    - remove outliers on (t_rel, y) with IQR and "keep if x<5"
    - LombScargle with nterms, min/max period in HOURS
    - peak detection on power; pick best frequency
    - phase plot using ls.model

    Units/Conventions:
    - Time is assumed in *days*.
    - Period bounds are given in *hours* (as in your notebook).
      frequency [cycles/day] = 24 / period_hours
    """
    def __init__(self, all_sheets, Asteroid_number, base_dir=r"C:\Users\kn18001\Documents\Asteroids\Combined-dataset-period-analysis"):
        self.all_sheets = all_sheets
        self.Asteroid_number = Asteroid_number
        self.base_dir = base_dir

    def removeOutliers(self, xdatas, ydatas, outlierConstant, x_threshold=5):
        # Compute quartiles of the ydata
        Q1, Q3 = np.percentile(ydatas, [25, 75])
        IQR = Q3 - Q1
        
        # Inlier bounds in y
        lower_bound = Q1 - outlierConstant * IQR
        upper_bound = Q3 + outlierConstant * IQR
        
        # Good by y mask
        good_y = (ydatas >= lower_bound) & (ydatas <= upper_bound)
        # Always keep mask for x < threshold
        always_keep = xdatas < x_threshold
        
        # Final mask: either inlier by y *or* x below threshold
        mask = good_y | always_keep
        
        # Filtered data
        x_filtered = xdatas[mask]
        y_filtered = ydatas[mask]
        
        # Outlier indices (only those NOT in mask)
        removed_indices = np.where(~mask)[0]
        
        return x_filtered, y_filtered, removed_indices

    def _flatten_all_observations(self):
        """
        Flatten all observatory sheets into aligned arrays.
        """
        H_all = []
        t_all = []
        ph_all = []
        sheet_all = []

        for sheet_name, (H_s, time_s, ph_s) in self.all_sheets.items():
            H_arr = np.asarray(H_s, dtype=float)
            t_arr = np.asarray(time_s, dtype=float)
            ph_arr = np.asarray(ph_s, dtype=float)

            n = min(len(H_arr), len(t_arr), len(ph_arr))
            if n == 0:
                continue

            H_arr = H_arr[:n]
            t_arr = t_arr[:n]
            ph_arr = ph_arr[:n]
            sheet_arr = np.array([sheet_name] * n, dtype=object)

            finite = np.isfinite(H_arr) & np.isfinite(t_arr) & np.isfinite(ph_arr)
            if not np.any(finite):
                continue

            H_all.append(H_arr[finite])
            t_all.append(t_arr[finite])
            ph_all.append(ph_arr[finite])
            sheet_all.append(sheet_arr[finite])

        if not H_all:
            return np.array([]), np.array([]), np.array([]), np.array([], dtype=object)

        y = np.concatenate(H_all)
        t = np.concatenate(t_all)
        ph = np.concatenate(ph_all)
        sheet = np.concatenate(sheet_all)
        return t, y, ph, sheet

    def split_by_opposition(self, t, min_gap_days=120.0):
        """
        Assign opposition IDs using time gaps.
        A new opposition starts when sorted consecutive points are separated
        by more than min_gap_days.
        """
        t = np.asarray(t, dtype=float)
        if t.size == 0:
            return np.array([], dtype=int)

        order = np.argsort(t)
        t_sorted = t[order]
        gaps = np.diff(t_sorted)
        starts = np.where(gaps > float(min_gap_days))[0] + 1

        opp_sorted = np.zeros_like(t_sorted, dtype=int)
        start_idx = 0
        opp_id = 0
        for s in starts:
            opp_sorted[start_idx:s] = opp_id
            start_idx = s
            opp_id += 1
        opp_sorted[start_idx:] = opp_id

        opp = np.empty_like(opp_sorted)
        opp[order] = opp_sorted
        return opp

    def find_opposition_epochs_horizons(
        self,
        t=None,
        observer_code="500@399",
        step_days=1,
        min_elong_deg=170.0,
        min_separation_days=120.0,
    ):
        """
        Find opposition epochs from JPL Horizons ephemerides.

        Opposition dates are identified as local maxima of solar elongation
        (near 180 deg) for the asteroid as seen from Earth center.
        """
        if t is None:
            if not hasattr(self, "t"):
                raise ValueError("No time array found. Run reading_data(...) or pass t explicitly.")
            t = self.t

        t = np.asarray(t, dtype=float)
        if t.size == 0:
            raise ValueError("Empty time array; cannot determine opposition epochs.")

        start_jd = float(np.min(t)) - 40.0
        stop_jd = float(np.max(t)) + 40.0
        if stop_jd <= start_jd:
            raise ValueError("Invalid time range for Horizons query.")

        start_iso = Time(start_jd, format="jd").utc.iso.split(".")[0]
        stop_iso = Time(stop_jd, format="jd").utc.iso.split(".")[0]
        step_txt = f"{int(step_days)}d"

        obj_id = str(self.Asteroid_number)
        try:
            eph = Horizons(
                id=obj_id,
                location=observer_code,
                epochs={"start": start_iso, "stop": stop_iso, "step": step_txt},
            ).ephemerides()
        except Exception as exc:
            raise RuntimeError(
                f"Horizons ephemeris query failed for asteroid {obj_id}. "
                f"Check internet/access and object id. Details: {exc}"
            )

        jd = np.asarray(eph["datetime_jd"], dtype=float)
        elong = np.asarray(eph["elong"], dtype=float)  # deg
        finite = np.isfinite(jd) & np.isfinite(elong)
        jd = jd[finite]
        elong = elong[finite]
        if jd.size < 5:
            raise RuntimeError("Horizons returned too few ephemeris points for opposition detection.")

        min_dist_samples = max(1, int(np.ceil(float(min_separation_days) / float(step_days))))
        peaks, _ = find_peaks(elong, height=float(min_elong_deg), distance=min_dist_samples)
        if len(peaks) == 0:
            # Fall back to global/local prominent maxima if strict threshold misses.
            peaks, _ = find_peaks(elong, distance=min_dist_samples, prominence=1.0)
            if len(peaks) == 0:
                raise RuntimeError("Could not detect any opposition peaks from Horizons elongation.")

        opp_jd = jd[peaks]
        opp_elong = elong[peaks]
        order = np.argsort(opp_jd)
        opp_jd = opp_jd[order]
        opp_elong = opp_elong[order]

        print("Detected opposition epochs from Horizons:")
        for i, (ojd, oel) in enumerate(zip(opp_jd, opp_elong)):
            dt = Time(float(ojd), format="jd").utc.iso.split(".")[0]
            print(f"  Opp {i}: JD={ojd:.3f}, UTC={dt}, elong={oel:.2f} deg")

        return opp_jd

    def assign_opposition_ids(self, t, opposition_jd):
        """
        Assign each observation to an opposition interval delimited by midpoints
        between consecutive opposition epochs.
        """
        t = np.asarray(t, dtype=float)
        opposition_jd = np.asarray(opposition_jd, dtype=float)
        if t.size == 0:
            return np.array([], dtype=int)
        if opposition_jd.size == 0:
            raise ValueError("opposition_jd is empty.")

        opposition_jd = np.sort(opposition_jd)
        if opposition_jd.size == 1:
            return np.zeros_like(t, dtype=int)

        bounds = 0.5 * (opposition_jd[:-1] + opposition_jd[1:])
        opp_idx = np.digitize(t, bounds, right=False)
        return opp_idx.astype(int)

    def _count_model_extrema(self, ls, frequency, n_grid=800):
        """
        Count maxima/minima in one cycle of the LS model.
        """
        phase = np.linspace(0.0, 1.0, int(n_grid), endpoint=False)
        y = ls.model(phase / frequency, frequency)
        n_max = len(find_peaks(y)[0])
        n_min = len(find_peaks(-y)[0])
        return n_max, n_min

    def _select_period_with_double_peak_rule(
        self,
        ls,
        frequency,
        power,
        peak_idx=0,
        double_period_tol=0.12,
        search_top_n=20,
    ):
        """
        If top LS period is single-peaked (1 max + 1 min) and a candidate near
        2x period exists among next peaks, use that second-best period.
        """
        order = np.argsort(power)[::-1]
        if len(order) == 0:
            raise ValueError("Empty LS spectrum.")

        chosen_rank = min(max(int(peak_idx), 0), len(order) - 1)
        chosen_idx = int(order[chosen_rank])
        primary_idx = int(order[0])

        f1 = float(frequency[primary_idx])
        p1 = 24.0 / f1
        n_max, n_min = self._count_model_extrema(ls, f1)
        is_single_peaked = (n_max <= 1) and (n_min <= 1)

        reason = "default rank selection"
        switched = False
        if chosen_rank == 0 and is_single_peaked:
            limit = min(len(order), int(search_top_n))
            for cand in order[1:limit]:
                f2 = float(frequency[int(cand)])
                p2 = 24.0 / f2
                if np.abs(p2 - 2.0 * p1) / (2.0 * p1) <= float(double_period_tol):
                    chosen_idx = int(cand)
                    switched = True
                    reason = (
                        "top period is single-peaked; switched to candidate near 2x period "
                        f"(P1={p1:.4f} h, P2={p2:.4f} h)"
                    )
                    break
            if not switched:
                reason = "top period is single-peaked; no ~2x candidate found"
        elif chosen_rank == 0:
            reason = f"top period kept (extrema count: n_max={n_max}, n_min={n_min})"

        return {
            "selected_index": int(chosen_idx),
            "selected_frequency": float(frequency[chosen_idx]),
            "selected_period_hours": float(24.0 / float(frequency[chosen_idx])),
            "primary_frequency": f1,
            "primary_period_hours": float(p1),
            "primary_n_max": int(n_max),
            "primary_n_min": int(n_min),
            "used_double_period_rule": bool(switched),
            "selection_reason": reason,
        }

    def _calibrate_opposition_series(
        self,
        t,
        y,
        ls,
        frequency,
        time_zero,
        calibrate_mag=True,
        calibrate_phase=True,
        mag_stat="median",
        phase_anchor="minimum",
    ):
        """
        Calibrate one opposition independently:
        - magnitude zero-point offset
        - rotational phase shift
        """
        t = np.asarray(t, dtype=float)
        y = np.asarray(y, dtype=float)
        t_rel = t - float(time_zero)

        y_model = ls.model(t_rel, float(frequency))
        residual = y - y_model
        if calibrate_mag:
            if str(mag_stat).lower() == "mean":
                mag_offset = float(np.mean(residual))
            else:
                mag_offset = float(np.median(residual))
        else:
            mag_offset = 0.0
        y_cal = y - mag_offset

        if calibrate_phase:
            phase_grid = np.linspace(0.0, 1.0, 1000, endpoint=False)
            y_grid = ls.model(phase_grid / float(frequency), float(frequency))
            if str(phase_anchor).lower() in ("max", "maximum"):
                anchor_idx = int(np.argmax(y_grid))
            else:
                anchor_idx = int(np.argmin(y_grid))
            phase_shift_cycles = float(phase_grid[anchor_idx])
        else:
            phase_shift_cycles = 0.0

        return {
            "y_calibrated": y_cal,
            "mag_offset": float(mag_offset),
            "phase_shift_cycles": float(phase_shift_cycles),
            "phase_shift_deg": float(phase_shift_cycles * 360.0),
        }



    def reading_data(
        self,
        outlier_param=1.8,
        split_oppositions=False,
        min_gap_days=120.0,
        opposition_method="horizons",
        observer_code="500@399",
        horizons_step_days=1,
        min_obs_per_opposition=50,
    ):
        plt.figure(dpi =300, figsize = (10, 10))
        for sheet_name in self.all_sheets:
            print (sheet_name)
            H_s, time_s, ph_s = self.all_sheets[sheet_name]
            plt.scatter(ph_s, H_s, label = sheet_name)
        plt.legend()
        plt.xlabel("Phase")
        plt.ylabel("Mag")
        plt.tight_layout()
        #%%
        t, y, ph, sheet = self._flatten_all_observations()

        self.t = t
        self.y = y
        self.ph = ph
        self.sheet = sheet
        #%%
        if not split_oppositions:
            # Zero-phase epoch: the earliest observation of the combined dataset.
            # Stored so that folded light curves can report the epoch explicitly.
            self.time_zero = float(np.min(t))
            t_rel = t - self.time_zero
            ts, ys, remove_idx = self.removeOutliers(t_rel, y, outlier_param)
            self.remove_idx = remove_idx
            return ts, ys

        if opposition_method == "horizons":
            try:
                opp_epochs = self.find_opposition_epochs_horizons(
                    t=t,
                    observer_code=observer_code,
                    step_days=horizons_step_days,
                    min_elong_deg=170.0,
                    min_separation_days=min_gap_days,
                )
                opp_idx = self.assign_opposition_ids(t, opp_epochs)
                self.opposition_epochs_jd = opp_epochs
            except Exception as exc:
                print(
                    "Horizons-based opposition detection failed; "
                    f"falling back to time-gap splitting. Details: {exc}"
                )
                opp_idx = self.split_by_opposition(t, min_gap_days=min_gap_days)
                self.opposition_epochs_jd = None
        elif opposition_method == "gap":
            opp_idx = self.split_by_opposition(t, min_gap_days=min_gap_days)
            self.opposition_epochs_jd = None
        else:
            raise ValueError("opposition_method must be 'horizons' or 'gap'.")

        opposition_data = {}

        for gid in np.unique(opp_idx):
            in_opp = opp_idx == gid
            t_opp = t[in_opp]
            y_opp = y[in_opp]
            ph_opp = ph[in_opp]
            sheet_opp = sheet[in_opp]
            if getattr(self, "opposition_epochs_jd", None) is not None and int(gid) < len(self.opposition_epochs_jd):
                opp_jd = float(self.opposition_epochs_jd[int(gid)])
            else:
                opp_jd = float(np.median(t_opp))
            opp_date_utc = Time(opp_jd, format="jd").utc.iso.split(".")[0]

            n_total = int(len(y_opp))
            print(f"Opposition {gid}: date={opp_date_utc}, total observations={n_total}")
            if n_total < int(min_obs_per_opposition):
                print(
                    f"Opposition {gid}: discarded "
                    f"(n={n_total} < {int(min_obs_per_opposition)})"
                )
                continue

            t_rel_opp = t_opp - np.min(t_opp)
            ts_opp, ys_opp, remove_idx_opp = self.removeOutliers(t_rel_opp, y_opp, outlier_param)

            opposition_data[int(gid)] = {
                "t": t_opp,
                "y": y_opp,
                "ph": ph_opp,
                "sheet": sheet_opp,
                "t_rel": t_rel_opp,
                "ts": ts_opp,
                "ys": ys_opp,
                "remove_idx": remove_idx_opp,
                "time_zero": float(np.min(t_opp)),
                "n_total": n_total,
                "n_kept": int(len(ys_opp)),
                "opposition_jd": opp_jd,
                "opposition_date_utc": opp_date_utc,
            }

            print(
                f"Opposition {gid} ({opp_date_utc}): {len(y_opp)} points, kept {len(ys_opp)} "
                f"after outlier filtering"
            )

        if len(opposition_data) == 0:
            raise ValueError(
                "All oppositions were discarded by min_obs_per_opposition. "
                "Lower the threshold or inspect input data."
            )

        self.opposition_data = opposition_data
        return opposition_data

    def LS_initiate_per_opposition(
        self,
        opposition_data=None,
        nterms=2,
        maxf=1 / (0.5 / 24),
        minf=1 / (55 / 24),
        samples_per_peak=15,
        save_bool=False,
        path_to_LS=None,
        peak_idx=0,
        double_period_tol=0.12,
        calibrate_each_opposition=False,
        calibrate_mag=True,
        calibrate_phase=True,
        calibration_mag_stat="median",
        calibration_phase_anchor="minimum",
    ):
        """
        Run Lomb-Scargle separately for each opposition.
        """
        self.nterms = nterms
        opposition_data = opposition_data or getattr(self, "opposition_data", None)
        if opposition_data is None:
            raise ValueError("No opposition data found. Run reading_data(..., split_oppositions=True).")

        all_results = {}
        for gid, data in opposition_data.items():
            ts = np.asarray(data["ts"], dtype=float)
            ys = np.asarray(data["ys"], dtype=float)
            if len(ts) < 5:
                print(f"Opposition {gid}: skipped (not enough points for LS).")
                continue

            ls = LombScargle(ts, ys, nterms=nterms)
            frequency, power = ls.autopower(
                minimum_frequency=minf,
                maximum_frequency=maxf,
                samples_per_peak=samples_per_peak,
            )

            df_results = pd.DataFrame({"Period": 1 / frequency * 24, "power": power})
            df_results = df_results.sort_values(by=["power"], ascending=False, ignore_index=True)

            selected = self._select_period_with_double_peak_rule(
                ls=ls,
                frequency=frequency,
                power=power,
                peak_idx=peak_idx,
                double_period_tol=double_period_tol,
            )
            calib = self._calibrate_opposition_series(
                t=np.asarray(data["t"], dtype=float),
                y=np.asarray(data["y"], dtype=float),
                ls=ls,
                frequency=selected["selected_frequency"],
                time_zero=data["time_zero"],
                calibrate_mag=calibrate_each_opposition and calibrate_mag,
                calibrate_phase=calibrate_each_opposition and calibrate_phase,
                mag_stat=calibration_mag_stat,
                phase_anchor=calibration_phase_anchor,
            )

            all_results[int(gid)] = {
                "ls": ls,
                "frequency": frequency,
                "power": power,
                "df_results": df_results,
                "time_zero": data["time_zero"],
                "t": np.asarray(data["t"], dtype=float),
                "y": np.asarray(data["y"], dtype=float),
                "sheet": np.asarray(data["sheet"], dtype=object),
                "opposition_jd": data.get("opposition_jd"),
                "opposition_date_utc": data.get("opposition_date_utc"),
                "selected_index": selected["selected_index"],
                "selected_frequency": selected["selected_frequency"],
                "selected_period_hours": selected["selected_period_hours"],
                "primary_period_hours": selected["primary_period_hours"],
                "primary_n_max": selected["primary_n_max"],
                "primary_n_min": selected["primary_n_min"],
                "used_double_period_rule": selected["used_double_period_rule"],
                "selection_reason": selected["selection_reason"],
                "calibrate_each_opposition": bool(calibrate_each_opposition),
                "y_calibrated": calib["y_calibrated"],
                "mag_offset": calib["mag_offset"],
                "phase_shift_cycles": calib["phase_shift_cycles"],
                "phase_shift_deg": calib["phase_shift_deg"],
            }

            opp_date = data.get("opposition_date_utc", "unknown")
            print(f"Opposition {gid} ({opp_date}): top periods (hours)")
            print(df_results.head(5))
            print(
                f"Opposition {gid}: selected period = {selected['selected_period_hours']:.4f} h; "
                f"reason: {selected['selection_reason']}"
            )
            if calibrate_each_opposition:
                print(
                    f"Opposition {gid}: calibration applied "
                    f"(mag offset={calib['mag_offset']:+.4f}, "
                    f"phase shift={calib['phase_shift_deg']:+.2f} deg)"
                )

            plt.figure(dpi=300, figsize=(7, 4))
            plt.plot(1 / frequency * 24, power, c="black")
            plt.xlabel("Period, hours")
            plt.ylabel("Power")
            plt.title(
                f"Opp {gid} ({opp_date}): LS, selected P={selected['selected_period_hours']:.3f} h"
            )
            plt.tight_layout()
            plt.show()

            if save_bool:
                path_to_LS = path_to_LS or os.path.join(self.base_dir, "LS_data")
                os.makedirs(path_to_LS, exist_ok=True)
                filename = os.path.join(
                    path_to_LS, f"{self.Asteroid_number}_opp{gid}_LS_results_n={nterms}.pkl"
                )
                with open(filename, "wb") as file:
                    pickle.dump((frequency, power), file)

        self.opposition_ls_results = all_results
        return all_results

    def visualize_ls_per_opposition(
        self,
        ls_results=None,
        peak_idx=0,
        use_calibrated=False,
        save_fig=False,
        save_figures="path",
    ):
        """
        Plot one composite (phase-folded) light curve per opposition.
        """
        ls_results = ls_results or getattr(self, "opposition_ls_results", None)
        if ls_results is None:
            raise ValueError("No per-opposition LS results found. Run LS_initiate_per_opposition first.")

        markers = ['o', 's', '^', 'D', 'v', '>', '<', 'p', '*', 'H', 'X', 'd', 'P', '8']

        for gid, res in ls_results.items():
            frequency = res["frequency"]
            power = res["power"]
            ls = res["ls"]
            t = res["t"]
            y = res["y"]
            sheet = res["sheet"]
            time_zero = res["time_zero"]

            if "selected_frequency" in res:
                best_frequency = float(res["selected_frequency"])
            else:
                order = np.argsort(power)[::-1]
                use_idx = order[min(max(int(peak_idx), 0), len(order) - 1)]
                best_frequency = float(frequency[use_idx])
            opp_date = res.get("opposition_date_utc", "unknown")
            phase_shift_cycles = float(res.get("phase_shift_cycles", 0.0)) if use_calibrated else 0.0
            y_use = np.asarray(res.get("y_calibrated", y), dtype=float) if use_calibrated else np.asarray(y, dtype=float)

            t_fit = np.linspace(0, 1, 500)
            y_fit = ls.model(((t_fit + phase_shift_cycles) % 1.0) / best_frequency, best_frequency)
            if use_calibrated:
                y_fit = y_fit - float(res.get("mag_offset", 0.0))
            phase_fit = t_fit * 360.0

            fig = plt.figure(figsize=(3.35, 3.35), constrained_layout=True)
            ax = plt.gca()

            unique_sheets = list(dict.fromkeys(sheet.tolist()))
            cmap = colormaps['YlOrBr'].resampled(max(len(unique_sheets), 2))
            palette = cmap(np.arange(len(unique_sheets)))
            color_map = {name: palette[i] for i, name in enumerate(unique_sheets)}

            for i, name in enumerate(unique_sheets):
                mask = sheet == name
                phase = (((t[mask] - time_zero) * best_frequency) - phase_shift_cycles) % 1.0
                phase_deg = phase * 360.0
                ax.errorbar(
                    np.asarray(phase_deg),
                    y_use[mask],
                    yerr=0.0,
                    fmt=markers[i % len(markers)],
                    ms=3.8,
                    ecolor="black",
                    elinewidth=0.6,
                    capsize=1.5,
                    color=color_map[name],
                    alpha=0.9,
                    label=name,
                    zorder=1,
                )

            ax.plot(
                phase_fit,
                y_fit,
                "--",
                lw=0.8,
                color="black",
                label=f"Opp {gid} ({opp_date}): P={24.0 / best_frequency:.3f} h",
                zorder=2,
            )

            ax.set_xlabel("Rotational phase (deg)")
            ax.set_ylabel("Reduced magnitude")
            ax.set_xlim(0, 360)
            ax.invert_yaxis()
            ax.minorticks_on()
            ax.grid(which="both", axis="y", linestyle=":", linewidth=0.4, alpha=0.6)
            ax.legend(
                frameon=False,
                handlelength=1.2,
                borderpad=0.2,
                labelspacing=0.2,
                fontsize=6,
                ncol=2,
                loc="lower center",
                bbox_to_anchor=(0.5, 1.01),
            )
            if save_fig:
                out = (
                    f"{save_figures}{self.Asteroid_number}_opp{gid}_phase_curve_"
                    f"P={24.0 / best_frequency:.3f}h_n={self.nterms}.pdf"
                )
                plt.savefig(out, dpi=600)
                print("Saved to:", out)
            plt.show()

    def visualize_time_series_all_oppositions(
        self,
        ls_results=None,
        best_period_hours=None,
        use_calibrated=False,
        n_harmonics=2,
        align_phase_offsets=True,
        phase_shift_grid=720,
        save_fig=False,
        save_figures="path",
    ):
        """
        Combined phase-folded plot for all oppositions using one user-supplied period.

        - X axis: rotational phase (deg)
        - Each opposition: different marker
        - Legend includes opposition observation date range
        - One global fit curve across all oppositions at fixed period
        """
        ls_results = ls_results or getattr(self, "opposition_ls_results", None)
        if ls_results is None:
            raise ValueError("No per-opposition LS results found. Run LS_initiate_per_opposition first.")
        if best_period_hours is None:
            period_pool = []
            for res in ls_results.values():
                p_sel = res.get("selected_period_hours", None)
                if p_sel is not None and np.isfinite(p_sel):
                    period_pool.append(float(p_sel))
            if len(period_pool) == 0:
                raise ValueError("best_period_hours is None and no selected periods found in ls_results.")
            best_period_hours = float(np.median(period_pool))
            print(f"best_period_hours not provided; using median selected period: {best_period_hours:.4f} h")
        if best_period_hours <= 0:
            raise ValueError("best_period_hours must be positive.")

        def _fit_fourier_phase_model(phase_cycles, y_vals, n_terms):
            phase_cycles = np.asarray(phase_cycles, dtype=float)
            y_vals = np.asarray(y_vals, dtype=float)
            cols = [np.ones_like(phase_cycles)]
            for k in range(1, int(n_terms) + 1):
                angle = 2.0 * np.pi * k * phase_cycles
                cols.append(np.sin(angle))
                cols.append(np.cos(angle))
            X = np.column_stack(cols)
            beta, _, _, _ = np.linalg.lstsq(X, y_vals, rcond=None)

            def _predict(phi):
                phi = np.asarray(phi, dtype=float)
                cols_phi = [np.ones_like(phi)]
                for kk in range(1, int(n_terms) + 1):
                    ang = 2.0 * np.pi * kk * phi
                    cols_phi.append(np.sin(ang))
                    cols_phi.append(np.cos(ang))
                X_phi = np.column_stack(cols_phi)
                return X_phi @ beta

            return _predict

        f_global = 24.0 / float(best_period_hours)
        gids = sorted(ls_results.keys())
        cmap = colormaps["tab20"].resampled(max(len(gids), 2))
        colors = cmap(np.arange(len(gids)))
        markers = ['o', 's', '^', 'D', 'v', '>', '<', 'p', '*', 'H', 'X', 'd', 'P', '8']

        fig = plt.figure(figsize=(11, 5), dpi=300)
        ax = plt.gca()

        # Use one shared epoch across all oppositions so phase folding is consistent.
        t0_global = min(float(np.min(np.asarray(res["t"], dtype=float))) for res in ls_results.values())

        # First pass: build phases and an initial global fit.
        per_opp = {}
        phase_all = []
        y_all = []

        for i, gid in enumerate(gids):
            res = ls_results[gid]
            t = np.asarray(res["t"], dtype=float)
            y_raw = np.asarray(res["y"], dtype=float)
            y_use = np.asarray(res.get("y_calibrated", y_raw), dtype=float) if use_calibrated else y_raw
            phase_shift_cycles = float(res.get("phase_shift_cycles", 0.0)) if use_calibrated else 0.0
            opp_date = res.get("opposition_date_utc", "unknown")

            phase = (((t - t0_global) * f_global) - phase_shift_cycles) % 1.0
            per_opp[int(gid)] = {
                "t": t,
                "y": y_use,
                "phase": phase,
                "opp_date": opp_date,
            }
            phase_all.append(phase)
            y_all.append(y_use)

        phase_all = np.concatenate(phase_all)
        y_all = np.concatenate(y_all)
        fit_predictor = _fit_fourier_phase_model(phase_all, y_all, n_harmonics)

        # Optional second pass: find per-opposition phase shifts that best align to global fit.
        if align_phase_offsets:
            shift_grid = np.linspace(0.0, 1.0, int(max(16, phase_shift_grid)), endpoint=False)
            phase_all_aligned = []
            y_all_aligned = []
            for gid in gids:
                phase_i = per_opp[int(gid)]["phase"]
                y_i = per_opp[int(gid)]["y"]
                errs = np.empty_like(shift_grid)
                for j, sh in enumerate(shift_grid):
                    model_i = fit_predictor((phase_i + sh) % 1.0)
                    errs[j] = np.mean((y_i - model_i) ** 2)
                best_shift = float(shift_grid[int(np.argmin(errs))])
                per_opp[int(gid)]["phase"] = (phase_i + best_shift) % 1.0
                per_opp[int(gid)]["alignment_shift_deg"] = best_shift * 360.0
                phase_all_aligned.append(per_opp[int(gid)]["phase"])
                y_all_aligned.append(y_i)

            phase_all = np.concatenate(phase_all_aligned)
            y_all = np.concatenate(y_all_aligned)
            fit_predictor = _fit_fourier_phase_model(phase_all, y_all, n_harmonics)

        # Plot aligned oppositions.
        for i, gid in enumerate(gids):
            d = per_opp[int(gid)]
            phase_deg = d["phase"] * 360.0
            y_use = d["y"]
            opp_date = d["opp_date"]
            t = d["t"]
            t_start = Time(float(np.min(t)), format="jd").utc.iso.split(" ")[0]
            t_stop = Time(float(np.max(t)), format="jd").utc.iso.split(" ")[0]
            obs_window = f"{t_start} to {t_stop}"
            shift_txt = ""
            if align_phase_offsets:
                shift_txt = f", dphi={d.get('alignment_shift_deg', 0.0):.1f} deg"

            ax.scatter(
                phase_deg,
                y_use,
                s=16,
                marker=markers[i % len(markers)],
                color=colors[i],
                alpha=0.65,
                edgecolors="none",
                label=f"{t_start[:7]} to {t_stop[:7]} ({len(y_use)})",
                zorder=1,
            )

        phase_grid = np.linspace(0.0, 1.0, 800)
        y_grid = fit_predictor(phase_grid)
        ax.plot(
            phase_grid * 360.0,
            y_grid,
            "--",
            color="black",
            lw=1.5,
            label=f"Common model, P = {float(best_period_hours):.3f} h",
            zorder=3,
        )

        ax.set_xlabel("Rotational phase (deg)")
        ax.set_ylabel("Reduced magnitude")
        ax.set_xlim(0, 360)
        ax.invert_yaxis()
        ax.grid(alpha=0.25, linestyle=":")
        ax.legend(frameon=False, fontsize=7, ncol=3, loc="lower center")
        plt.tight_layout()

        if save_fig:
            out = (
                f"{save_figures}{self.Asteroid_number}_all_oppositions_phase_"
                f"P={float(best_period_hours):.4f}h_global_fit.pdf"
            )
            plt.savefig(out, dpi=600)
            print("Saved to:", out)
        plt.show()

    def LS_initiate(self, ts, ys, nterms= 2, maxf =1/(0.5/24), minf = 1/(50/24), samples_per_peak=15, save_bool = False,
                    path_to_LS = None):
        self.nterms = nterms
        if isinstance(save_bool, str):
            save_bool = save_bool.strip().lower() in ("true", "1", "yes", "y")
        # Lomb scargle periodgramma
        ls = LombScargle(ts, ys, nterms=nterms)
        self.ls = ls
        # frequency, power = ls.autopower(minimum_frequency=0.5, maximum_frequency=20,
        #                                 samples_per_peak=10)
        
        # frekvenu spektrs 
        frequency, power = ls.autopower(minimum_frequency=minf, maximum_frequency= maxf,samples_per_peak=samples_per_peak)

        if save_bool:
            path_to_LS = path_to_LS or os.path.join(self.base_dir, "LS_data")
            os.makedirs(path_to_LS, exist_ok=True)
            filename = os.path.join(path_to_LS, f"{self.Asteroid_number}_LS_results_n=2.pkl")
            
            # Save
            with open(filename, "wb") as file:
                pickle.dump((frequency, power), file)


        # Plotting to see what is the threshold
        plt.figure(dpi = 300, figsize = (10,10))
    
        plt.plot(1/frequency*24, power, c="black")
        # plt.scatter(1/(database_period/24), 1)
        # plt.ylim((0,4))
        plt.xlabel(r"Period, hours")
        plt.ylabel("Power")
        plt.show()

                # Making a dataframe so it is easier to manipulate with data
        df_results = pd.DataFrame({"Period": 1/frequency*24, "power": power})
        df_results = df_results.sort_values(
            by=["power"], ascending=False, ignore_index=True)
        # %%
        print ("TOP20 results from the power specrum")
        print (df_results[:20])
        return frequency, power, df_results

    def find_peaks(self, frequency, power, height=None, distance=1000, save_fig=False, number_of_sigma = 3, save_figures="path",
                   period_range=None, filename_suffix="", max_plot_points=400000,
                   xscale="linear", merge_frac=0.02, max_peaks=10):
        """
        Detect and plot significant periodogram peaks.

        period_range : (min_hours, max_hours) or None
            Restrict peak detection and plotting to this period interval.
        filename_suffix : str
            Appended to the output file name (before .pdf).
        max_plot_points : int
            Spectra longer than this are decimated (block min/max) for plotting only.
        xscale : "linear" or "log"
            Period axis scale ("log" is readable over the full 0.5-240 h search).
        merge_frac : float or None
            Alias side-lobes: of all peaks above threshold lying within +/- merge_frac
            (relative, in period) of a stronger peak, only the strongest is kept.
        max_peaks : int or None
            Keep at most this many (strongest) peaks.

        Returns indices of the retained peaks in the *full* frequency array.
        The threshold used is stored in self.last_peak_threshold.
        """

        # --- Figure & font setup for Elsevier ---
        plt.rcParams.update({
            "font.family": "serif",     # Elsevier uses serif in figures
            "font.size": 8,             # main font size (8 pt for single-column)
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.6,
            "xtick.major.size": 3, "ytick.major.size": 3,
            "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
            "savefig.bbox": "tight",
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42
        })
        
        # --- Data ---
        frequency = np.asarray(frequency, dtype=float)
        power = np.asarray(power, dtype=float)
        if period_range is not None:
            p_all = 24.0 / frequency
            in_range = (p_all >= float(min(period_range))) & (p_all <= float(max(period_range)))
            idx_map = np.where(in_range)[0]
            if idx_map.size < 10:
                raise ValueError("period_range selects too few periodogram points.")
        else:
            idx_map = np.arange(len(frequency))
        x_p = frequency[idx_map]
        y_p = power[idx_map]

        # Automatic noise-based threshold:
        # 1) detect candidate peaks
        # 2) remove their local Gaussian-like profiles from the noise estimate
        # 3) smooth with a large kernel
        # 4) compute threshold = background level + 3 * sigma(background residual)
        cand_peaks, _ = find_peaks(y_p, distance=distance)
        y_bg = np.array(y_p, dtype=float)
        if cand_peaks.size > 0:
            widths = peak_widths(y_p, cand_peaks, rel_height=0.5)[0]
            mask = np.zeros_like(y_bg, dtype=bool)
            for p_idx, w in zip(cand_peaks, widths):
                half_span = max(2, int(np.ceil(1.5 * w)))
                lo = max(0, int(p_idx) - half_span)
                hi = min(len(y_bg), int(p_idx) + half_span + 1)
                mask[lo:hi] = True
            y_bg[mask] = np.nan

            valid = np.isfinite(y_bg)
            if np.count_nonzero(valid) >= 2:
                x_all = np.arange(len(y_bg), dtype=float)
                y_bg[~valid] = np.interp(x_all[~valid], x_all[valid], y_bg[valid])
            else:
                y_bg = np.array(y_p, dtype=float)

        kernel_size = max(51, int(len(y_p) // 20))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = min(kernel_size, len(y_p) if len(y_p) % 2 == 1 else max(1, len(y_p) - 1))
        if kernel_size < 3:
            kernel_size = min(3, len(y_p))
        # Running mean with an O(N) filter (np.convolve is O(N*K) and far too slow
        # for the multi-million-point grids of the full 0.5-240 h search).
        y_bg_smooth = uniform_filter1d(y_bg, size=int(kernel_size), mode="nearest")

        bg_level = float(np.nanmedian(y_bg_smooth))
        bg_sigma = float(np.nanstd(y_bg - y_bg_smooth))
        auto_threshold = bg_level + number_of_sigma * bg_sigma

        # keep optional manual override for compatibility
        threshold = float(height) if height is not None else auto_threshold
        print(f"Background noise sigma = {bg_sigma:.6f}")
        print(f"Background level = {bg_level:.6f}")
        print(f"Peaks > {threshold:.6f}")
        self.last_peak_threshold = float(threshold)
        peaks, _ = find_peaks(y_p, height=threshold, distance=distance)
        n_raw = len(peaks)
        # Alias side-lobes (e.g. yearly aliases around a strong peak) are merged:
        # keep only the strongest peak within +/- merge_frac in period, at most max_peaks.
        if len(peaks) > 0 and (merge_frac is not None or max_peaks is not None):
            order_pk = peaks[np.argsort(y_p[peaks])[::-1]]
            accepted = []
            for i in order_pk:
                P_i = 24.0 / x_p[i]
                if merge_frac is not None and any(
                    abs(P_i - 24.0 / x_p[j]) <= float(merge_frac) * P_i for j in accepted
                ):
                    continue
                accepted.append(int(i))
                if max_peaks is not None and len(accepted) >= int(max_peaks):
                    break
            peaks = np.array(sorted(accepted), dtype=int)
        print(f"Peaks above threshold: {n_raw}; retained after merging side-lobes: {len(peaks)}")
        for i in peaks[np.argsort(y_p[peaks])[::-1]]:
            print(f"   P = {24.0 / x_p[i]:.4f} h   power = {y_p[i]:.4f}")
        # `peaks` indexes the (possibly range-restricted) local arrays and is used
        # for plotting; `peaks_full` indexes the full frequency array (returned).
        peaks_full = idx_map[peaks]

        # Convert frequency  period (hours) and sort
        period = 24 / x_p
        order = np.argsort(period)
        period_sorted = period[order]
        power_sorted  = y_p[order]
        
        peak_period = 24 / x_p[peaks]
        peak_power  = y_p[peaks]
        
        # --- Plot ---
        fig = plt.figure(figsize=(3.35, 2.2), constrained_layout=True)  # 3.35 in = 85 mm
        ax = plt.gca()
        
        # Decimate very long spectra for plotting only (keep block min and max so
        # the visual envelope is preserved).
        if len(period_sorted) > int(max_plot_points):
            n_blocks = max(1, int(max_plot_points) // 2)
            edges = np.linspace(0, len(period_sorted), n_blocks + 1).astype(int)
            sel = []
            for a, b in zip(edges[:-1], edges[1:]):
                if b <= a:
                    continue
                seg = power_sorted[a:b]
                i_max = a + int(np.argmax(seg))
                i_min = a + int(np.argmin(seg))
                sel.append(min(i_min, i_max))
                if i_min != i_max:
                    sel.append(max(i_min, i_max))
            sel = np.asarray(sel, dtype=int)
            plot_period, plot_power = period_sorted[sel], power_sorted[sel]
        else:
            plot_period, plot_power = period_sorted, power_sorted

        thr_txt = f"{threshold:.2f}" if threshold >= 0.1 else f"{threshold:.3f}"
        ax.plot(plot_period, plot_power, lw=0.8, color="black", label="Power")
        ax.scatter(peak_period, peak_power, s=14, color="red", zorder=3,
                   label=f"Peaks > {thr_txt}")
        # Legend above the frame so that it never competes with peak labels.
        ax.legend(frameon=False, handlelength=1.2, borderpad=0.2, labelspacing=0.3,
                  ncol=2, loc="lower right", bbox_to_anchor=(1.0, 1.01))

        # Head-room above the strongest peak so that labels can stack without clipping.
        ax.set_ylim(top=float(np.max(y_p)) * 1.40)

        # Axis scale must be fixed before the annotation layout below.
        if str(xscale).lower() == "log":
            ax.set_xscale("log")
            lo, hi = float(np.min(period_sorted)), float(np.max(period_sorted))
            ax.set_xlim(lo, hi)
            ticks = [tv for tv in (0.5, 1, 2, 5, 10, 20, 50, 100, 240) if lo <= tv <= hi]
            ax.set_xticks(ticks)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.xaxis.set_minor_formatter(NullFormatter())
        elif period_range is not None:
            ax.set_xlim(float(min(period_range)), float(max(period_range)))

        # Annotate peaks: labels in two alternating rows above the peaks, spread
        # horizontally so they never overlap, connected to the peaks by leader lines.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        ax_bbox = ax.get_window_extent(renderer=renderer)
        self._place_peak_labels(ax, fig, renderer, ax_bbox, peak_period, peak_power)

        ax.set_xlabel("Rotational period (hours)")
        ax.set_ylabel("Power")
        
        ax.minorticks_on()
        ax.grid(which="both", axis="y", linestyle=":", linewidth=0.4, alpha=0.6)
        
        # Save as vector PDF (Elsevier prefers .pdf or .eps)

        if save_fig:
            out = f"{save_figures}{self.Asteroid_number}_power_spectrum_n={self.nterms}{filename_suffix}.pdf"
            plt.savefig(out, dpi=600)
            plt.savefig(out[:-4] + ".png", dpi=200)
            print("Saved to:", out)
        plt.show()
        return peaks_full


    @staticmethod
    def _place_peak_labels(ax, fig, renderer, ax_bbox, peak_period, peak_power,
                           rows=(0.975, 0.885), fontsize=7, pad_pt=4.0):
        """
        Place period labels in alternating rows (axes fraction) above the peaks.
        Labels in the same row are pushed apart so that they never overlap; a thin
        leader line connects each label to its peak.
        """
        if len(peak_period) == 0:
            return
        order = np.argsort(peak_period)
        labels = [f"{float(peak_period[i]):.2f} h" for i in order]
        # Peak x positions in axes fraction (works for log axes).
        disp = ax.transData.transform(np.column_stack([peak_period[order], peak_power[order]]))
        axf = ax.transAxes.inverted().transform(disp)
        x_peak = axf[:, 0]
        # Label widths in axes fraction (measured with a temporary text).
        widths = []
        for lab in labels:
            t = ax.text(0, 0, lab, fontsize=fontsize, transform=ax.transAxes)
            fig.canvas.draw()
            bb = t.get_window_extent(renderer=renderer)
            widths.append(bb.width / ax_bbox.width)
            t.remove()
        widths = np.asarray(widths)
        pad = pad_pt * fig.dpi / 72.0 / ax_bbox.width

        n_rows = len(rows)
        row_of = np.arange(len(labels)) % n_rows
        x_lab = x_peak.copy()
        for r in range(n_rows):
            idx = np.where(row_of == r)[0]
            # push apart from left to right, then pull back inside the axes
            for k in range(1, len(idx)):
                a, b = idx[k - 1], idx[k]
                min_dx = 0.5 * (widths[a] + widths[b]) + pad
                if x_lab[b] - x_lab[a] < min_dx:
                    x_lab[b] = x_lab[a] + min_dx
            if len(idx):
                last = idx[-1]
                excess = x_lab[last] + 0.5 * widths[last] + 0.01 - 1.0
                if excess > 0:
                    x_lab[idx] -= excess
                first = idx[0]
                deficit = 0.01 + 0.5 * widths[first] - x_lab[first]
                if deficit > 0:
                    x_lab[idx] += deficit

        for k, i in enumerate(order):
            ax.annotate(
                labels[k],
                xy=(float(peak_period[i]), float(peak_power[i])), xycoords="data",
                xytext=(float(x_lab[k]), float(rows[row_of[k]])), textcoords="axes fraction",
                ha="center", va="top", fontsize=fontsize, zorder=4, clip_on=False,
                arrowprops=dict(arrowstyle="-", lw=0.35, color="0.45", shrinkA=0.5, shrinkB=2.0),
            )

    def visualize_ls(
        self,
        frequency,
        peaks,
        peak_idx,
        save_fig=False,
        save_figures="path",
        sigma_clip=None,
        show_epoch=False,
        filename_suffix="",
        bin_deg=10.0,
        amp_sigma_clip=3.0,
        min_points_per_bin=5,
        save_png=True,
        show_bins=True,
        amplitude_box=None,
    ):
        """
        Folded light curve at the selected peak.

        show_bins : draw the phase-bin means (default True).
        amplitude_box : optional (A, A_err, label) shown instead of the binned amplitude; the dotted lines
            then mark the extremes of the Fourier model.

        The amplitude shown in the legend is measured from the observations, not
        from the Fourier model: residuals about the model are sigma-clipped
        (amp_sigma_clip), the surviving points are averaged in phase bins of
        bin_deg degrees, and A = max(bin mean) - min(bin mean) with the standard
        errors of those two bins propagated. The binned means are over-plotted.

        show_epoch : bool
            Print the zero-phase epoch inside the axes (off by default; the epoch is
            returned for the caption: phi = frac((t - t0) * f)).
        Returns a dict with period, epoch, amplitudes, point counts and the output file.
        """
        # --- Figure & font setup for Elsevier (same as your first plot) ---
        plt.rcParams.update({
            "font.family": "serif",
            "font.size": 8,            # main font size (8 pt for single-column)
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.6,
            "xtick.major.size": 3, "ytick.major.size": 3,
            "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
            "savefig.bbox": "tight",
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42
        })
        
        # --- Inputs you already have ---
        # t, ls, peak_x, all_sheets, save_figures, Asteroid_number, nterms

        # Zero-phase epoch = earliest observation (same epoch used to fit the LS model).
        time_zero = float(getattr(self, "time_zero", np.min(self.t)))
        best_frequency = frequency[peaks][peak_idx]   # cycles per day
        # best_frequency = 1/(5.754/24)

        # Optional sigma clipping against best-fit LS residuals.
        keep_mask = {}
        clipped_points = 0
        total_points = 0
        if sigma_clip is not None:
            for sheet_name, (H_s, time_s, _) in self.all_sheets.items():
                H_arr = np.asarray(H_s, dtype=float)
                t_arr = np.asarray(time_s, dtype=float)
                valid = np.isfinite(H_arr) & np.isfinite(t_arr)
                mask = np.zeros_like(H_arr, dtype=bool)
                if np.any(valid):
                    t_rel = t_arr[valid] - time_zero
                    y_model = self.ls.model(t_rel, best_frequency)
                    residuals = H_arr[valid] - y_model
                    center = np.median(residuals)
                    sigma = np.std(residuals, ddof=1) if residuals.size > 1 else np.nan
                    if np.isfinite(sigma) and sigma > 0:
                        mask_valid = np.abs(residuals - center) <= float(sigma_clip) * sigma
                    else:
                        mask_valid = np.ones_like(residuals, dtype=bool)
                    mask[valid] = mask_valid
                    total_points += int(np.sum(valid))
                    clipped_points += int(np.sum(valid) - np.sum(mask_valid))
                keep_mask[sheet_name] = mask

        # Model (phase grid from 0..1); convert to degrees for plotting
        t_fit      = np.linspace(0, 1, 500)                         # phase (cycles)
        y_fit      = self.ls.model(t_fit / best_frequency, best_frequency)
        phase_fit = t_fit * 360.0
        
        # Palette & markers
        markers = ['o','s','^','D','v','>','<','p','*','H','X','d','P','8',
                   '. ',',','1','2','3','4','+','x','|','_']
        N       = len(self.all_sheets)
        # Keep YlOrBr tones but avoid near-white colors on white background.
        cmap = plt.get_cmap("YlOrBr")
        if N <= 1:
            palette = np.asarray([cmap(0.65)])
        else:
            # Sample only the darker part of the map: yellow-orange -> red-brown.
            palette = cmap(np.linspace(0.45, 0.95, N))
        
        # --- Plot ---
        fig = plt.figure(figsize=(3.35, 3.35), constrained_layout=True)
        ax  = plt.gca()

        all_phase_deg, all_H, all_t_rel = [], [], []
        for i, (sheet_name, (H_s, time_s, ph_s)) in enumerate(self.all_sheets.items()):
            H_plot = np.asarray(H_s, dtype=float)
            t_plot = np.asarray(time_s, dtype=float)
            if sigma_clip is not None:
                mask = keep_mask.get(sheet_name, np.ones_like(H_plot, dtype=bool))
                H_plot = H_plot[mask]
                t_plot = t_plot[mask]

            # Phase in cycles  degrees
            phase = ((t_plot - time_zero) * best_frequency) % 1.0
            phase_deg = phase * 360.0

            ok = np.isfinite(H_plot) & np.isfinite(t_plot)
            all_phase_deg.append(phase_deg[ok])
            all_H.append(H_plot[ok])
            all_t_rel.append(t_plot[ok] - time_zero)

            ax.scatter(np.asarray(phase_deg), H_plot,
                       marker=markers[i % len(markers)], s=18,
                       color=palette[i], edgecolors="none",
                       alpha=0.9, label=sheet_name, zorder=1)

        # Peak-to-peak range of the fitted Fourier model (kept for reference only).
        model_amplitude = float(np.max(y_fit) - np.min(y_fit))

        # Observed amplitude: sigma-clip residuals about the model, bin in phase,
        # amplitude = highest bin mean - lowest bin mean.
        phase_all = np.concatenate(all_phase_deg)
        H_all = np.concatenate(all_H)
        t_rel_all = np.concatenate(all_t_rel)
        y_model_all = self.ls.model(t_rel_all, best_frequency)
        amp = self._binned_amplitude(phase_all, H_all, y_model_all, bin_deg=bin_deg,
                                     sigma_clip=amp_sigma_clip, min_points=min_points_per_bin)
        # n=6 Fourier fit at the same frequency, for comparison in the summary.
        try:
            keep_r = amp["keep_mask"]
            ls6 = LombScargle(t_rel_all[keep_r], H_all[keep_r], nterms=6)
            y6 = ls6.model(t_fit / best_frequency, best_frequency)
            amplitude_n6 = float(np.max(y6) - np.min(y6))
        except Exception:
            amplitude_n6 = np.nan

        ax.plot(phase_fit, y_fit, "--", lw=0.9, color="black",
                label=f"n={self.nterms} model", zorder=2)
        if show_bins:
            ax.errorbar(
                amp["bin_centers"], amp["bin_means"], yerr=amp["bin_sem"],
                fmt="s", ms=2.6, color="black", ecolor="black", elinewidth=0.5, capsize=1.2,
                zorder=4, label=f"{bin_deg:g}$^\\circ$ bin means",
            )
        if amplitude_box is None:
            for level in (amp["max_mag"], amp["min_mag"]):
                ax.axhline(level, ls=":", lw=0.6, color="0.35", zorder=3)
            box = (f"P = {24.0/best_frequency:.3f} h\n"
                   f"A = {amp['amplitude']:.2f} $\\pm$ {amp['amplitude_err']:.2f} mag\n"
                   f"({amp_sigma_clip:g}$\\sigma$-clipped {bin_deg:g}$^\\circ$ bins)")
        else:
            box = (f"P = {24.0/best_frequency:.3f} h\n"
                   f"A = {amplitude_box[0]:.2f} $\\pm$ {amplitude_box[1]:.2f} mag")
        # Period and amplitude, in a small box inside the axes.
        ax.text(
            0.02, 0.97, box,
            transform=ax.transAxes, fontsize=6.5, ha="left", va="top", zorder=6,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.5),
        )
        
        # Axes & grid
        ax.set_xlabel("Rotational phase (deg)")
        ax.set_ylabel("Reduced magnitude")
        ax.set_xlim(0, 360)
        ax.invert_yaxis()
        ax.minorticks_on()
        ax.grid(which="both", axis="y", linestyle=":", linewidth=0.4, alpha=0.6)
        
        # Legend sits fully above the axes: anchored by its *lower* edge at the top of
        # the frame, so it grows upwards however many observatories there are.
        ax.legend(
            frameon=False,
            handlelength=1.2,
            borderpad=0.2,
            labelspacing=0.2,
            fontsize=6,
            ncol=3,                     # adjust for your number of labels
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01))
        #plt.ylim(12.9, 14.5)

        epoch_utc = Time(time_zero, format="jd").utc.iso.split(".")[0]
        if show_epoch:
            ax.text(
                0.01, 0.02,
                f"Phase 0: JD {time_zero:.4f} ({epoch_utc} UTC)",
                transform=ax.transAxes, fontsize=6, ha="left", va="bottom", color="0.25",
                zorder=5,
            )

        # Save as vector PDF for LaTeX/Overleaf (+ PNG preview)
        out = None
        if save_fig:
            out = (f"{save_figures}{self.Asteroid_number}_phase_curve_"
                   f"P={24.0/best_frequency:.3f}h_n={self.nterms}{filename_suffix}.pdf")
            plt.savefig(out, dpi=600)
            if save_png:
                plt.savefig(out[:-4] + ".png", dpi=200)
            print("Saved to:", out)
        plt.show()

        print(f"Amplitude (binned, {amp_sigma_clip:g} sigma clipped): {amp['amplitude']:.3f} +/- {amp['amplitude_err']:.3f} mag "
              f"[max bin {amp['max_phase_deg']:.0f} deg, min bin {amp['min_phase_deg']:.0f} deg, "
              f"{amp['n_clipped']} of {amp['n_total']} points clipped]; model n={self.nterms}: {model_amplitude:.3f}; n=6 fit: {amplitude_n6:.3f}")

        n_plotted = int(sum(np.sum(m) for m in keep_mask.values())) if sigma_clip is not None else int(total_points_all(self.all_sheets))
        return {
            "period_hours": float(24.0 / best_frequency),
            "best_frequency": float(best_frequency),
            "time_zero_jd": float(time_zero),
            "time_zero_utc": epoch_utc,
            "caption_epoch": f"phase 0 at JD {time_zero:.2f}",
            "amplitude_mag": amp["amplitude"],
            "amplitude_err_mag": amp["amplitude_err"],
            "amplitude_max_phase_deg": amp["max_phase_deg"],
            "amplitude_min_phase_deg": amp["min_phase_deg"],
            "amplitude_bin_deg": float(bin_deg),
            "amplitude_sigma_clip": float(amp_sigma_clip),
            "amplitude_points_clipped": amp["n_clipped"],
            "model_amplitude_mag": model_amplitude,
            "amplitude_n6_fit_mag": amplitude_n6,
            "n_points_plotted": n_plotted,
            "n_points_clipped": int(clipped_points),
            "sigma_clip": sigma_clip,
            "file": out,
        }

                                

    @staticmethod
    def _binned_amplitude(phase_deg, y, y_model, bin_deg=10.0, sigma_clip=3.0, min_points=5, n_iter=5):
        """
        Amplitude from the observations: iterative sigma clipping of the residuals
        (y - y_model), then phase-binned means; A = max(mean) - min(mean).
        """
        phase_deg = np.asarray(phase_deg, dtype=float)
        y = np.asarray(y, dtype=float)
        resid = y - np.asarray(y_model, dtype=float)
        keep = np.isfinite(resid)
        for _ in range(int(n_iter)):
            centre = float(np.median(resid[keep]))
            sigma = float(np.std(resid[keep], ddof=1))
            if not np.isfinite(sigma) or sigma <= 0:
                break
            new_keep = np.isfinite(resid) & (np.abs(resid - centre) <= float(sigma_clip) * sigma)
            if np.array_equal(new_keep, keep):
                break
            keep = new_keep

        edges = np.arange(0.0, 360.0 + 1e-9, float(bin_deg))
        if edges[-1] < 360.0:
            edges = np.append(edges, 360.0)
        idx = np.clip(np.digitize(phase_deg[keep], edges, right=False) - 1, 0, len(edges) - 2)
        yk = y[keep]
        n_bins = len(edges) - 1
        means = np.full(n_bins, np.nan)
        sems = np.full(n_bins, np.nan)
        counts = np.zeros(n_bins, dtype=int)
        for b in range(n_bins):
            sel = yk[idx == b]
            counts[b] = len(sel)
            if len(sel) >= int(min_points):
                means[b] = np.mean(sel)
                sems[b] = np.std(sel, ddof=1) / np.sqrt(len(sel)) if len(sel) > 1 else 0.0
        centres = 0.5 * (edges[:-1] + edges[1:])
        valid = np.isfinite(means)
        if np.sum(valid) < 2:
            raise ValueError("Too few populated phase bins to measure the amplitude.")
        i_max = int(np.where(valid)[0][np.argmax(means[valid])])
        i_min = int(np.where(valid)[0][np.argmin(means[valid])])
        amplitude = float(means[i_max] - means[i_min])
        amplitude_err = float(np.sqrt(sems[i_max] ** 2 + sems[i_min] ** 2))
        return {
            "amplitude": amplitude, "amplitude_err": amplitude_err,
            "max_mag": float(means[i_max]), "min_mag": float(means[i_min]),
            "max_phase_deg": float(centres[i_max]), "min_phase_deg": float(centres[i_min]),
            "bin_centers": centres[valid], "bin_means": means[valid], "bin_sem": sems[valid],
            "bin_counts": counts[valid], "keep_mask": keep,
            "n_total": int(len(y)), "n_clipped": int(np.sum(~keep)),
        }

    def compute_amplitude_from_ls(
        self,
        frequency,
        peaks,
        peak_idx=0,
        phase_bin_deg=5.0,
        sigma_clip=3.0,
        min_points_per_bin=4,
    ):
        """
        Compute light-curve amplitude after phase-bin sigma clipping around LS model.

        Workflow:
        1) Fold all observations using best LS frequency.
        2) In each phase bin (default 5 deg), clip residuals (obs - model) at N sigma.
        3) Compute mean magnitude per phase bin and find bins with min/max mean.
        4) Use 3*sigma in those bins as bin errors and propagate to amplitude error.
        """
        if not hasattr(self, "ls"):
            raise AttributeError("Lomb-Scargle model not found. Run LS_initiate(...) first.")
        if not hasattr(self, "t"):
            raise AttributeError("Time series not found. Run reading_data(...) first.")

        best_frequency = frequency[peaks][peak_idx]
        time_zero = float(getattr(self, "time_zero", np.min(self.t)))

        # Flatten all observations from all sheets (same data source as visualize_ls).
        y_obs_parts = []
        t_obs_parts = []
        for H_s, time_s, _ in self.all_sheets.values():
            y_obs_parts.append(np.asarray(H_s, dtype=float))
            t_obs_parts.append(np.asarray(time_s, dtype=float))

        y_obs = np.concatenate(y_obs_parts)
        t_obs = np.concatenate(t_obs_parts)

        finite = np.isfinite(y_obs) & np.isfinite(t_obs)
        y_obs = y_obs[finite]
        t_obs = t_obs[finite]

        # Relative time (days) because LS was fitted using relative time in LS_initiate.
        t_rel = t_obs - time_zero
        phase = (t_rel * best_frequency) % 1.0
        phase_deg = phase * 360.0
        y_model = self.ls.model(t_rel, best_frequency)
        residuals = y_obs - y_model

        # Phase-bin sigma clipping around LS residuals.
        bin_edges = np.arange(0.0, 360.0 + phase_bin_deg, phase_bin_deg)
        if bin_edges[-1] < 360.0:
            bin_edges = np.append(bin_edges, 360.0)
        bin_idx = np.digitize(phase_deg, bin_edges, right=False) - 1

        keep = np.ones_like(y_obs, dtype=bool)
        n_bins = len(bin_edges) - 1
        for b in range(n_bins):
            idx = np.where(bin_idx == b)[0]
            if len(idx) < min_points_per_bin:
                continue
            r_bin = residuals[idx]
            center = np.median(r_bin)
            sigma = np.std(r_bin, ddof=1)
            if not np.isfinite(sigma) or sigma == 0:
                continue
            keep[idx] = np.abs(r_bin - center) <= sigma_clip * sigma

        phase_clean = phase_deg[keep]
        y_clean = y_obs[keep]
        y_model_clean = y_model[keep]

        if len(y_clean) == 0:
            raise ValueError("No points left after sigma clipping. Try larger sigma_clip.")

        # Bin cleaned data to identify extrema from average signal per rotational-phase bin.
        clean_bin_idx = np.digitize(phase_clean, bin_edges, right=False) - 1
        n_bins = len(bin_edges) - 1
        mean_per_bin = np.full(n_bins, np.nan, dtype=float)
        std_per_bin = np.full(n_bins, np.nan, dtype=float)
        count_per_bin = np.zeros(n_bins, dtype=int)
        center_per_bin = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        for b in range(n_bins):
            idx = np.where(clean_bin_idx == b)[0]
            count_per_bin[b] = len(idx)
            if len(idx) < min_points_per_bin:
                continue
            yb = y_clean[idx]
            mean_per_bin[b] = np.mean(yb)
            std_per_bin[b] = np.std(yb, ddof=1) if len(yb) > 1 else 0.0

        valid_bins = np.where(np.isfinite(mean_per_bin))[0]
        if len(valid_bins) < 2:
            raise ValueError(
                "Not enough populated phase bins after clipping to estimate amplitude. "
                "Try larger phase_bin_deg, lower min_points_per_bin, or larger sigma_clip."
            )

        min_bin = valid_bins[np.argmin(mean_per_bin[valid_bins])]
        max_bin = valid_bins[np.argmax(mean_per_bin[valid_bins])]

        min_mag = float(mean_per_bin[min_bin])
        max_mag = float(mean_per_bin[max_bin])
        min_phase_deg = float(center_per_bin[min_bin])
        max_phase_deg = float(center_per_bin[max_bin])

        min_err_3sigma = float(3.0 * std_per_bin[min_bin]) if np.isfinite(std_per_bin[min_bin]) else np.nan
        max_err_3sigma = float(3.0 * std_per_bin[max_bin]) if np.isfinite(std_per_bin[max_bin]) else np.nan
        if not np.isfinite(min_err_3sigma):
            min_err_3sigma = 0.0
        if not np.isfinite(max_err_3sigma):
            max_err_3sigma = 0.0

        amplitude = float(max_mag - min_mag)
        amplitude_err = float(np.sqrt(min_err_3sigma**2 + max_err_3sigma**2))

        return {
            "best_frequency": float(best_frequency),
            "period_hours": float(24.0 / best_frequency),
            "phase_deg": phase_deg,
            "y_obs": y_obs,
            "y_model": y_model,
            "keep_mask": keep,
            "phase_clean": phase_clean,
            "y_clean": y_clean,
            "y_model_clean": y_model_clean,
            "min_mag": min_mag,
            "max_mag": max_mag,
            "amplitude": amplitude,
            "amplitude_err": amplitude_err,
            "min_phase_deg": min_phase_deg,
            "max_phase_deg": max_phase_deg,
            "min_err_3sigma": min_err_3sigma,
            "max_err_3sigma": max_err_3sigma,
            "min_bin_count": int(count_per_bin[min_bin]),
            "max_bin_count": int(count_per_bin[max_bin]),
            "n_points_total": int(len(y_obs)),
            "n_points_kept": int(np.sum(keep)),
            "phase_bin_deg": float(phase_bin_deg),
            "sigma_clip": float(sigma_clip),
        }

    def visualize_ls_amplitude(
        self,
        frequency,
        peaks,
        peak_idx=0,
        phase_bin_deg=5.0,
        sigma_clip=3.0,
        min_points_per_bin=4,
        save_fig=False,
        save_figures="path",
    ):
        """
        Visualize LS fit with cleaned light curve and amplitude summary.
        """
        results = self.compute_amplitude_from_ls(
            frequency=frequency,
            peaks=peaks,
            peak_idx=peak_idx,
            phase_bin_deg=phase_bin_deg,
            sigma_clip=sigma_clip,
            min_points_per_bin=min_points_per_bin,
        )

        best_frequency = results["best_frequency"]
        period_hours = results["period_hours"]

        # Smooth model line over one cycle.
        t_fit = np.linspace(0, 1, 500)
        y_fit = self.ls.model(t_fit / best_frequency, best_frequency)
        phase_fit = t_fit * 360.0

        plt.rcParams.update({
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.6,
            "xtick.major.size": 3, "ytick.major.size": 3,
            "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
            "savefig.bbox": "tight",
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42
        })

        fig = plt.figure(figsize=(3.35, 3.35), constrained_layout=True)
        ax = plt.gca()

        keep = results["keep_mask"]
        phase_all = results["phase_deg"]
        y_all = results["y_obs"]
        phase_clean = results["phase_clean"]
        y_clean = results["y_clean"]

        # Rejected points in light gray, cleaned points in black.
        ax.scatter(
            phase_all[~keep],
            y_all[~keep],
            s=9,
            color="lightgray",
            alpha=0.7,
            zorder=1,
        )
        ax.scatter(
            phase_clean,
            y_clean,
            s=10,
            color="black",
            alpha=0.85,
            zorder=2,
        )

        # Highlight min/max phase bins (mean +/- 3 sigma).
        ax.errorbar(
            [results["min_phase_deg"]],
            [results["min_mag"]],
            yerr=[results["min_err_3sigma"]],
            fmt="v",
            markersize=5,
            color="tab:blue",
            ecolor="tab:blue",
            elinewidth=0.8,
            capsize=2,
            zorder=3,
        )
        ax.errorbar(
            [results["max_phase_deg"]],
            [results["max_mag"]],
            yerr=[results["max_err_3sigma"]],
            fmt="^",
            markersize=5,
            color="tab:red",
            ecolor="tab:red",
            elinewidth=0.8,
            capsize=2,
            zorder=3,
        )

        ax.plot(
            phase_fit,
            y_fit,
            "--",
            lw=0.8,
            color="red",
            label=(
                f"P={period_hours:.3f} h, "
                f"Amp={results['amplitude']:.3f} +/- {results['amplitude_err']:.3f} mag"
            ),
            zorder=4,
        )

        ax.set_xlabel("Rotational phase (deg)")
        ax.set_ylabel("Reduced magnitude")
        ax.set_xlim(0, 360)
        ax.invert_yaxis()
        ax.minorticks_on()
        ax.grid(which="both", axis="y", linestyle=":", linewidth=0.4, alpha=0.6)
        ax.legend(frameon=False, fontsize=7, loc="best")

        if save_fig:
            out = (
                f"{save_figures}{self.Asteroid_number}_phase_curve_sigma_{sigma_clip}_amp_{results['amplitude']:.3f}"
                f"_err_{results['amplitude_err']:.3f}_"
                f"P={period_hours:.3f}h_n={self.nterms}.pdf"
            )
            plt.savefig(out, dpi=600)
            print("Saved to:", out)

        plt.show()

        n_rejected = results["n_points_total"] - results["n_points_kept"]
        print(
            f"Sigma clip: {sigma_clip}; rejected={n_rejected}; kept={results['n_points_kept']}/{results['n_points_total']}"
        )
        print(
            f"Min bin mean: {results['min_mag']:.4f} +/- {results['min_err_3sigma']:.4f} mag (3sigma), "
            f"phase={results['min_phase_deg']:.1f} deg, N={results['min_bin_count']}"
        )
        print(
            f"Max bin mean: {results['max_mag']:.4f} +/- {results['max_err_3sigma']:.4f} mag (3sigma), "
            f"phase={results['max_phase_deg']:.1f} deg, N={results['max_bin_count']}"
        )
        print(
            f"Amplitude summary: min={results['min_mag']:.4f}, "
            f"max={results['max_mag']:.4f}, amp={results['amplitude']:.4f} +/- "
            f"{results['amplitude_err']:.4f} mag "
            f"(kept {results['n_points_kept']}/{results['n_points_total']} points)"
        )

        return results

    # ------------------------------------------------------------------
    # Diagnostics requested in the review: spectral window and peak shape
    # ------------------------------------------------------------------
    def spectral_window(
        self,
        ts=None,
        fmax=2.6,
        samples_per_peak=10,
        candidates_hours=(),
        harmonic=2,
        n_label_peaks=6,
        save_fig=False,
        save_figures="path",
        filename_suffix="",
    ):
        """
        Spectral window of the observation times, W(f) = |(1/N) sum exp(2 pi i f t)|^2.

        candidates_hours : iterable of candidate rotation periods (hours). For every
            pair the difference and sum of their light-curve modulation frequencies
            (harmonic * 24 / P cycles/day; harmonic=2 for a double-peaked curve) are
            marked, so that alias relations through the window can be checked.
        Returns a dict with the window, its main peaks and the candidate relations.
        """
        t = np.asarray(self.t if ts is None else ts, dtype=float)
        t = t[np.isfinite(t)]
        t = t - float(np.min(t))
        baseline = float(np.max(t))
        df = 1.0 / (float(samples_per_peak) * baseline)
        f = np.arange(df, float(fmax), df)

        W = np.empty_like(f)
        chunk = 1000
        for i in range(0, len(f), chunk):
            ph = 2.0 * np.pi * np.outer(f[i:i + chunk], t)
            W[i:i + chunk] = (np.abs(np.exp(1j * ph).sum(axis=1)) / len(t)) ** 2

        # Main window peaks (ignore the trivial f -> 0 lobe).
        min_dist = max(1, int(0.02 / df))
        pk, _ = find_peaks(W, distance=min_dist, height=0.02)
        pk = pk[np.argsort(W[pk])[::-1]][: int(n_label_peaks)]
        window_peaks = [(float(f[i]), float(W[i])) for i in sorted(pk)]

        # Candidate alias relations.
        relations = []
        cands = [float(p) for p in candidates_hours]
        for a in range(len(cands)):
            for b in range(a + 1, len(cands)):
                fa = float(harmonic) * 24.0 / cands[a]
                fb = float(harmonic) * 24.0 / cands[b]
                for kind, val in (("diff", abs(fa - fb)), ("sum", fa + fb)):
                    if 0 < val < float(fmax):
                        j = int(np.argmin(np.abs(f - val)))
                        relations.append({
                            "P1_hours": cands[a], "P2_hours": cands[b], "kind": kind,
                            "frequency_cpd": float(val), "window_power": float(W[j]),
                        })

        plt.rcParams.update({
            "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
            "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6,
            "axes.linewidth": 0.6, "savefig.bbox": "tight", "savefig.dpi": 600,
            "pdf.fonttype": 42, "ps.fonttype": 42,
        })
        fig = plt.figure(figsize=(3.35, 2.4), constrained_layout=True)
        ax = plt.gca()
        ax.plot(f, W, lw=0.6, color="black", label="Spectral window")
        for fp, wp in window_peaks:
            ax.annotate(f"{fp:.4f}", xy=(fp, wp), xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=6)
        colors = plt.get_cmap("YlOrBr")(np.linspace(0.45, 0.95, max(len(relations), 2)))
        for k, rel in enumerate(relations):
            sym = "-" if rel["kind"] == "diff" else "+"
            ax.axvline(rel["frequency_cpd"], color=colors[k], lw=0.9, ls="--",
                       label=(f"|f({rel['P1_hours']:.2f} h) {sym} f({rel['P2_hours']:.2f} h)|"
                              f" = {rel['frequency_cpd']:.4f} d$^{{-1}}$"))
        ax.set_xlabel("Frequency (cycles per day)")
        ax.set_ylabel("Window power")
        ax.set_xlim(0, float(fmax))
        ax.set_ylim(bottom=0)
        ax.minorticks_on()
        ax.grid(which="both", axis="y", linestyle=":", linewidth=0.4, alpha=0.6)
        # Legend below the axes so it never hides the window peaks.
        ax.legend(frameon=False, handlelength=1.2, borderpad=0.2, labelspacing=0.25,
                  loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=1)

        out = None
        if save_fig:
            out = f"{save_figures}{self.Asteroid_number}_spectral_window{filename_suffix}.pdf"
            plt.savefig(out, dpi=600)
            print("Saved to:", out)
        plt.show()

        print("Spectral window main peaks (cycles/day, power):")
        for fp, wp in window_peaks:
            print(f"  {fp:.5f}  {wp:.3f}")
        if relations:
            print("Candidate frequency relations (harmonic=%d):" % int(harmonic))
            for rel in relations:
                print(f"  {rel['kind']:4s} P1={rel['P1_hours']:.3f} h, P2={rel['P2_hours']:.3f} h -> "
                      f"{rel['frequency_cpd']:.5f} d^-1, window power there = {rel['window_power']:.3f}")
        return {"frequency": f, "window": W, "window_peaks": window_peaks,
                "relations": relations, "baseline_days": baseline, "file": out}

    def plot_peak_zoom_gaussian(
        self,
        frequency,
        power,
        period_hours,
        wide_frac=0.10,
        n_resolution=3.0,
        save_fig=False,
        save_figures="path",
        filename_suffix="",
    ):
        """
        Two-panel figure: (left) periodogram within +/- wide_frac of the peak,
        (right) zoom on the peak with a fitted Gaussian profile.

        The zoom half-width is n_resolution * dP, where dP = P^2 / (24 * baseline)
        is the period resolution set by the time baseline.
        Returns the Gaussian parameters (centre, sigma, FWHM in hours).
        """
        frequency = np.asarray(frequency, dtype=float)
        power = np.asarray(power, dtype=float)
        P = 24.0 / frequency
        near = np.abs(P - float(period_hours)) <= 0.005 * float(period_hours)
        if not np.any(near):
            raise ValueError("No periodogram points within 0.5% of the requested period.")
        i0 = int(np.where(near)[0][np.argmax(power[near])])
        P0 = float(P[i0])
        baseline = float(np.max(self.t) - np.min(self.t))
        dP = P0 ** 2 / 24.0 / baseline

        wide = np.abs(P - P0) <= float(wide_frac) * P0
        narrow = np.abs(P - P0) <= float(n_resolution) * dP

        def gauss(x, A, mu, sig, c):
            return A * np.exp(-0.5 * ((x - mu) / sig) ** 2) + c

        x_n, y_n = P[narrow], power[narrow]
        order = np.argsort(x_n)
        x_n, y_n = x_n[order], y_n[order]
        p0 = [float(power[i0]), P0, dP / 2.0, float(np.min(y_n))]
        popt, pcov = curve_fit(gauss, x_n, y_n, p0=p0, maxfev=20000)
        A, mu, sig, c = popt
        sig = abs(float(sig))
        fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sig
        mu_err = float(np.sqrt(np.abs(pcov[1, 1]))) if np.all(np.isfinite(pcov)) else np.nan

        plt.rcParams.update({
            "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
            "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
            "axes.linewidth": 0.6, "savefig.bbox": "tight", "savefig.dpi": 600,
            "pdf.fonttype": 42, "ps.fonttype": 42,
        })
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.5), constrained_layout=True)

        xw, yw = P[wide], power[wide]
        ow = np.argsort(xw)
        xw, yw = xw[ow], yw[ow]
        if len(xw) > 200000:
            step = int(np.ceil(len(xw) / 200000))
            nb = len(xw) // step
            xw_b = xw[: nb * step].reshape(nb, step)
            yw_b = yw[: nb * step].reshape(nb, step)
            xw, yw = xw_b.mean(axis=1), yw_b.max(axis=1)
        ax1.plot(xw, yw, lw=0.7, color="black", label="Power")
        ax1.scatter([P0], [power[i0]], s=16, color="red", zorder=3, label=f"Highest peak: {P0:.3f} h")
        ax1.axvspan(P0 - n_resolution * dP, P0 + n_resolution * dP, color="tab:red", alpha=0.25, lw=0,
                    label="Zoom region (right panel)")
        ax1.set_xlabel("Rotational period (hours)")
        ax1.set_ylabel("Power")
        ax1.set_xlim(P0 * (1 - wide_frac), P0 * (1 + wide_frac))
        ax1.minorticks_on()
        ax1.grid(which="both", axis="y", linestyle=":", linewidth=0.4, alpha=0.6)
        ax1.legend(frameon=False, loc="upper right", handlelength=1.2, borderpad=0.2)
        ax1.set_title("(a)", loc="left", fontsize=8)

        x_fit = np.linspace(x_n.min(), x_n.max(), 600)
        ax2.plot(x_n, y_n, lw=0.9, color="black", label="Power")
        ax2.scatter([P0], [power[i0]], s=16, color="red", zorder=3)
        ax2.plot(x_fit, gauss(x_fit, *popt), "--", lw=1.0, color="tab:red",
                 label=(f"Gaussian fit: $\\mu$ = {mu:.5f} h,\nFWHM = {fwhm:.5f} h "
                        f"({fwhm * 3600:.1f} s)"))
        ax2.set_xlabel("Rotational period (hours)")
        ax2.set_ylabel("Power")
        ax2.minorticks_on()
        ax2.grid(which="both", axis="y", linestyle=":", linewidth=0.4, alpha=0.6)
        ax2.legend(frameon=False, loc="upper right", handlelength=1.2, borderpad=0.2)
        ax2.set_title("(b)", loc="left", fontsize=8)
        ax2.ticklabel_format(axis="x", useOffset=False)

        out = None
        if save_fig:
            out = f"{save_figures}{self.Asteroid_number}_peak_gaussian_P={P0:.3f}h_n={self.nterms}{filename_suffix}.pdf"
            plt.savefig(out, dpi=600)
            plt.savefig(out[:-4] + ".png", dpi=200)
            print("Saved to:", out)
        plt.show()

        print(f"Gaussian fit: mu = {mu:.6f} +/- {mu_err:.2e} h, sigma = {sig:.3e} h, FWHM = {fwhm:.3e} h "
              f"(period resolution dP = {dP:.3e} h, baseline = {baseline:.1f} d)")
        return {"peak_period_hours": P0, "peak_power": float(power[i0]), "gauss_mu_hours": float(mu),
                "gauss_mu_err_hours": mu_err, "gauss_sigma_hours": sig, "gauss_fwhm_hours": float(fwhm),
                "period_resolution_hours": float(dP), "baseline_days": baseline, "file": out}


def total_points_all(all_sheets):
    """Number of finite (H, t) observations across all sheets."""
    n = 0
    for H_s, time_s, _ in all_sheets.values():
        H_arr = np.asarray(H_s, dtype=float)
        t_arr = np.asarray(time_s, dtype=float)
        n += int(np.sum(np.isfinite(H_arr) & np.isfinite(t_arr)))
    return n


# Backward-compatible alias for older notebooks/scripts.
LS_period = AsteroidLSPipeline
