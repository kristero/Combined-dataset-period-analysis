# asteroid_ls.py
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import pickle
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional

from astropy.timeseries import LombScargle
from astropy.time import Time
from scipy.signal import find_peaks
import pandas as pd
from matplotlib import colormaps
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
            t_rel = t - t[0]
            ts, ys, remove_idx = self.removeOutliers(t_rel, y, outlier_param)
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
            }

            opp_date = data.get("opposition_date_utc", "unknown")
            print(f"Opposition {gid} ({opp_date}): top periods (hours)")
            print(df_results.head(5))
            print(
                f"Opposition {gid}: selected period = {selected['selected_period_hours']:.4f} h; "
                f"reason: {selected['selection_reason']}"
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

    def visualize_ls_per_opposition(self, ls_results=None, peak_idx=0, save_fig=False, save_figures="path"):
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

            t_fit = np.linspace(0, 1, 500)
            y_fit = ls.model(t_fit / best_frequency, best_frequency)
            phase_fit = t_fit * 360.0

            fig = plt.figure(figsize=(3.35, 3.35), constrained_layout=True)
            ax = plt.gca()

            unique_sheets = list(dict.fromkeys(sheet.tolist()))
            cmap = colormaps['YlOrBr'].resampled(max(len(unique_sheets), 2))
            palette = cmap(np.arange(len(unique_sheets)))
            color_map = {name: palette[i] for i, name in enumerate(unique_sheets)}

            for i, name in enumerate(unique_sheets):
                mask = sheet == name
                phase = ((t[mask] - time_zero) * best_frequency) % 1.0
                phase_deg = phase * 360.0
                ax.errorbar(
                    np.asarray(phase_deg),
                    y[mask],
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
                loc="upper center",
                bbox_to_anchor=(0.5, 1.1),
            )
            fig.subplots_adjust(top=1)

            if save_fig:
                out = (
                    f"{save_figures}{self.Asteroid_number}_opp{gid}_phase_curve_"
                    f"P={24.0 / best_frequency:.3f}h_n={self.nterms}.pdf"
                )
                plt.savefig(out, dpi=600)
                print("Saved to:", out)
            plt.show()

    def visualize_time_series_all_oppositions(self, ls_results=None, save_fig=False, save_figures="path"):
        """
        Plot all opposition data in time domain (JD vs reduced magnitude)
        with LS fitted model per opposition overlaid.
        """
        ls_results = ls_results or getattr(self, "opposition_ls_results", None)
        if ls_results is None:
            raise ValueError("No per-opposition LS results found. Run LS_initiate_per_opposition first.")

        gids = sorted(ls_results.keys())
        cmap = colormaps["tab20"].resampled(max(len(gids), 2))
        colors = cmap(np.arange(len(gids)))

        fig = plt.figure(figsize=(11, 5), dpi=300)
        ax = plt.gca()

        for i, gid in enumerate(gids):
            res = ls_results[gid]
            ls = res["ls"]
            t = np.asarray(res["t"], dtype=float)
            y = np.asarray(res["y"], dtype=float)
            time_zero = float(res["time_zero"])
            f = float(res.get("selected_frequency", res["frequency"][np.argmax(res["power"])]))
            p = 24.0 / f
            opp_date = res.get("opposition_date_utc", "unknown")

            ax.scatter(
                t,
                y,
                s=12,
                color=colors[i],
                alpha=0.45,
                edgecolors="none",
                label=f"Opp {gid} data ({opp_date})",
                zorder=1,
            )

            t_line = np.linspace(float(np.min(t)), float(np.max(t)), 800)
            y_line = ls.model(t_line - time_zero, f)
            ax.plot(
                t_line,
                y_line,
                lw=1.2,
                color=colors[i],
                alpha=0.95,
                label=f"Opp {gid} fit: P={p:.3f} h",
                zorder=2,
            )

        ax.set_xlabel("Time (JD)")
        ax.set_ylabel("Reduced magnitude")
        ax.invert_yaxis()
        ax.grid(alpha=0.25, linestyle=":")
        ax.set_title("All oppositions: data and LS fits in time domain")
        ax.legend(frameon=False, fontsize=7, ncol=2)
        plt.tight_layout()

        if save_fig:
            out = f"{save_figures}{self.Asteroid_number}_all_oppositions_time_with_ls_fits.pdf"
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

    def find_peaks(self, frequency, power, height = 0.5, distance = 1000, save_fig = False, save_figues = "path"):
        
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
        x_p = frequency
        y_p = power
        
        # Peaks
        peaks, _ = find_peaks(y_p, height=height, distance=distance)
        
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
        
        ax.plot(period_sorted, power_sorted, lw=0.8, color="black", label="Power")
        ax.scatter(peak_period, peak_power, s=14, color="red", zorder=3,
                   label=f"Peaks > {height}")
        
        # Annotate peaks
        i = -1
        for P, Y in zip(peak_period, peak_power):
            ax.annotate(f"{P:.2f} h",
                        xy=(P, Y), xytext=(0, 4+7*i),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=7)
            i = i * (-1)
        
        ax.set_xlabel("Period (hours)")
        ax.set_ylabel("Power")
        
        ax.minorticks_on()
        ax.grid(which="both", axis="y", linestyle=":", linewidth=0.4, alpha=0.6)
        
        ax.legend(frameon=False, handlelength=1.2, borderpad=0.2, labelspacing=0.3)
        
        # Save as vector PDF (Elsevier prefers .pdf or .eps)

        if save_fig:
            out = f"{save_figures}{self.Asteroid_number}_power_spectrum_n={self.nterms}.pdf"
            plt.savefig(out, dpi=600)
            print("Saved to:", out)
        plt.show()
        return peaks

    def visualize_ls(self, frequency, peaks, peak_idx, save_fig = False, save_figures = "path"):
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
        
        time_zero = self.t[0]
        best_frequency = frequency[peaks][peak_idx]   # cycles per day
        # best_frequency = 1/(5.754/24)
        
        # Model (phase grid from 0..1); convert to degrees for plotting
        t_fit      = np.linspace(0, 1, 500)                         # phase (cycles)
        y_fit      = self.ls.model(t_fit / best_frequency, best_frequency)
        phase_fit = t_fit * 360.0
        
        # Palette & markers
        markers = ['o','s','^','D','v','>','<','p','*','H','X','d','P','8',
                   '. ',',','1','2','3','4','+','x','|','_']
        N       = len(self.all_sheets)
        cmap    = colormaps['YlOrBr'].resampled(max(N, 2))  # safe if N==1
        palette = cmap(np.arange(N))
        
        # --- Plot ---
        fig = plt.figure(figsize=(3.35, 3.35), constrained_layout=True)
        ax  = plt.gca()
        
        for i, (sheet_name, (H_s, time_s, ph_s)) in enumerate(self.all_sheets.items()):
            # Phase in cycles  degrees
            phase = ((time_s - time_zero) * best_frequency) % 1.0
            phase_deg = phase * 360.0
        
            ax.errorbar(np.asarray(phase_deg), H_s, yerr=0.00,
                        fmt=markers[i % len(markers)], ms=3.8,
                        ecolor="black", elinewidth=0.6, capsize=1.5,
                        color=palette[i],
                        alpha=0.9, label=sheet_name, zorder=1)
        
        # Overplot model (dashed black)
        ax.plot(phase_fit, y_fit, "--", lw=0.8, color="black",
                label=f"Period: {24.0/best_frequency:.3f} h", zorder=2)
        
        # Axes & grid
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
            ncol=3,                     # adjust for your number of labels
            loc="upper center",
            bbox_to_anchor=(0.5, 1.1))  # move legend a bit above the axes
        # 1.22
        # Make space on top so legend fits without overlapping
        fig.subplots_adjust(top=1)
        #plt.ylim(12.9, 14.5)
        
        # Save as vector PDF for LaTeX/Overleaf
        if save_fig:
            out = (f"{save_figures}{self.Asteroid_number}_phase_curve_"
                   f"P={24.0/best_frequency:.3f}h_n={self.nterms}.pdf")
            plt.savefig(out, dpi=600)
            print("Saved to:", out)
        plt.show()
        
                                

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
        3) Use cleaned observations to compute min, max, and peak-to-peak amplitude.
        """
        if not hasattr(self, "ls"):
            raise AttributeError("Lomb-Scargle model not found. Run LS_initiate(...) first.")
        if not hasattr(self, "t"):
            raise AttributeError("Time series not found. Run reading_data(...) first.")

        best_frequency = frequency[peaks][peak_idx]
        time_zero = self.t[0]

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

        min_idx_local = np.argmin(y_clean)
        max_idx_local = np.argmax(y_clean)
        min_mag = float(y_clean[min_idx_local])
        max_mag = float(y_clean[max_idx_local])
        amplitude = float(max_mag - min_mag)

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
            "min_phase_deg": float(phase_clean[min_idx_local]),
            "max_phase_deg": float(phase_clean[max_idx_local]),
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
            label=f"Rejected ({sigma_clip:g} sigma)",
            zorder=1,
        )
        ax.scatter(
            phase_clean,
            y_clean,
            s=10,
            color="black",
            alpha=0.85,
            label=f"Cleaned points (N={results['n_points_kept']})",
            zorder=2,
        )

        # Highlight min/max points used for amplitude.
        ax.scatter(
            [results["min_phase_deg"]],
            [results["min_mag"]],
            s=24,
            marker="v",
            color="tab:blue",
            label=f"Min mag: {results['min_mag']:.3f}",
            zorder=3,
        )
        ax.scatter(
            [results["max_phase_deg"]],
            [results["max_mag"]],
            s=24,
            marker="^",
            color="tab:red",
            label=f"Max mag: {results['max_mag']:.3f}",
            zorder=3,
        )

        ax.plot(
            phase_fit,
            y_fit,
            "--",
            lw=0.8,
            color="black",
            label=f"P={period_hours:.3f} h, Amp={results['amplitude']:.3f} mag",
            zorder=4,
        )

        ax.set_xlabel("Rotational phase (deg)")
        ax.set_ylabel("Reduced magnitude")
        ax.set_xlim(0, 360)
        ax.invert_yaxis()
        ax.minorticks_on()
        ax.grid(which="both", axis="y", linestyle=":", linewidth=0.4, alpha=0.6)
        ax.legend(frameon=False, fontsize=6, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.15))
        fig.subplots_adjust(top=1)

        if save_fig:
            out = (
                f"{save_figures}{self.Asteroid_number}_phase_curve_sigma_{sigma_clip}_amp_{results['amplitude']:.3f}_"
                f"P={period_hours:.3f}h_n={self.nterms}.pdf"
            )
            plt.savefig(out, dpi=600)
            print("Saved to:", out)

        plt.show()

        print(
            f"Amplitude summary: min={results['min_mag']:.4f}, "
            f"max={results['max_mag']:.4f}, amp={results['amplitude']:.4f} mag "
            f"(kept {results['n_points_kept']}/{results['n_points_total']} points)"
        )

        return results




