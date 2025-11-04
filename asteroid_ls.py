# asteroid_ls.py
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import pickle
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional

from astropy.timeseries import LombScargle
from scipy.signal import find_peaks
import pandas as pd
from matplotlib import colormaps

class AsteroidLSPipeline():
    """
    Reproduces your notebook's flow:

    - load all_sheets from pickle: {sheet_name: (H_s, time_s, ph_s)}
    - flatten to arrays t, y, ph
    - remove outliers on (t_rel, y) with IQR and "keep if x<5"
    - Lomb–Scargle with nterms, min/max period in HOURS
    - peak detection on power; pick best frequency
    - phase plot using ls.model

    Units/Conventions:
    - Time is assumed in *days*.
    - Period bounds are given in *hours* (as in your notebook).
      frequency [cycles/day] = 24 / period_hours
    """
    def __init__(self, all_sheets, Asteroid_number):
        self.all_sheets = all_sheets
        self.Asteroid_number = Asteroid_number

    def removeOutliers(self, xdatas, ydatas, outlierConstant, x_threshold=5):
        # Compute quartiles of the y–data
        Q1, Q3 = np.percentile(ydatas, [25, 75])
        IQR = Q3 - Q1
        
        # Inlier bounds in y
        lower_bound = Q1 - outlierConstant * IQR
        upper_bound = Q3 + outlierConstant * IQR
        
        # “Good by y” mask
        good_y = (ydatas >= lower_bound) & (ydatas <= upper_bound)
        # “Always keep” mask for x < threshold
        always_keep = xdatas < x_threshold
        
        # Final mask: either inlier by y *or* x below threshold
        mask = good_y | always_keep
        
        # Filtered data
        x_filtered = xdatas[mask]
        y_filtered = ydatas[mask]
        
        # Outlier indices (only those NOT in mask)
        removed_indices = np.where(~mask)[0]
        
        return x_filtered, y_filtered, removed_indices



    def reading_data(self, outlier_param = 1.8):
        plt.figure(dpi =300, figsize = (10, 10))
        H = []
        time = []
        ph =[]
        for sheet_name in self.all_sheets:
            print (sheet_name)
            H_s, time_s, ph_s = self.all_sheets[sheet_name]
            H.append(np.array(H_s))
            time.append(np.array(time_s))
            ph.append(np.array(ph_s))
            
            plt.scatter(ph_s, H_s, label = sheet_name)
        plt.legend()
        plt.xlabel("Phase")
        plt.ylabel("Mag")
        plt.tight_layout()
        #%%
        y = np.array([item for sublist in H for item in sublist])
        
        t = np.array([item for sublist in time for item in sublist])
        
        ph = np.array([item for sublist in ph for item in sublist])

        self.t = t
        self.y = y
        self.ph = ph
        #%%
        dy = 0.3
        
        
        t_rel = t-t[0]
        ts, ys, remove_idx= self.removeOutliers(t_rel, y, outlier_param)
        return ts, ys

    def LS_initiate(self, ts, ys, nterms= 2, maxf =1/(0.5/24), minf = 1/(50/24), samples_per_peak=15, save_bool = False, 
                    path_to_LS = r"C:\Users\nagai\Documents\LS_data\\"):
        self.nterms = nterms
        # Lomb scargle periodgramma
        ls = LombScargle(ts, ys, nterms=nterms)
        self.ls = ls
        # frequency, power = ls.autopower(minimum_frequency=0.5, maximum_frequency=20,
        #                                 samples_per_peak=10)
        
        # frekvenču spektrs 
        frequency, power = ls.autopower(minimum_frequency=minf, maximum_frequency= maxf,samples_per_peak=samples_per_peak)

        if save_bool:
            path_to_LS = r"C:\Users\nagai\Documents\LS_data\\"
            filename = path_to_LS + f"{self.Asteroid_number}_LS_results_n=2.pkl"
            
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
            "savefig.bbox": "tight"
        })
        
        # --- Data ---
        x_p = frequency
        y_p = power
        
        # Peaks
        peaks, _ = find_peaks(y_p, height=height, distance=distance)
        
        # Convert frequency → period (hours) and sort
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
            plt.savefig(out)
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
            "savefig.bbox": "tight"
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
            # Phase in cycles → degrees
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
        ax.set_xlabel("Phase (°)")
        ax.set_ylabel("Reduced magnitude")
        ax.set_xlim(0, 360)
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
            plt.savefig(out)
            print("Saved to:", out)
        plt.show()
        
                                
