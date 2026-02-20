import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
import astropy.units as u
import os
import pandas as pd
from astroquery.jplsbdb import SBDB
import pickle

import phunk
from sbpy import photometry as phot
import lmfit
from scipy.interpolate import CubicSpline
from scipy import stats

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pickle
from matplotlib import colormaps




class DatasetGenerator():
    """
    Applies the neccesary corrections to all the observatories, and then
    produces the combined dataset later to be used for the light curve analysis
    """

    def __init__(self, path, file_name, Asteroid_number, reduced_obs, base_dir=r"C:\Users\kn18001\Documents\Asteroids\Combined-dataset-period-analysis", file_path_ph=None):
        self.path = path
        self.file_name = file_name
        self.base_dir = base_dir
        self.file_path_ph = file_path_ph or os.path.join(self.base_dir, "phases_and_phi.txt")
        self.Asteroid_number = Asteroid_number
        self.reduced_obs = reduced_obs
        self.filter_bias = {"B": 0.11,
              "g": -0.325,
              "c": -0.017, 
              "V": 0.085,
              "w": 0.111,
              "r": 0.126,
               "R": 0.282,
               "G" : 0.154,
               "o": 0.325,
               "i": 0.334,
               "I": 0.246,
               "z": 0.287,
               "y": 0.336,
               "Y": 0.906,
               "J": 1.362,
               "H": 1.81,
               "K": 1.835,
               "-": -0.037,
               "u": -2.436,
               "C": 0.351
              }

    def removeOutliers(self, xdatas, ydatas, outlierConstant=3.0, x_threshold=5):
        """
        Remove outliers by sigma-clipping inside phase-angle bins.

        Parameters:
            xdatas: phase angles (deg)
            ydatas: magnitudes
            outlierConstant: sigma threshold (default 3.0)
            x_threshold: kept for backward compatibility (unused)
        """
        xdatas = np.asarray(xdatas, dtype=float)
        ydatas = np.asarray(ydatas, dtype=float)

        valid = np.isfinite(xdatas) & np.isfinite(ydatas)
        mask = np.zeros_like(valid, dtype=bool)

        if not np.any(valid):
            return xdatas[mask], ydatas[mask], np.arange(len(xdatas))

        sigma_clip = float(outlierConstant)
        low_phase_limit_deg = 7.0
        low_phase_sigma = 5.0
        phase_bin_deg = 5.0

        x_valid = xdatas[valid]
        y_valid = ydatas[valid]
        idx_valid = np.where(valid)[0]

        # Build 5-degree bins covering the full phase-angle range in data.
        x_min = np.floor(np.min(x_valid) / phase_bin_deg) * phase_bin_deg
        x_max = np.ceil(np.max(x_valid) / phase_bin_deg) * phase_bin_deg
        if x_max == x_min:
            x_max = x_min + phase_bin_deg
        edges = np.arange(x_min, x_max + phase_bin_deg, phase_bin_deg)
        bin_idx = np.digitize(x_valid, edges, right=False) - 1

        keep_valid = np.ones_like(y_valid, dtype=bool)
        n_bins = len(edges) - 1
        for b in range(n_bins):
            in_bin = np.where(bin_idx == b)[0]
            if len(in_bin) < 2:
                continue
            y_bin = y_valid[in_bin]
            mu = np.mean(y_bin)
            sigma = np.std(y_bin, ddof=1)
            if not np.isfinite(sigma) or sigma == 0:
                continue
            # For phase angles below 7 deg, apply a looser 5-sigma criterion.
            x_bin = x_valid[in_bin]
            thresholds = np.where(x_bin < low_phase_limit_deg, low_phase_sigma, sigma_clip)
            keep_valid[in_bin] = np.abs(y_bin - mu) <= thresholds * sigma

        mask[idx_valid] = keep_valid

        x_filtered = xdatas[mask]
        y_filtered = ydatas[mask]
        removed_indices = np.where(~mask)[0]
        return x_filtered, y_filtered, removed_indices

    #%%
    def spline(self):
        delimiter = None
        with open(self.file_path_ph, 'r') as file:
            lines = file.readlines()
    
        # Initialize an empty list to store columns
        columns = []
    
        for line in lines:
            # Split the line by the delimiter
            values = line.strip().split(delimiter)
    
            # Append values to corresponding column lists
            if not columns:
                columns = [[] for _ in values]  # Initialize columns on the first row
    
            for i, value in enumerate(values):
                columns[i].append(float(value))
        ph = np.array(columns[0] + columns[4] + columns[8])
        phi1 = np.array(columns[1] + columns[5] + columns[9])
        phi2 = np.array(columns[2] + columns[6] + columns[10])
        phi3 = np.array(columns[3] + columns[7] + columns[11])
        
        idx = np.where(ph == 8)
        
        ph = np.delete(ph, idx[0])
        phi1 = np.delete(phi1, idx[0])
        phi2 = np.delete(phi2, idx[0])
        phi3 = np.delete(phi3, idx[0])
        
        return CubicSpline(ph, phi1), CubicSpline(ph, phi2), CubicSpline(ph, phi3)
    
    #%%
    def hg1g2_phase_function(self, alpha, H, G1, G2):
        """
        Calculate the apparent magnitude V of an asteroid at phase angle alpha
        using the HG1G2 phase function.
    
        Parameters:
        - H (float): Absolute magnitude of the asteroid.
        - G1 (float): Slope parameter 1 of the asteroid (0  G1  1).
        - G2 (float): Slope parameter 2 of the asteroid (0  G2  1).
        - alpha (float or array-like): Phase angle(s) in degrees.
    
        Returns:
        - V (float or array-like): Apparent magnitude(s) at phase angle(s) alpha.
        """
        # Convert alpha to radians
        alpha_rad = np.radians(alpha)
        
        cs_ph1, cs_ph2, cs_ph3 = self.spline()
        # Compute Phi1 and Phi2
        phi1 = cs_ph1(alpha)
        phi2 = cs_ph2(alpha)
    
        # Compute Phi3 using alpha in degrees
        phi3 = cs_ph3(alpha)
    
        # Compute the magnitude V(alpha) using the HG1G2 function
        V = H - 2.5 * np.log10(G1 * phi1 + G2 * phi2 + (1 - G1 - G2) * phi3)
    
        return V
    #%%
    def mag_area_HG1G2(self, results, H_val_2, G1_val, G2_val, ph_an):
        """
        Generate sampled magnitudes for the HG1G2 model considering parameter uncertainties.
    
        Parameters:
        - results: Fitting result object containing the covariance matrix.
        - H_val_2 (float): Best-fit absolute magnitude H.
        - G1_val (float): Best-fit slope parameter G1.
        - G2_val (float): Best-fit slope parameter G2.
        - ph_an (array-like): Phase angles in degrees.
    
        Returns:
        - mag_samples (ndarray): Simulated magnitudes for parameter samples and phase angles.
        """
    
        # Get the covariance matrix
        cov_matrix = results.covar  # Ensure this is a 3x3 matrix for [H, G1, G2]
        
        # Construct mean parameter vector
        params_mean = np.array([H_val_2, G1_val, G2_val])
        #print(cov_matrix)
        # Ensure covariance matrix shape matches parameters
        if cov_matrix.shape != (3, 3):
            raise ValueError("Covariance matrix must be 3x3 for parameters [H, G1, G2]")
    
        # Generate sample parameters from multivariate normal distribution
        n_samples = 3000  # Number of simulations
        samples = np.random.multivariate_normal(params_mean, cov_matrix, n_samples)
    
        # Initialize array to store sampled magnitudes
        mag_samples = np.zeros((n_samples, len(ph_an)))
    
        # Loop through samples and compute model magnitudes
        for i in range(n_samples):
            H_sample, G1_sample, G2_sample = samples[i]
    
            # Apply constraints to G1 and G2
            if G1_sample < 0 or G2_sample < 0 or (G1_sample + G2_sample > 1):
                continue  # Skip invalid parameter samples
    
            # Compute magnitudes using the HG1G2 phase function
            mag_samples[i] = self.hg1g2_phase_function(ph_an, H_sample, G1_sample, G2_sample)
        # Step 1: Identify rows that are not all zeros
        non_zero_rows = ~np.all(mag_samples == 0, axis=1)
    
        # Step 2: Filter out rows with all zeros
        filtered_data = mag_samples[non_zero_rows]
        return filtered_data
    #%%
    def fit(self, phase, mag, weights=None, method="HG", mag_errors=None, G1= None, G2 = None):
        """
        Fit a phase curve using the HG or HG1G2 model with constraints.
    
        Parameters:
        - phase (array-like): Phase angles.
        - mag (array-like): Observed magnitudes.
        - weights (array-like, optional): Weights for fitting. Defaults to 1/mag_errors.
        - method (str): Phase function model to use ("HG" or "HG1G2").
        - mag_errors (array-like, optional): Magnitude uncertainties. Defaults to 0.03 for all data points.
    
        Returns:
        - result: Fitting result object from lmfit.
        """
        # Select the appropriate model
        if method == "HG":
            model = lmfit.Model(self.eval_fit_HG)
        elif method == "HG1G2":
            model = lmfit.Model(self.eval_fit_HG1G2)
        else:
            raise ValueError("Invalid method. Use 'HG' or 'HG1G2'.")
    
        # Initialize parameters
        params = lmfit.Parameters()
        params.add("H", value=15, min=0, max=30)
    
        if method == "HG":
            #params['G'].vary = False
            if G1 and G2 == None:
                params.add("G", value=G1, min=0, max=1.0, vary = False) 
            else:
                params.add("G", value=0.15, min=0, max=1.0)
        elif method == "HG1G2":
            if G1:
                params.add("G1", value=G1, min=0, max=1.0, vary = False)    
            else:
                params.add("G1", value=0.15, min=0, max=1.0)  # Free parameter
            if G2:
                params.add("G2", value = G2, min=0, max = 1.0, vary = False)
            else:    
                print ("Not fixed")
                params.add("G2", value = 0.2, min=0, max = 1.0)        # Dependent parameter ensuring G1 + G2 <= 1
    
        # Set default magnitude errors if not provided
        if mag_errors is None:
            mag_errors = np.full(len(phase), 0.03)
    
        # Default weights if not provided
        if weights is None:
            weights = 1.0 / mag_errors
    
        # Perform the fit
        result = model.fit(
            mag,
            params,
            phase=phase,
            method="least_squares",
            weights=weights,
            fit_kws={"loss": "soft_l1"},
        )
    
        return result
    #%%
    def eval_fit_HG1G2(self, phase, H, G1, G2):
        """Evaluation function for fitting. Required for lmfit."""
        return phot.HG1G2.evaluate(np.radians(phase), H, G1,G2) 
    #%%
    
    def eval_fit_HG(self, phase, H, G):
        """Evaluation function for fitting. Required for lmfit."""
        return phot.HG.evaluate(np.radians(phase), H, G)
    
        
    def hg_phase_function(self, H, G, alpha):
        """
        Calculate the apparent magnitude V of an asteroid at phase angle alpha
        using the HG phase function.
    
        Parameters:
        - H (float): Absolute magnitude of the asteroid.
        - G (float): Slope parameter of the asteroid (0  G  1).
        - alpha (float or array-like): Phase angle(s) in degrees.
    
        Returns:
        - V (float or array-like): Apparent magnitude(s) at phase angle(s) alpha.
        """
        # Convert alpha to radians
        alpha_rad = np.radians(alpha)
    
        # Compute Phi1 and Phi2
        phi1 = np.exp(-3.33 * np.power(np.tan(alpha_rad / 2), 0.63))
        phi2 = np.exp(-1.87 * np.power(np.tan(alpha_rad / 2), 1.22))
    
        # Compute the magnitude V(alpha)
        V = H - 2.5 * np.log10((1 - G) * phi1 + G * phi2)
    
        return V
    def mag_area_HG(self, results, H_val, G_val, ph_an):
            
        # Get the covariance matrix
        cov_matrix = results.covar  # result.covar is the covariance matrix
        
        # Construct covariance matrix for multivariate normal sampling
        params_mean = np.array([H_val, G_val])
        #params_mean = np.array([H_val])
        
        # Just to make consistent the following code
        params_cov = cov_matrix
        # 8. Generate sample parameters from multivariate normal distribution
        n_samples = 3000  # Number of simulations
        samples = np.random.multivariate_normal(params_mean, params_cov, n_samples)
    
        # 9. Compute model magnitudes for each set of sampled parameters
        mag_samples = np.zeros((n_samples, len(ph_an)))
    
        for i in range(n_samples):
            H_sample, G_sample = samples[i]
            #H_sample = samples[i]
            
            mag_samples[i] = hg_phase_function(H_sample, G_sample, ph_an)
            #mag_samples[i] = hg_phase_function(H_sample, 0.15, ph_an)
        return mag_samples
    

    def reference_obs_comp(self, sheet_name, method = "HG1G2", return_errors=False): 
        '''
        Computing both the outlier removing and also the H G_1 G_2 values

        Input:
            sheet_name: the reference sheet

        Output: 
            H: absolute mag
            G_1, G_2: slope parameters
            optional errors if return_errors=True
        '''
        # sheet_name= nosaukums worksheetam
        df = pd.read_excel(self.path+self.file_name, index_col=None, sheet_name=sheet_name)
        df = df.dropna(subset=['magred'])
        # Reading mag (absolute mag preferably) and time
        H =df["magred"]
        mag = df["mag"]
        time = df["epoch"]
        Ph = df["Ph"]
        sol_dis =df.iloc[:,3]
        geo_dis = df.iloc[:,4]
        
        
        H = mag - 5*np.log10(geo_dis*sol_dis)
        try: 
            time_red = df["Jdc2"]
        except:
            time_red = time+(sol_dis- sol_dis[0])*4.99/36/24
        
        try:
            filter_name = sheet_name[-1]
            H = H + self.filter_bias[filter_name]
        except:
            H= H
        
        Ph_r, H_r, remove_idx = self.removeOutliers(np.array(Ph), np.array(H), 1.5)
        plt.scatter(Ph, H, c = "goldenrod", s = 16)
        plt.scatter(Ph, mag, marker = "o", s = 16, c = "darkgoldenrod")
        plt.scatter(np.array(Ph)[remove_idx], np.array(H)[remove_idx], s = 25, c = "red", marker = "x")
        plt.title(f"{sheet_name} data")
        plt.xlabel("Phase, deg")
        plt.ylabel("Magnitude")
        plt.show()

        ##### Computing HG1G2 params ##################
        
        H2 = mag - 5*np.log10(geo_dis*sol_dis)
        try: 
            time_red2 = df["Jdc2"]
        except:
            time_red2 = time+(sol_dis- sol_dis[0])*4.99/36/24
        
        try:
            filter_name2 = sheet_name[-1]
            H2 = H2 + self.filter_bias[filter_name2]
        except:
            H2 = H2
        
        ph_an2 = np.linspace(0, max(Ph) + 2, 100)
        
        results22 = self.fit(Ph_r, H_r, method=method)
        if method == "HG1G2":
            H_val_22 = results22.params["H"].value
            G1_val2 = results22.params["G1"].value
            G2_val2 = results22.params["G2"].value
            
            H_err_22 = results22.params["H"].stderr or 0
            G1_err2 = results22.params["G1"].stderr or 0
            G2_err2 = results22.params["G2"].stderr or 0
        
            mag_analy_22 = self.hg1g2_phase_function(ph_an2, H_val_22, G1_val2, G2_val2)
            
            #mag_samples_22 = mag_area_HG1G2(results22, H_val_22, G1_val2, G2_val2, ph_an2)
            
            print ("Absolute magnitude: {:.2f}".format(H_val_22))
            print ("G1 = {:.3f}, G2 = {:.7f}".format(G1_val2, G2_val2))
            print ("Parameter errors: dH = {:.4f}, dG1 = {:.4f}, dG2 = {:.4f}".format(H_err_22, G1_err2, G2_err2))
            print ("G1 = {:.3f} +/- {:.4f}, G2 = {:.7f} +/- {:.4f}".format(G1_val2, G1_err2, G2_val2, G2_err2))
            
            plt.plot(ph_an2, mag_analy_22, label = "H ={:.2f}, G1 = {:.2f}, G2 = {:.2f}".format(H_val_22, G1_val2, G2_val2), c = "black")
            plt.scatter(Ph_r, H_r, label = "T08o: reference observatory", c= "goldenrod")
            plt.gca().invert_yaxis()
            plt.show()
            if return_errors:
                return H_val_22, G1_val2, G2_val2, H_err_22, G1_err2, G2_err2
            return H_val_22, G1_val2, G2_val2

    def all_obs_comb(self, H_val_2, G1_val = 1, G2_val = 0, method = "HG1G2", save_figure = False, save_figures = None,
                     save_path = None, save_file = False, ref_redchi = None, chi2_factor = 3.0):

        i = 0
        plt.figure(dpi =300, figsize = (10, 10))
        dict_sheets = {}
        for sheet_name in self.reduced_obs:
            try:
                filter_name = sheet_name[-1]
                # sheet_name= nosaukums worksheetam
                df = pd.read_excel(self.path+self.file_name, index_col=None, sheet_name=sheet_name)
                df = df.dropna(subset=['magred'])
                # Reading mag (absolute mag preferably) and time
                H =df["magred"]
                mag = df["mag"]
                time = df["epoch"]
                Ph = df["Ph"]
                sol_dis =df.iloc[:,3]
                geo_dis = df.iloc[:,4]
            
                H = mag - 5*np.log10(geo_dis*sol_dis)
            except Exception as e:
                print (f"Error with the magnitude: {e}")
                return None
            try: 
                time_red = df["Jdc2"]
            except:
                time_red = time+(sol_dis- sol_dis[0])*4.99/36/24
        
            try:
                H = H + self.filter_bias[filter_name] 
            except:
                H = H
            
            ph_an_obs = np.linspace(0, max(Ph) + 2, 100)
        
            Ph, H, remove_idx = self.removeOutliers(np.array(Ph), np.array(H), 1.8)
        
        
            if method == "HG1G2":
                results2 = self.fit(Ph, H, method=method, G1 = G1_val, G2 = G2_val)
            
                H_val_2_obs = results2.params["H"].value
                G1_val_obs = results2.params["G1"].value
                G2_val_obs = results2.params["G2"].value
                
                #print (sheet_name, "G1 = {}, G2 = {}".format(G1_val_obs, G2_val_obs))
                red_chi2 = results2.redchi
                if ref_redchi is None:
                    ref_redchi = red_chi2
                    print(f"Reference reduced chi^2 set to {ref_redchi:.3f} (from {sheet_name})")
                chi2_status = "PASS" if red_chi2 <= chi2_factor * ref_redchi else "FAIL"
                print(f"{sheet_name}: HG1G2 reduced chi^2 = {red_chi2:.3f} [{chi2_status}]")
            
            
                mag_analy_2_obs = self.hg1g2_phase_function(ph_an_obs, H_val_2_obs, G1_val_obs, G2_val_obs)
                
            
                # Computing the HG1G2 profile of the best obs
                H_model = self.hg1g2_phase_function(Ph, H_val_2, G1_val, G2_val)
                # Computing the HG1G2 current observatory model
                H_current_model = self.hg1g2_phase_function(Ph, H_val_2_obs, G1_val_obs, G2_val_obs)
            if method == "HG":
                results2 = self.fit(Ph, H, method=method, G1 = G1_val, G2 = None)
            
                H_val_2_obs = results2.params["H"].value
                G1_val_obs = results2.params["G"].value
                
                #print (sheet_name, "G1 = {}, G2 = {}".format(G1_val_obs, G2_val_obs))
            
            
                mag_analy_2_obs = self.hg_phase_function(H_val_2_obs, G1_val_obs, ph_an_obs)
                
            
                # Computing the HG1G2 profile of the best obs
                H_model = self.hg_phase_function(H_val_2, G1_val, Ph)
                # Computing the HG1G2 current observatory model
                H_current_model = self.hg_phase_function(H_val_2_obs, G1_val_obs, Ph)
            # The difference between best and current obs
            diff_for_H = H_model - H_current_model
        
            # applyin gthe correction
            H_corr = H + diff_for_H
        
            #Phase correction
            H_reduced = H_corr - (H_current_model - H_val_2_obs)
            #H_reduced = H_corr
            
            plt.scatter(Ph, H_reduced, label = sheet_name)
            plt.legend()
            plt.xlabel("Phase")
            plt.ylabel("Mag")
            plt.tight_layout()
        
            time_idx_removed = np.delete(time_red, remove_idx)
            dict_sheets[sheet_name] = np.array([H_reduced, time_idx_removed, Ph])
            i+=1
        if save_figure:
            save_figures = save_figures or self.base_dir
            print ("Saving figure!")
            plt.savefig(os.path.join(save_figures, "{}_data_reduction_plot_flat.png".format(self.Asteroid_number)))
            plt.show()

        if save_file:
            save_path = save_path or self.base_dir
            print (f"Saving the file in location: {save_path}")
            #%%
            with open(os.path.join(save_path, '{}_data_compile_fix_G1G2.pkl'.format(self.Asteroid_number)), 'wb') as file:
                pickle.dump(dict_sheets, file)
        return dict_sheets

    def compare_phase_curve_fit_strategies(
        self,
        reference_sheet="T08o1",
        outlier_param=1.8,
        phase_margin_deg=2.0,
        save_figures=False,
        save_dir=None,
    ):
        """
        Build 3 plots for HG1G2 phase-curve fitting strategies:

        1) Free-fit per observatory: H, G1, G2 all free.
        2) Reference-fixed fit: fit reference sheet first, then fix G1/G2 to reference for all observatories.
        3) Chi2 comparison bar plot (free-fit vs reference-fixed) by observatory.

        Returns:
            pandas.DataFrame with observatory code and chi2/reduced-chi2 for both approaches.
        """

        def _load_observatory_data(sheet_name):
            df = pd.read_excel(self.path + self.file_name, index_col=None, sheet_name=sheet_name)
            df = df.dropna(subset=["magred"])

            mag = np.array(df["mag"])
            phase = np.array(df["Ph"])
            sol_dis = np.array(df.iloc[:, 3])
            geo_dis = np.array(df.iloc[:, 4])
            H = mag - 5 * np.log10(geo_dis * sol_dis)

            # Apply single-band correction when filter code is available in sheet name.
            try:
                filter_name = sheet_name[-1]
                H = H + self.filter_bias[filter_name]
            except Exception:
                pass

            phase_clean, H_clean, _ = self.removeOutliers(np.array(phase), np.array(H), outlier_param)
            return phase_clean, H_clean

        if reference_sheet is None:
            raise ValueError("reference_sheet must be provided, e.g. 'T08o1'.")

        obs_list = list(self.reduced_obs)
        if reference_sheet not in obs_list:
            obs_list = [reference_sheet] + obs_list

        # Fit reference observatory first to get fixed G1/G2 strategy values.
        Ph_ref, H_ref = _load_observatory_data(reference_sheet)
        ref_result = self.fit(Ph_ref, H_ref, method="HG1G2")
        ref_H = ref_result.params["H"].value
        ref_G1 = ref_result.params["G1"].value
        ref_G2 = ref_result.params["G2"].value
        ref_H_err = ref_result.params["H"].stderr or 0.0
        ref_G1_err = ref_result.params["G1"].stderr or 0.0
        ref_G2_err = ref_result.params["G2"].stderr or 0.0

        print(
            f"Reference {reference_sheet}: H={ref_H:.3f} +/- {ref_H_err:.4f}, "
            f"G1={ref_G1:.4f} +/- {ref_G1_err:.4f}, G2={ref_G2:.4f} +/- {ref_G2_err:.4f}"
        )

        # Containers for plotting/reporting.
        records = []

        cmap = colormaps["tab20"]
        colors = cmap(np.linspace(0, 1, max(len(obs_list), 2)))
        markers = ["o", "s", "^", "D", "v", "P", "X", ">", "<", "*", "h", "8"]

        # -------- Plot 1: all observatories free fit (H, G1, G2 free) --------
        fig1 = plt.figure(figsize=(10, 8), dpi=300)
        ax1 = plt.gca()

        for i, sheet_name in enumerate(obs_list):
            Ph, H = _load_observatory_data(sheet_name)
            ph_grid = np.linspace(0, max(Ph) + phase_margin_deg, 120)

            res_free = self.fit(Ph, H, method="HG1G2")
            H_free = res_free.params["H"].value
            G1_free = res_free.params["G1"].value
            G2_free = res_free.params["G2"].value
            H_free_err = res_free.params["H"].stderr or 0.0
            G1_free_err = res_free.params["G1"].stderr or 0.0
            G2_free_err = res_free.params["G2"].stderr or 0.0

            y_free = self.hg1g2_phase_function(ph_grid, H_free, G1_free, G2_free)
            marker = markers[i % len(markers)]
            ax1.scatter(
                Ph,
                H,
                s=20,
                marker=marker,
                color=colors[i],
                alpha=0.30,
                edgecolors="none",
                zorder=1,
            )
            ax1.plot(
                ph_grid,
                y_free,
                lw=1.1,
                color=colors[i],
                label=(
                    f"{sheet_name}: H={H_free:.2f}+/-{H_free_err:.2f}, "
                    f"G1={G1_free:.3f}+/-{G1_free_err:.3f}, "
                    f"G2={G2_free:.3f}+/-{G2_free_err:.3f}"
                ),
            )

            # Approach 2 (fixed G1/G2 from reference)
            res_fix = self.fit(Ph, H, method="HG1G2", G1=ref_G1, G2=ref_G2)
            H_fix = res_fix.params["H"].value
            H_fix_err = res_fix.params["H"].stderr or 0.0

            records.append(
                {
                    "observatory": sheet_name,
                    "H_free": H_free,
                    "G1_free": G1_free,
                    "G2_free": G2_free,
                    "H_free_err": H_free_err,
                    "G1_free_err": G1_free_err,
                    "G2_free_err": G2_free_err,
                    "chi2_free": float(res_free.chisqr),
                    "redchi_free": float(res_free.redchi),
                    "H_fix": H_fix,
                    "H_fix_err": H_fix_err,
                    "G1_fix": ref_G1,
                    "G2_fix": ref_G2,
                    "chi2_fix": float(res_fix.chisqr),
                    "redchi_fix": float(res_fix.redchi),
                }
            )

        ax1.set_xlabel("Phase (deg)")
        ax1.set_ylabel("Reduced magnitude")
        ax1.invert_yaxis()
        ax1.grid(alpha=0.25, linestyle=":")
        ax1.legend(fontsize=7, ncol=1, frameon=False)
        ax1.set_title("Free HG1G2 fits per observatory (H, G1, G2 free)")
        plt.tight_layout()

        # -------- Plot 2: reference curve + fixed G1/G2 fits for others --------
        fig2 = plt.figure(figsize=(10, 8), dpi=300)
        ax2 = plt.gca()

        Ph_ref_plot, H_ref_plot = _load_observatory_data(reference_sheet)
        ph_ref_grid = np.linspace(0, max(Ph_ref_plot) + phase_margin_deg, 120)
        y_ref = self.hg1g2_phase_function(ph_ref_grid, ref_H, ref_G1, ref_G2)
        ax2.scatter(
            Ph_ref_plot,
            H_ref_plot,
            s=24,
            marker="o",
            color="black",
            alpha=0.22,
            edgecolors="none",
            zorder=1,
        )
        ax2.plot(
            ph_ref_grid,
            y_ref,
            color="black",
            lw=2.0,
            label=(
                f"Reference {reference_sheet}: H={ref_H:.2f}+/-{ref_H_err:.2f}, "
                f"G1={ref_G1:.3f}+/-{ref_G1_err:.3f}, G2={ref_G2:.3f}+/-{ref_G2_err:.3f}"
            ),
        )

        for i, rec in enumerate(records):
            Ph, H = _load_observatory_data(rec["observatory"])
            ph_grid = np.linspace(0, max(Ph) + phase_margin_deg, 120)
            y_fix = self.hg1g2_phase_function(ph_grid, rec["H_fix"], rec["G1_fix"], rec["G2_fix"])
            marker = markers[i % len(markers)]
            ax2.scatter(
                Ph,
                H,
                s=20,
                marker=marker,
                color=colors[i],
                alpha=0.30,
                edgecolors="none",
                zorder=1,
            )
            ax2.plot(
                ph_grid,
                y_fix,
                lw=1.1,
                color=colors[i],
                alpha=0.9,
                label=(
                    f"{rec['observatory']}: H={rec['H_fix']:.2f}+/-{rec['H_fix_err']:.2f}, "
                    f"G1={rec['G1_fix']:.3f}, G2={rec['G2_fix']:.3f}"
                ),
            )

        ax2.set_xlabel("Phase (deg)")
        ax2.set_ylabel("Reduced magnitude")
        ax2.invert_yaxis()
        ax2.grid(alpha=0.25, linestyle=":")
        ax2.legend(fontsize=7, ncol=1, frameon=False)
        ax2.set_title(f"Reference-fixed HG1G2 fits (G1,G2 fixed to {reference_sheet})")
        plt.tight_layout()

        # -------- Plot 3: reduced chi2 comparison (bar chart over observatories) --------
        df_cmp = pd.DataFrame.from_records(records)
        x = np.arange(len(df_cmp))
        width = 0.4

        fig3 = plt.figure(figsize=(11, 4.8), dpi=300)
        ax3 = plt.gca()
        ax3.bar(x - width / 2, df_cmp["redchi_free"], width=width, label="reduced chi2: free HG1G2", color="#4C78A8")
        ax3.bar(x + width / 2, df_cmp["redchi_fix"], width=width, label="reduced chi2: reference-fixed HG1G2", color="#F58518")
        ax3.set_xticks(x)
        ax3.set_xticklabels(df_cmp["observatory"], rotation=45, ha="right")
        ax3.set_ylabel("Reduced chi2")
        ax3.set_xlabel("Observatory")
        ax3.set_title("Reduced chi2 comparison by observatory")
        ax3.grid(axis="y", alpha=0.25, linestyle=":")
        ax3.legend(frameon=False)
        plt.tight_layout()

        if save_figures:
            out_dir = save_dir or self.base_dir
            os.makedirs(out_dir, exist_ok=True)
            fig1.savefig(os.path.join(out_dir, f"{self.Asteroid_number}_phase_free_fits.pdf"), dpi=600)
            fig2.savefig(os.path.join(out_dir, f"{self.Asteroid_number}_phase_reference_fixed_fits.pdf"), dpi=600)
            fig3.savefig(os.path.join(out_dir, f"{self.Asteroid_number}_chi2_comparison.pdf"), dpi=600)
            print(f"Saved comparison figures to: {out_dir}")

        return df_cmp

