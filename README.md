# Combined-dataset-period-analysis

Repository for combining observatory datasets and performing period (rotation/photometry) analyses of asteroids.

## Overview
This project merges multiple observatory light-curve datasets, applies outlier removal, detrending, and uses the Lomb–Scargle method (via `astropy.timeseries.LombScargle`) to identify rotation periods. It also includes phase-curve fitting using the HG and HG1G2 models (via `lmfit` and `sbpy.photometry`).

## Setup
### 1. Clone this repository
```bash
git clone https://github.com/kristero/Combined-dataset-period-analysis.git
cd Combined-dataset-period-analysis
