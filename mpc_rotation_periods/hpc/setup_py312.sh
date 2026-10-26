#!/bin/bash
# Python 3.12 environment on the LU HPC with the package versions of the local Anaconda installation that ran the
# period pipeline (CPython 3.12.7, needed for the sourceless repo_snap/asteroid_ls.pyc; Astropy 6.1.3 as in the paper).
# Self-contained in ~/mpc312 (micromamba binary and root prefix); does not touch ~/.bashrc or the system Anaconda.
set -e
BASE=$HOME/mpc312
mkdir -p $BASE/bin
cd $BASE
if [ ! -x bin/micromamba ]; then
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba
fi
export MAMBA_ROOT_PREFIX=$BASE/root
./bin/micromamba create -y -q -n py312 -c conda-forge \
  python=3.12.7 numpy=1.26.4 scipy=1.13.1 pandas=2.2.2 matplotlib=3.9.2 astropy=6.1.3 \
  openpyxl=3.1.5 scikit-learn=1.5.1 requests h5py pyyaml
PY=$HOME/micromamba/envs/py312/bin/python   # envs_dirs of the existing conda configuration
$PY -m pip install -q lmfit==1.3.4 sbpy==0.6.0 astroquery==0.4.11
$PY - <<'EOF'
import sys, importlib.util, astropy, numpy, scipy, pandas, lmfit, sbpy, sklearn, astroquery
print(sys.version.split()[0], "magic", importlib.util.MAGIC_NUMBER.hex(), "astropy", astropy.__version__, "numpy", numpy.__version__,
      "scipy", scipy.__version__, "pandas", pandas.__version__, "lmfit", lmfit.__version__, "sbpy", sbpy.__version__,
      "sklearn", sklearn.__version__, "astroquery", astroquery.__version__)
EOF
echo SETUP_DONE
