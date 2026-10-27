ssh -q -o BatchMode=yes luhpc 'cd ~/mpc_redo && $HOME/micromamba/envs/py312/bin/python - <<EOF
import json, sys
import numpy as np
sys.path.insert(0, ".")
import damit_run as R
import fine_checks as F
rows = F.synsid_rows()
for a in ["paper:4618", "paper:4761", "val:594"]:
    kind, n = a.split(":")
    run = f"damit_runs/{kind}_{n}_fsc"
    x = rows[a]
    info = json.load(open(run + "/prep.json"))
    sc = R.read_scan(run)
    ok = np.isfinite(sc["dev"]); P, d = sc["P"][ok], sc["dev"][ok]
    P_alt = x["P_mpc"] - x["expected_own"] * x["P2_over_Porb"]
    near = lambda Pc: d[np.abs(P - Pc) <= 1.5 * info["dP"]].min()
    print(a, "n_trials", ok.sum(), "P_best %.6f" % P[np.argmin(d)], "P_C %.6f" % x["P_C"], "P_alt %.6f" % P_alt, "ratio alt/C %.3f" % ((near(P_alt) / near(x["P_C"])) ** 2))
EOF' 2>&1 | grep -v "micromamba/bin/micromamba: No such"
