"""Status of the DAMIT runs: WSL cores/load and per-stage progress."""
import glob
import json
import os
import subprocess

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "damit_runs")
p = subprocess.run(["wsl", "-d", "myrootfs", "-e", "bash", "-c", "nproc; cat /proc/loadavg; pgrep -c -f 'period_scan|convexinv'"],
                   capture_output=True, text=True)
print("WSL nproc / loadavg / damit processes:", p.stdout.split())
runs = sorted(d for d in os.listdir(R) if os.path.isfile(os.path.join(R, d, "prep.json")))
st = {"scan": 0, "scan_lines": 0, "poles": 0, "loao": 0, "ellipsoid": 0, "eval": 0}
for d in runs:
    info = json.load(open(os.path.join(R, d, "prep.json")))
    st["scan"] += os.path.exists(os.path.join(R, d, "scan", "done"))
    f = os.path.join(R, d, "scan", "out_periods")
    if os.path.exists(f):
        st["scan_lines"] += sum(1 for _ in open(f))
    st["poles"] += os.path.exists(os.path.join(R, d, "poles", "poles.json"))
    st["loao"] += os.path.exists(os.path.join(R, d, "loao", "loao.json"))
    st["ellipsoid"] += os.path.exists(os.path.join(R, d, "ellipsoid.json"))
    st["eval"] += os.path.exists(os.path.join(R, d, "eval.json"))
tot = sum(json.load(open(os.path.join(R, d, "prep.json")))["n_trial_periods"] for d in runs)
print(f"{len(runs)} runs; done: {st}; scan trial periods written {st['scan_lines']}/{tot}")
