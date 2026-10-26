"""Run the DAMIT stages after the period scans: waits for all scan/done files, then poles -> loao -> ellipsoid -> eval."""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(ROOT, "damit_runs")
runs = [d for d in os.listdir(R) if os.path.isfile(os.path.join(R, d, "prep.json"))]


def scans_finished():
    # every run either has scan/done or is reported (also as FAILED) in the scan log
    log = os.path.join(R, "scan_log.txt")
    n_log = 0
    if os.path.exists(log):
        with open(log, encoding="utf-8", errors="replace") as f:
            n_log = sum(1 for ln in f if ln.startswith("["))
    n_done = sum(os.path.exists(os.path.join(R, d, "scan", "done")) for d in runs)
    return n_done >= len(runs) or n_log >= len(runs)


while not scans_finished():
    time.sleep(60)
for args in (["damit_run.py", "poles", "--workers", "18"], ["damit_run.py", "loao", "--workers", "18"],
             ["damit_ellipsoid.py", "--workers", "18"], ["damit_eval.py", "--workers", "16"]):
    t = time.time()
    with open(os.path.join(R, f"log_{args[0].split('.')[0]}_{args[1] if args[1][0] != '-' else 'all'}.txt"), "w") as f:
        subprocess.run([sys.executable] + [os.path.join(ROOT, args[0])] + args[1:], stdout=f, stderr=subprocess.STDOUT,
                       cwd=ROOT)
    print(f"{' '.join(args)} finished in {(time.time() - t) / 60:.1f} min", flush=True)
