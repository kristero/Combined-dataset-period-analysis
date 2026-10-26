"""Detached driver for the DAMIT pipeline: waits for an earlier orchestrator (PID argument) to exit, then runs
scan (resumable) -> poles -> loao -> ellipsoid -> eval, logging to damit_runs/supervisor_log.txt; writes ALL_DONE."""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(ROOT, "damit_runs")
LOG = os.path.join(R, "supervisor_log.txt")


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")


def alive(pid):
    import psutil
    return psutil.pid_exists(pid)


if len(sys.argv) > 1:
    pid = int(sys.argv[1])
    log(f"waiting for PID {pid}")
    while alive(pid):
        time.sleep(30)
    log(f"PID {pid} gone")
for args in (["damit_run.py", "scanpoles", "--workers", "18"],
             ["damit_run.py", "loao", "--workers", "18"], ["damit_ellipsoid.py", "--workers", "18"],
             ["damit_eval.py", "--workers", "16"]):
    t = time.time()
    name = args[0].split(".")[0] + ("_" + args[1] if not args[1].startswith("-") else "")
    log("start " + " ".join(args))
    with open(os.path.join(R, f"log_{name}.txt"), "a", encoding="utf-8") as f:
        p = subprocess.run([sys.executable, "-u", os.path.join(ROOT, args[0])] + args[1:], stdout=f,
                           stderr=subprocess.STDOUT, cwd=ROOT)
    log(f"end {' '.join(args)} rc={p.returncode} {(time.time() - t) / 60:.1f} min")
open(os.path.join(R, "ALL_DONE"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S\n"))
log("ALL_DONE")
