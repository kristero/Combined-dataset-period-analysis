"""Kill DAMIT processes in WSL whose working directory matches a pattern: python damit_kill.py <pattern> [--list]"""
import subprocess
import sys

pat = sys.argv[1] if len(sys.argv) > 1 else "NOMATCH"
listonly = "--list" in sys.argv
cmd = ("for p in $(pgrep -f 'period_scan|convexinv'); do c=$(readlink /proc/$p/cwd); echo \"$p $c\"; "
       + ("" if listonly else f"case \"$c\" in *{pat}*) kill $p; echo killed $p;; esac; ")
       + "done")
print(subprocess.run(["wsl", "-d", "myrootfs", "-e", "bash", "-c", cmd], capture_output=True, text=True).stdout)
