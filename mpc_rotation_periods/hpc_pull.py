"""Pull results from the LU HPC into this folder: rsync as WSL user kn18001 (SSH key) into /tmp/mpc_pull, then copy
into the Windows folder as root (kn18001 cannot write to /mnt/c)."""
import subprocess

SRC = "/mnt/c/Users/kn18001/Documents/Asteroids/MPC_redo"
p1 = subprocess.run(["wsl", "-d", "myrootfs", "-u", "kn18001", "-e", "bash", "-c",
                     f"mkdir -p /tmp/mpc_pull && cd {SRC} && bash hpc_sync.sh pull"], capture_output=True, text=True)
print(p1.stdout[-1500:], p1.stderr[-800:] if p1.returncode else "")
p2 = subprocess.run(["wsl", "-d", "myrootfs", "-u", "root", "-e", "bash", "-c", f"cd {SRC} && bash hpc_sync.sh install"],
                    capture_output=True, text=True)
print("install rc", p2.returncode, p2.stderr[-500:])
