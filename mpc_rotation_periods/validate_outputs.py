"""Check that every bootstrap result file is complete, valid JSON (a failed flush can leave empty files).
Usage: python validate_outputs.py [extra_dir ...]"""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
patterns = [os.path.join(ROOT, "results", "*", "*_rebootstrap*.json")] + [os.path.join(d, "*.json") for d in sys.argv[1:]]
bad, n = [], 0
for pat in patterns:
    for f in sorted(glob.glob(pat)):
        n += 1
        try:
            d = json.load(open(f, encoding="utf-8"))
            if not d.get("primary"):
                bad.append((f, "no primary candidates"))
        except Exception as e:  # noqa: BLE001
            bad.append((f, f"{type(e).__name__}: {e}"[:80]))
print(f"{n} files checked, {len(bad)} bad")
for f, why in bad:
    print("  BAD", f, "--", why, "-- size", os.path.getsize(f))
