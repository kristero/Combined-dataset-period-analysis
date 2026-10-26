"""
JPL SBDB records (physical parameters and orbit) of the six test asteroids added in the revision, in the format of
data/sbdb_phys_orbit.json. The full records go to data/sbdb_testswap_full.json. Until the blind review is finished,
data/sbdb_phys_orbit.json receives them without the rotation period (rot_per, from the LCDB); run with --unblind
afterwards to add it.
Usage: python testswap_sbdb.py [--unblind]
"""
import json
import os
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
NUMS = ["249", "486", "870", "1600", "1700", "7750"]
FULL = os.path.join(ROOT, "data", "sbdb_testswap_full.json")
MAIN = os.path.join(ROOT, "data", "sbdb_phys_orbit.json")

if "--unblind" not in sys.argv:
    full = {}
    for n in NUMS:
        for attempt in range(5):
            try:
                with urllib.request.urlopen(f"https://ssd-api.jpl.nasa.gov/sbdb.api?sstr={n}&phys-par=1", timeout=60) as r:
                    full[n] = json.load(r)
                break
            except OSError:
                time.sleep(5 * (attempt + 1))
        print(n, full[n]["object"]["fullname"], [p["name"] for p in full[n].get("phys_par", [])])
    json.dump(full, open(FULL, "w", encoding="utf-8"), indent=1)
else:
    full = json.load(open(FULL, encoding="utf-8"))
main = json.load(open(MAIN, encoding="utf-8"))
for n, e in full.items():
    e = json.loads(json.dumps(e))
    if "--unblind" not in sys.argv:
        e["phys_par"] = [p for p in e.get("phys_par", []) if p["name"] != "rot_per"]
    main[n] = e
json.dump(main, open(MAIN, "w", encoding="utf-8"), indent=1)
print("written", MAIN, "(blind)" if "--unblind" not in sys.argv else "(with rot_per)")
