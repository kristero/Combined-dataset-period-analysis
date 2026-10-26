"""Review A08: rotation periods listed by the JPL Small-Body Database (LCDB revision of 2023 October) for the 44
asteroids of the main sample. Writes review_lcdb_main.json"""
import json
import os
import time
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
AD = json.load(open(os.path.join(ROOT, "adopted.json"), encoding="utf-8"))
out = {}
for n in sorted((k for k, a in AD.items() if a["table"] in (3, 4)), key=int):
    for attempt in range(6):
        try:
            with urllib.request.urlopen(f"https://ssd-api.jpl.nasa.gov/sbdb.api?sstr={n}&phys-par=1", timeout=30) as r:
                d = json.load(r)
            break
        except OSError:
            time.sleep(5 * (attempt + 1))
    else:
        raise RuntimeError(f"SBDB query failed for {n}")
    rp = next((p for p in d.get("phys_par", []) if p["name"] == "rot_per"), None)
    out[n] = None if rp is None else {k: rp.get(k) for k in ("value", "sigma", "ref", "notes")}
    print(n, None if rp is None else (rp["value"], rp.get("ref"), (rp.get("notes") or "")[:160]), flush=True)
    time.sleep(0.3)
json.dump(out, open(os.path.join(ROOT, "review_lcdb_main.json"), "w"), indent=1)
print(len(out), "asteroids;", sum(v is not None for v in out.values()), "with a rot_per")
