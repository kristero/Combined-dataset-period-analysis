"""Download the MPC observations (ADES fields, including the star catalogs astCat/photCat) of the
paper asteroids from the MPC get-obs API. Used only to attach the catalog code of every measurement,
which the DePhOCUS corrections (Hoffmann et al. 2025) need."""
import json, os, sys, time
import requests

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
os.makedirs(OUT, exist_ok=True)
NUMS = [1951, 1963, 2134, 2150, 2607, 2968, 2971, 3081, 3173, 3473, 3716, 4303]

for num in (map(int, sys.argv[1:]) if len(sys.argv) > 1 else NUMS):
    path = os.path.join(OUT, f"{num}_ades.json")
    if os.path.exists(path):
        print(num, "exists"); continue
    for attempt in range(5):
        try:
            r = requests.get("https://data.minorplanetcenter.net/api/get-obs",
                             json={"desigs": [str(num)], "output_format": ["ADES_DF"]}, timeout=300)
            r.raise_for_status()
            ades = r.json()[0]["ADES_DF"]
            json.dump(ades, open(path, "w", encoding="utf-8"))
            print(num, len(ades), "records")
            break
        except Exception as exc:
            print(num, "attempt", attempt + 1, "failed:", exc)
            time.sleep(5 * (attempt + 1))
    time.sleep(2)
