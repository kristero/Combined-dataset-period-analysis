"""
Reference shape models from DAMIT (damit.cuni.cz) for the validation asteroids (validation/sample.json): the CSV tables
(damit_ref/tables/, downloaded with curl from /projects/damit/exports/table/<name>) give the models of every asteroid
and their references; for each model this script fetches spin.txt and shape.obj (generated files,
/projects/damit/generated_files/open/AsteroidModel/<model id>/...), one request per second, and writes
damit_ref/models.json. Run on the LU HPC login node (compute nodes have no internet access): python3 damit_ref_fetch.py
"""
from __future__ import annotations

import csv
import io
import json
import os
import time
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(ROOT, "damit_ref")
BASE = "https://damit.cuni.cz/projects/damit/generated_files/open/AsteroidModel/{}/{}"


def table(name):
    raw = open(os.path.join(REF, "tables", f"{name}.csv"), "rb").read().decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(raw)))


def fetch(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return False
    req = urllib.request.Request(url, headers={"User-Agent": "MPC-rotation-period-validation (LU, research use)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    with open(path, "wb") as fh:
        fh.write(data)
    return True


def main():
    sample = json.load(open(os.path.join(ROOT, "validation", "sample.json"), encoding="utf-8"))["sample"]
    nums = {str(s["num"]) for s in sample}
    ast = {a["id"]: a for a in table("asteroids")}
    by_num = {a["number"]: a for a in ast.values() if a["number"]}
    refs = {r["id"]: r for r in table("references")}
    mref = {}
    for x in table("asteroid_models_references"):
        mref.setdefault(x["asteroid_model_id"], []).append(refs.get(x["reference_id"], {}))
    models = {}
    for m in table("asteroid_models"):
        a = ast.get(m["asteroid_id"])
        if a is None or a["number"] not in nums:
            continue
        rl = mref.get(m["id"], [])
        models[m["id"]] = dict(model_id=int(m["id"]), number=int(a["number"]), name=a["name"],
                               lambda_=float(m["lambda"]), beta=float(m["beta"]), period=float(m["period"]),
                               jd0=float(m["jd0"]) if m["jd0"] else None, phi0=float(m["phi0"]) if m["phi0"] else 0.0,
                               yorp=m["yorp"] or None, nonconvex=m["nonconvex"] or None, quality_flag=m["quality_flag"] or None,
                               version=m["version"] or None, comment=m["comment"] or None,
                               references=[dict(author_short=r.get("author_short"), year=r.get("year"),
                                                bibcode=r.get("bibcode"), title=r.get("title")) for r in rl])
    print(f"{len(nums)} validation asteroids, {len(set(m['number'] for m in models.values()))} with DAMIT models, "
          f"{len(models)} models", flush=True)
    n_new = 0
    for k, (mid, m) in enumerate(sorted(models.items(), key=lambda kv: int(kv[0]))):
        d = os.path.join(REF, "models", mid)
        os.makedirs(d, exist_ok=True)
        for fn in ("spin.txt", "shape.obj"):
            try:
                if fetch(BASE.format(mid, fn), os.path.join(d, fn)):
                    n_new += 1
                    time.sleep(1.0)
            except Exception as e:                       # keep going; the model is flagged below
                m.setdefault("fetch_errors", []).append(f"{fn}: {e}")
        m["files_ok"] = all(os.path.exists(os.path.join(d, f)) and os.path.getsize(os.path.join(d, f)) > 0
                            for f in ("spin.txt", "shape.obj"))
        if k % 25 == 0:
            print(f"  {k + 1}/{len(models)} models, {n_new} files fetched", flush=True)
    json.dump(dict(source="DAMIT, https://damit.cuni.cz (tables and generated files)",
                   fetched=time.strftime("%Y-%m-%d %H:%M"), models=models),
              open(os.path.join(REF, "models.json"), "w", encoding="utf-8"), indent=1)
    size = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(os.path.join(REF, "models")) for f in fs)
    print(f"done: {sum(m['files_ok'] for m in models.values())}/{len(models)} complete, {size / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
