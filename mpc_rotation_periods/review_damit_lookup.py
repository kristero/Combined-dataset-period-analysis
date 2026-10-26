"""Review A08: DAMIT models (with references) of (4245), (4396), (4761) and of the other main-sample asteroids."""
import csv
import io
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
rd = lambda n: list(csv.DictReader(io.StringIO(open(os.path.join(ROOT, "damit_ref", "tables", n + ".csv"), "rb")
                                               .read().decode("utf-8-sig"))))
ast = {a["id"]: a for a in rd("asteroids")}
refs = {r["id"]: r for r in rd("references")}
mref = {}
for x in rd("asteroid_models_references"):
    mref.setdefault(x["asteroid_model_id"], []).append(refs.get(x["reference_id"], {}))
AD = json.load(open(os.path.join(ROOT, "adopted.json"), encoding="utf-8"))
main = {n for n, a in AD.items() if a["table"] in (3, 4)}
for m in rd("asteroid_models"):
    a = ast.get(m["asteroid_id"])
    if a and a["number"] in main:
        print(a["number"], m["id"], m["lambda"], m["beta"], m["period"],
              [(r.get("author_short"), r.get("year"), r.get("journal")) for r in mref.get(m["id"], [])])
