"""
Make the zero-point-corrected run (zp/, 703 and G96 datasets split at 2018 July 1; zp_split.py, zp_setup.py) the main
data and results of MPC_redo. The unsplit versions are kept:
  results/ -> results_unsplit/, results_geo/ -> results_geo_unsplit/, data/workbooks and data/obs_tables ->
  data_unsplit/, and the summary JSON files of the unsplit analysis -> unsplit_snapshot/.
Then zp/results, zp/data/workbooks and zp/data/obs_tables take their places. The downstream steps (aspect model,
corrected-data bootstrap, period refinement, quality rule, figures, tables, DAMIT) are re-run afterwards.
Usage: python zp_integrate.py
"""
import json
import os
import shutil
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
ZP = os.path.join(ROOT, "zp")
SNAP = os.path.join(ROOT, "unsplit_snapshot")
SUMMARIES = ["adopted.json", "quality_final.json", "geometry_results.json", "geometry_validation.json",
             "geometry_reanalysis.json", "geometry_reanalysis_validation.json", "figure_info.json",
             "longperiod_ls.json", "longperiod_tests.json", "longperiod_damit.json", "longperiod_longrange.json",
             "damit_paper_table.json", "damit_comparison.json", "inspect_3216_short.json"]


def mv(src, dst):
    src, dst = os.path.join(ROOT, src), os.path.join(ROOT, dst)
    if not os.path.exists(src):
        print("  (missing)", src)
        return
    assert not os.path.exists(dst), f"{dst} exists already"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.move(src, dst)
    print("  moved", os.path.relpath(src, ROOT), "->", os.path.relpath(dst, ROOT))


if __name__ == "__main__":
    assert not os.path.exists(os.path.join(ROOT, "ZP_INTEGRATED.json")), "already integrated"
    n = len([d for d in os.listdir(os.path.join(ZP, "results")) if os.path.exists(
        os.path.join(ZP, "results", d, f"{d}_rebootstrap2.json"))])
    assert n == 57, f"zp run incomplete ({n} of 57)"
    os.makedirs(SNAP, exist_ok=True)
    for f in SUMMARIES:
        if os.path.exists(os.path.join(ROOT, f)):
            shutil.copy(os.path.join(ROOT, f), os.path.join(SNAP, f))
    mv("results", "results_unsplit")
    mv("results_geo", "results_geo_unsplit")
    mv(os.path.join("data", "workbooks"), os.path.join("data_unsplit", "workbooks"))
    mv(os.path.join("data", "obs_tables"), os.path.join("data_unsplit", "obs_tables"))
    mv(os.path.join("zp", "results"), "results")
    mv(os.path.join("zp", "data", "workbooks"), os.path.join("data", "workbooks"))
    mv(os.path.join("zp", "data", "obs_tables"), os.path.join("data", "obs_tables"))
    json.dump({"date": time.strftime("%Y-%m-%d %H:%M"), "split": "703 and G96 at JD 2458300.5, parts < 30 points dropped",
               "code_snapshot": open(os.path.join(ROOT, "repo_snap", "SNAPSHOT_REV.txt"), encoding="utf-8").read(),
               "kept": ["results_unsplit", "results_geo_unsplit", "data_unsplit", "unsplit_snapshot"]},
              open(os.path.join(ROOT, "ZP_INTEGRATED.json"), "w"), indent=1)
    print("integrated")
