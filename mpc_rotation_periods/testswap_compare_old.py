"""
Check that the test-sample swap (retired.json, six new test asteroids) left the results of the other asteroids unchanged:
deep comparison of every shared entry of the pipeline outputs pulled from the HPC (<staged dir>) with the copies made
before the swap (backup_before_testswap/, or the local file when the backup has none). Numbers are compared with a
relative tolerance; added and removed top-level keys are listed separately.
Usage: python testswap_compare_old.py <staged dir> [rtol]
"""
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
STAGE = sys.argv[1]
RTOL = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-6
BACKUP = os.path.join(ROOT, "backup_before_testswap")
NEW6 = {"249", "486", "870", "1600", "1700", "7750"}
RETIRED = {"1951", "1963", "2134", "2150", "2318"}


def diff(a, b, path, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: only in {'new' if k in b else 'old'}")
            else:
                diff(a[k], b[k], f"{path}/{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: list length {len(a)} -> {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, f"{path}[{i}]", out)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if (isinstance(a, float) and math.isnan(a)) and (isinstance(b, float) and math.isnan(b)):
            return
        if abs(a - b) > RTOL * max(abs(a), abs(b), 1e-12):
            out.append(f"{path}: {a!r} -> {b!r}")
    elif a != b:
        out.append(f"{path}: {str(a)[:80]!r} -> {str(b)[:80]!r}")


def key_kind(k):
    s = str(k).split(":")[-1]
    return "new6" if s in NEW6 else "retired" if s in RETIRED else "other"


report = {}
for dirpath, _, files in os.walk(STAGE):
    for f in files:
        if not f.endswith(".json") or "results" in os.path.relpath(dirpath, STAGE):
            continue
        rel = os.path.relpath(os.path.join(dirpath, f), STAGE)
        new = json.load(open(os.path.join(STAGE, rel), encoding="utf-8"))
        bk = os.path.join(BACKUP, rel.replace(os.sep, "__"))
        src = bk if os.path.exists(bk) else os.path.join(ROOT, rel)
        if not os.path.exists(src):
            report[rel] = {"reference": None}
            continue
        old = json.load(open(src, encoding="utf-8"))
        r = {"reference": os.path.relpath(src, ROOT)}
        if isinstance(old, dict) and isinstance(new, dict):
            r["added"] = sorted((k for k in new if k not in old), key=str)
            r["removed"] = sorted((k for k in old if k not in new), key=str)
            changed = {}
            for k in sorted(set(old) & set(new), key=str):
                out = []
                diff(old[k], new[k], "", out)
                if out:
                    changed[str(k)] = {"kind": key_kind(k), "n": len(out), "first": out[:6]}
            r["changed"] = changed
            r["changed_other"] = sorted(k for k, v in changed.items() if v["kind"] == "other")
        else:
            out = []
            diff(old, new, "", out)
            r["n_diff"], r["first"] = len(out), out[:12]
        report[rel] = r
        print(f"== {rel}  (vs {r['reference']})")
        if "changed" in r:
            print(f"   added {r['added'][:20]}  removed {r['removed'][:20]}")
            print(f"   changed: {len(r['changed'])} keys, of them not new/retired: {r['changed_other'][:40]}")
            for k in r["changed_other"][:12]:
                print(f"     {k}: {r['changed'][k]['n']} diffs, e.g. {r['changed'][k]['first'][:3]}")
        else:
            print(f"   {r['n_diff']} diffs, e.g. {r['first'][:6]}")
json.dump(report, open(os.path.join(ROOT, "testswap_compare_old.json"), "w", encoding="utf-8"), indent=1)
