"""Build the Overleaf upload zip: the manuscript, its bibliography, the AASTeX class/style files and exactly the
figure files referenced by \\includegraphics (no generated table snippets, PNG previews or reference figures)."""
import os
import re
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
MS = os.path.join(ROOT, "manuscript_revised")
tex = open(os.path.join(MS, "mpc_rotation_periods.tex"), encoding="utf-8").read()
figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
files = ["mpc_rotation_periods.tex", "mpc_rotation_periods.bib", "aastex701.cls", "aasjournalv7.bst", "orcid-ID.png"]
files += figs
missing = [f for f in files if not os.path.exists(os.path.join(MS, f))]
if missing:
    raise SystemExit(f"missing files: {missing}")
out = os.path.join(ROOT, "mpc_rotation_periods_overleaf.zip")
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        z.write(os.path.join(MS, f), f)
print(f"{out}: {len(files)} files")
for f in files:
    print("  ", f)
