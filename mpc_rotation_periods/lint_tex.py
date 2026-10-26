"""
Static checks of the revised manuscript (no TeX installation is available here):
brace balance, begin/end pairs, labels vs references, citation keys vs .bib, unused .bib entries,
leftover %%%MARKERS%%%, figure files, and non-ASCII characters outside comments.
"""
import os
import re
import sys

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manuscript_revised")
tex = open(os.path.join(D, "mpc_rotation_periods.tex"), encoding="utf-8").read()
bib = open(os.path.join(D, "mpc_rotation_periods.bib"), encoding="utf-8").read()
problems = []

body = re.sub(r"(?<!\\)%.*", "", tex)          # strip comments
depth = 0
for i, ch in enumerate(body):
    if ch == "{" and body[i - 1] != "\\":
        depth += 1
    elif ch == "}" and body[i - 1] != "\\":
        depth -= 1
        if depth < 0:
            problems.append(f"unbalanced '}}' near: {body[max(0, i - 60):i + 10]!r}")
            depth = 0
if depth:
    problems.append(f"{depth} unclosed '{{'")

stack = []
for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", body):
    kind, env = m.groups()
    if kind == "begin":
        stack.append(env)
    elif not stack or stack.pop() != env:
        problems.append(f"\\end{{{env}}} does not match")
if stack:
    problems.append(f"unclosed environments: {stack}")

labels = set(re.findall(r"\\label\{([^}]+)\}", body))
refs = set(re.findall(r"\\ref\{([^}]+)\}", body))
for r in sorted(refs - labels):
    problems.append(f"undefined reference: {r}")
for l in sorted(labels - refs):
    problems.append(f"label never referenced (check): {l}")

keys = set(re.findall(r"@\w+\{([^,\s]+),", bib))
cited = set()
for m in re.finditer(r"\\cite[pt]?\*?(?:\[[^\]]*\])*\{([^}]+)\}", body):
    cited.update(k.strip() for k in m.group(1).split(","))
for k in sorted(cited - keys):
    problems.append(f"citation key not in .bib: {k}")
unused = sorted(keys - cited)

for m in re.findall(r"%%%[A-Z0-9_]+%%%", tex):
    problems.append(f"leftover marker {m}")

for f in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body):
    if not os.path.exists(os.path.join(D, f)):
        problems.append(f"missing figure file {f}")
    if " " in f:
        problems.append(f"space in figure file name {f}")

for ln, line in enumerate(tex.splitlines(), 1):
    code = re.sub(r"(?<!\\)%.*", "", line)
    bad = [c for c in code if ord(c) > 127]
    if bad:
        problems.append(f"non-ASCII characters {set(bad)} on line {ln}")

print("\n".join(problems) if problems else "no problems found")
print(f"{len(labels)} labels, {len(refs)} refs, {len(cited)} cited keys; unused bib entries: {unused}")
sys.exit(1 if any(not p.startswith("label never") for p in problems) else 0)
