"""
Is the committed asteroid_ls.py (Combined-dataset-period-analysis, git commit 5a7642a) equivalent to the sourceless
repo_snap/asteroid_ls.pyc used by the period pipeline? Compiles the committed source with the same CPython 3.12 and
compares every code object (bytecode, constants, names) with the pyc, function by function.
Usage: python compare_ls_source.py <source.py>
"""
import marshal
import sys
import types


def walk(co, prefix=""):
    name = f"{prefix}.{co.co_name}" if prefix else co.co_name
    yield name, co
    for c in co.co_consts:
        if isinstance(c, types.CodeType):
            yield from walk(c, name)


def sig(co):
    consts = tuple(c if not isinstance(c, types.CodeType) else ("<code>", c.co_name) for c in co.co_consts)
    return co.co_code, consts, co.co_names, co.co_varnames


pyc = marshal.loads(open(r"C:\Users\kn18001\Documents\Asteroids\MPC_redo\repo_snap\asteroid_ls.pyc", "rb").read()[16:])
src = compile(open(sys.argv[1], encoding="utf-8-sig").read(), "asteroid_ls.py", "exec")
A = dict(walk(pyc))
B = dict(walk(src))
same = [k for k in A if k in B and sig(A[k]) == sig(B[k])]
diff = [k for k in A if k in B and sig(A[k]) != sig(B[k])]
only_pyc = [k for k in A if k not in B]
only_src = [k for k in B if k not in A]
print(f"code objects: pyc {len(A)}, source {len(B)}; identical {len(same)}, different {len(diff)}, "
      f"only in pyc {len(only_pyc)}, only in source {len(only_src)}")
for k in diff:
    a, b = A[k], B[k]
    why = [w for w, x, y in (("bytecode", a.co_code, b.co_code), ("consts", sig(a)[1], sig(b)[1]),
                             ("names", a.co_names, b.co_names), ("varnames", a.co_varnames, b.co_varnames)) if x != y]
    print("  DIFF", k, why)
for k in only_pyc:
    print("  ONLY PYC", k)
for k in only_src:
    print("  ONLY SOURCE", k)
