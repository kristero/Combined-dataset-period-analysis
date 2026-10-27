SSH="ssh -q -o BatchMode=yes"
mkdir -p ~/pull/step2b
rsync -a -e "$SSH" luhpc:mpc_redo/review_alt_period.json luhpc:mpc_redo/hpc/alt_collect.out ~/pull/step2b/ 2>&1 | grep -v "micromamba/bin/micromamba: No such"
python3 - <<'EOF'
import json, os
r = json.load(open(os.path.expanduser("~/pull/step2b/review_alt_period.json")))
for n, d in r.items():
    for p, v in d.items():
        st = v.get("scan_test") or {}
        print(n, p, "chi2", round(v.get("chi2_ratio_alt_over_adopted", 0), 3), "unique", st.get("unique"), round(st.get("chi2_ratio_next") or 0, 4), round(st.get("threshold") or 0, 4),
              "mirror", v["only_mirror_ambiguity"], "dark", v.get("dark_pct"),
              {k: round(x.get("angle_maxI_to_spin_deg", -1), 1) for k, x in v["principal_axis"].items()},
              [(w["areas"], w["lambda"], w["beta"], round(w["ratio"], 4)) for w in v["within"]])
EOF
