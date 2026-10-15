# -*- coding: utf-8 -*-
"""Rebuild the ADReal leaderboard from the surviving Modal artifacts and check it against the numbers
stated in the paper. Inputs (all under artifacts/modal/):
  bv2/_index.csv        per-task n_h (test anomalies)          [adrank-bundle volume]
  probe1000/*.jsonl     per-task test ap_norm, all 32 detectors [adrank-probe1000-results volume]
Definitions follow the recovered finalize_v3.py: oracle = max ap over the pool; random-pick regret =
oracle - mean ap; keep = oracle > 0.05 & reg_random > 0.02 & n_h >= 100. Each check states the paper's
value in advance; the script prints observed vs expected and PASS/DIFF."""
import json, glob, os, sys
import numpy as np, pandas as pd

SRC = sys.argv[1] if len(sys.argv) > 1 else "probe1000"
A = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modal")
J = {}
for f in glob.glob(os.path.join(A, SRC, "*.jsonl")):
    for ln in open(f):
        r = json.loads(ln)
        if r.get("ap"): J[r["key"]] = r["ap"]
idx = pd.read_csv(os.path.join(A, "bv2", "_index.csv")); nh = dict(zip(idx.key, idx.n_h))
DEEP = lambda d: d.startswith("D_")

def src(k):
    c = k.split("__")[0].lower()
    return {"ovrbench": "OvrBench", "oddbench": "OddBench", "tsbad_m": "TS", "adbench": "ADBench", "dami": "DAMI"}.get(c, c)

rows = []
for k, ap in J.items():
    ap = {d: v for d, v in ap.items() if v is not None and np.isfinite(v)}
    cls = {d: v for d, v in ap.items() if not DEEP(d)}; dp = {d: v for d, v in ap.items() if DEEP(d)}
    o = max(ap.values())
    rows.append(dict(key=k, source=src(k), n_h=nh.get(k, -1), oracle=o, oracle_cls=max(cls.values()),
                     oracle_deep=max(dp.values()) if dp else np.nan, reg_random=o - np.mean(list(ap.values())),
                     reg_random_cls=o - np.mean(list(cls.values())), reg_random_deep=(o - np.mean(list(dp.values()))) if dp else np.nan,
                     reg_fixed=(o - cls["LOF_k10"]) if "LOF_k10" in cls else np.nan,
                     best_is_cls=max(ap, key=ap.get) in cls))
df = pd.DataFrame(rows)
keep = df[(df.oracle > 0.05) & (df.reg_random > 0.02) & (df.n_h >= 100)]

# per-detector mean ap over kept tasks (for the "fixed detector" and Figure 5 ordering)
dets = sorted({d for k in keep.key for d in J[k]})
mean_ap = {d: np.nanmean([J[k].get(d, np.nan) for k in keep.key]) for d in dets}
cls_reg = {d: np.nanmean([keep.set_index("key").oracle[k] - J[k][d] for k in keep.key if d in J[k]]) for d in dets if not DEEP(d)}
fixed_name = min(cls_reg, key=cls_reg.get)

checks = [
    ("constructed tasks scored", len(df), 173, 0),
    ("tasks in benchmark (keep filter)", len(keep), 151, 0),
    ("OvrBench tasks", int((keep.source == "OvrBench").sum()), 66, 0),
    ("OddBench tasks", int((keep.source == "OddBench").sum()), 39, 0),
    ("TSB-AD tasks", int((keep.source == "TS").sum()), 38, 0),
    ("ADBench tasks", int((keep.source == "ADBench").sum()), 5, 0),
    ("DAMI tasks", int((keep.source == "DAMI").sum()), 3, 0),
    ("best fixed classical detector is LOF_k10", fixed_name == "LOF_k10", True, 0),
    ("fixed-detector regret (classical/full)", round(keep.reg_fixed.mean(), 3), 0.130, 0.002),
    ("random-pick regret, full pool", round(keep.reg_random.mean(), 3), 0.220, 0.002),
    ("random-pick regret, classical pool", round(keep.reg_random_cls.mean(), 3), 0.219, 0.002),
    ("random-pick regret, deep pool", round(keep.reg_random_deep.mean(), 3), 0.223, 0.002),
    ("tasks where a classical detector is oracle", int(keep.best_is_cls.sum()), 118, 0),
    ("deep pool raises the oracle by", round((keep.oracle - keep.oracle_cls).mean(), 3), 0.021, 0.002),
    ("best deep trails best classical by", round((keep.oracle_cls - keep.oracle_deep).mean(), 3), 0.071, 0.002),
    ("top detector by mean ap_norm", max(mean_ap, key=mean_ap.get), "LOF_k10", 0),
]
npass = 0
print(f"{'check':45s} {'observed':>10s} {'paper':>10s}")
for name, obs, exp, tol in checks:
    ok = (abs(obs - exp) <= tol) if isinstance(exp, float) else (obs == exp)
    npass += ok
    print(f"{name:45s} {str(obs):>10s} {str(exp):>10s}  {'PASS' if ok else 'DIFF'}")
print(f"\n{npass}/{len(checks)} checks pass")
df.to_csv(os.path.join(os.path.dirname(A), f"LEADERBOARD_REBUILT_{SRC}.csv"), index=False)
print("wrote artifacts/LEADERBOARD_REBUILT.csv")
