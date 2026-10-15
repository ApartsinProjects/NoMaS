# -*- coding: utf-8 -*-
"""Oracle family-routing bound on OddBench.

Question: if we KNEW whether the true-best detector is local-family or
global-family, and picked within that family using the EXISTING NoMaS score,
how much regret would that remove? This caps the whole routing direction.

Invariants checked before any number is reported:
  I1  recomputed regret == stored regret column (harness matches the paper's)
  I2  routing restricted to the full detector set == current NoMaS exactly
  I3  oracle routing can never be worse than... (not true in general: within-family
      argmax can pick worse than global argmax IF the global argmax was already
      the best. So I2 is the real invariant; I3 is reported, not asserted.)
"""
import json
import numpy as np
import pandas as pd

CSV = r"E:\Projects\Submitted\ADRank\results\oddbench.csv"

# Families exactly as the paper defines them (Section 5.4).
LOCAL = {"LOF", "KNN", "CBLOF"}
GLOBAL = {"HBOS", "COPOD", "ECOD", "PCA"}
OTHER = {"IForest", "LODA"}


def fam(d):
    if d in LOCAL:
        return "local"
    if d in GLOBAL:
        return "global"
    return "other"


df = pd.read_csv(CSV)
df = df[df.status == "ok"].copy()
print(f"rows (dataset x seed): {len(df)}   datasets: {df.dataset.nunique()}")

rows = []
for _, r in df.iterrows():
    ta = json.loads(r.true_auc_json)
    sc = json.loads(r.score_json)
    dets = [d for d in ta if d in sc and not pd.isna(ta[d])]
    if len(dets) < 3:
        continue
    best_auc = max(ta[d] for d in dets)
    best_det = max(dets, key=lambda d: ta[d])
    pick_all = max(dets, key=lambda d: sc[d])
    reg_now = best_auc - ta[pick_all]

    # oracle family routing: restrict argmax(score) to the true best's family
    f = fam(best_det)
    in_fam = [d for d in dets if fam(d) == f]
    pick_fam = max(in_fam, key=lambda d: sc[d])
    reg_route = best_auc - ta[pick_fam]

    rows.append(dict(dataset=r.dataset, seed=r.seed,
                     stored_regret=r.regret, reg_now=reg_now, reg_route=reg_route,
                     best_det=best_det, best_fam=f, pick_all=pick_all,
                     pick_all_fam=fam(pick_all), pick_fam=pick_fam,
                     best_auc=best_auc, n=r.n, d=r.d))

R = pd.DataFrame(rows)
print(f"usable rows: {len(R)}")

# ---------------- INVARIANTS ----------------
gap = (R.reg_now - R.stored_regret).abs()
print(f"\nI1  recomputed vs stored regret: max|diff| = {gap.max():.3e}  "
      f"({'PASS' if gap.max() < 1e-9 else 'FAIL'})")

# I2: routing to the full set must equal current
chk = []
for _, r in df.iterrows():
    ta = json.loads(r.true_auc_json); sc = json.loads(r.score_json)
    dets = [d for d in ta if d in sc and not pd.isna(ta[d])]
    if len(dets) < 3:
        continue
    chk.append(max(dets, key=lambda d: sc[d]))
print(f"I2  full-set routing == current pick: "
      f"{'PASS' if list(R.pick_all) == chk else 'FAIL'}")

# ---------------- FAMILY DISTRIBUTION ----------------
print("\n--- where does the truth live vs where does NoMaS go ---")
print("true-best family :"); print(R.best_fam.value_counts(normalize=True).round(3).to_string())
print("NoMaS pick family:"); print(R.pick_all_fam.value_counts(normalize=True).round(3).to_string())

# ---------------- ORACLE BOUND ----------------
print("\n--- ORACLE FAMILY-ROUTING BOUND (OddBench, 187 unseen datasets) ---")
print(f"current NoMaS regret        : {R.reg_now.mean():.4f}   (median {R.reg_now.median():.4f})")
print(f"oracle family-routed regret : {R.reg_route.mean():.4f}   (median {R.reg_route.median():.4f})")
red = 100 * (1 - R.reg_route.mean() / R.reg_now.mean()) if R.reg_now.mean() else 0
print(f"headroom from perfect routing: {R.reg_now.mean()-R.reg_route.mean():+.4f} AUC  ({red:.1f}% of current regret)")

# breakdown by whether the truth is global
for f in ["local", "global", "other"]:
    s = R[R.best_fam == f]
    if len(s):
        print(f"  best={f:6s} n={len(s):4d}  now={s.reg_now.mean():.4f}  routed={s.reg_route.mean():.4f}  "
              f"delta={s.reg_now.mean()-s.reg_route.mean():+.4f}")

# how often does NoMaS already land in the right family?
hit = (R.pick_all_fam == R.best_fam).mean()
print(f"\nNoMaS already picks the right FAMILY: {hit:.3f} of rows")

R.to_csv(r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d\scratchpad\route_rows.csv", index=False)
print("\nwrote route_rows.csv")
