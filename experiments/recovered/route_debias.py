# -*- coding: utf-8 -*-
"""Family-offset debiasing: the pseudo-task systematically penalizes global
detectors. Add ONE scalar bonus to the global family's z-scored pseudo-score
(and one to 'other'), then pick argmax as usual.

No classifier. Two parameters. Honest test: fit the offsets on one half of the
DATASETS, evaluate frozen on the held-out half (grouped, so seeds cannot leak).

Invariant: delta = 0 must reproduce current NoMaS regret exactly.
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

CSV = r"E:\Projects\Submitted\ADRank\results\oddbench.csv"
LOCAL = {"LOF", "KNN", "CBLOF"}
GLOBAL = {"HBOS", "COPOD", "ECOD", "PCA"}


def fam(d):
    return "local" if d in LOCAL else ("global" if d in GLOBAL else "other")


df = pd.read_csv(CSV)
df = df[df.status == "ok"]

rows = []
for _, r in df.iterrows():
    ta = json.loads(r.true_auc_json); sc = json.loads(r.score_json)
    dets = [d for d in ta if d in sc and not pd.isna(ta[d])]
    if len(dets) < 3:
        continue
    s = np.array([sc[d] for d in dets], float)
    z = (s - s.mean()) / (s.std() + 1e-9)
    rows.append(dict(dataset=r.dataset, dets=dets, z=dict(zip(dets, z)), ta=ta,
                     best=max(ta[d] for d in dets)))
print(f"rows={len(rows)} datasets={len(set(r['dataset'] for r in rows))}")


def regret(rowset, dg, do):
    out = []
    for r in rowset:
        adj = {d: r["z"][d] + (dg if fam(d) == "global" else (do if fam(d) == "other" else 0.0))
               for d in r["dets"]}
        pick = max(r["dets"], key=lambda d: adj[d])
        out.append(r["best"] - r["ta"][pick])
    return np.array(out)


base = regret(rows, 0.0, 0.0)
print(f"\nINVARIANT delta=0 regret = {base.mean():.4f}  (must equal current NoMaS)")

# ---- sweep on ALL data (upper bound, in-sample) ----
grid = np.arange(0.0, 2.01, 0.1)
best = None
for dg in grid:
    for do in grid:
        m = regret(rows, dg, do).mean()
        if best is None or m < best[0]:
            best = (m, dg, do)
print(f"in-sample best: delta_global={best[1]:.1f} delta_other={best[2]:.1f} -> regret {best[0]:.4f} "
      f"({100*(1-best[0]/base.mean()):+.1f}% vs base)")

# ---- HONEST test: fit offsets on half the datasets, freeze on the other half ----
ds = sorted(set(r["dataset"] for r in rows))
rng = np.random.RandomState(0)
folds = 5
perm = rng.permutation(len(ds))
assign = {d: perm[i] % folds for i, d in enumerate(ds)}

held_base, held_deb, chosen = [], [], []
for k in range(folds):
    tr = [r for r in rows if assign[r["dataset"]] != k]
    te = [r for r in rows if assign[r["dataset"]] == k]
    b = None
    for dg in grid:
        for do in grid:
            m = regret(tr, dg, do).mean()
            if b is None or m < b[0]:
                b = (m, dg, do)
    chosen.append((b[1], b[2]))
    held_base.append(regret(te, 0.0, 0.0))
    held_deb.append(regret(te, b[1], b[2]))

hb = np.concatenate(held_base); hd = np.concatenate(held_deb)
print(f"\n--- HELD-OUT (5-fold grouped by dataset) ---")
print(f"offsets chosen per fold: {chosen}")
print(f"current NoMaS   : {hb.mean():.4f}")
print(f"family-debiased : {hd.mean():.4f}   ({100*(1-hd.mean()/hb.mean()):+.1f}%)")
d = hb - hd
print(f"wins={int((d>0).sum())} losses={int((d<0).sum())} ties={int((d==0).sum())}")
if np.any(d != 0):
    print(f"paired Wilcoxon p = {wilcoxon(hb, hd).pvalue:.2e}")
