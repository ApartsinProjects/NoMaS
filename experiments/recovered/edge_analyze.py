# -*- coding: utf-8 -*-
"""Edge-stratified scoring: does it beat the unstratified baseline?

Scoring variants (all label-free; the edge tag comes from X only):
  V0 baseline   mean pseudo_auc over ALL groups                (current NoMaS proxy)
  V3 control    mean of WITHIN-GROUP z-scores over ALL groups  (z-scoring control)
  V1 family     globals scored on EDGE groups, locals on INTERIOR, other on all
  V2 agnostic   each detector takes its BEST stratum (max over strata of mean z)
                -- needs no hand-coded family list

INVARIANT: with a single stratum, V1 must equal V3 exactly.
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

CSV = sys.argv[1] if len(sys.argv) > 1 else \
    r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d\edge_full_dami.csv"
TAG = sys.argv[2] if len(sys.argv) > 2 else "edge_centroid"
LOCAL = {"LOF", "KNN", "CBLOF"}
GLOBAL = {"HBOS", "COPOD", "ECOD", "PCA"}
fam = lambda d: "local" if d in LOCAL else ("global" if d in GLOBAL else "other")

D = pd.read_csv(CSV)
print(f"rows={len(D)} datasets={D.dataset.nunique()} arms={sorted(D.arm.unique())}")

# z-score each detector's pseudo_auc WITHIN a group (comparable across groups)
D["z"] = D.groupby(["dataset", "arm", "group"]).pseudo_auc.transform(
    lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-9))
D["fam"] = D.detector.map(fam)


def regret_of(pick, tr):
    return max(tr.values()) - tr[pick]


def evaluate(sub, include_noise=True):
    """sub = one (dataset, arm) block -> dict of variant -> regret"""
    if not include_noise:
        sub = sub[sub.group != -1]
    if sub.empty:
        return None
    tr = sub.groupby("detector").true_auc.first().to_dict()
    if len(tr) < 5:
        return None
    # stratify groups by the edge tag (tertiles within this block)
    g = sub.groupby("group")[TAG].first()
    if g.nunique() < 3:
        return None
    hi, lo = g.quantile(0.67), g.quantile(0.33)
    edge_g = set(g[g >= hi].index); int_g = set(g[g <= lo].index)
    if not edge_g or not int_g:
        return None

    v0 = sub.groupby("detector").pseudo_auc.mean()
    v3 = sub.groupby("detector").z.mean()
    z_edge = sub[sub.group.isin(edge_g)].groupby("detector").z.mean()
    z_int = sub[sub.group.isin(int_g)].groupby("detector").z.mean()

    v1 = {}
    for d in tr:
        f = fam(d)
        src = z_edge if f == "global" else (z_int if f == "local" else v3)
        v1[d] = src.get(d, v3.get(d, np.nan))
    v2 = {d: np.nanmax([z_edge.get(d, np.nan), z_int.get(d, np.nan)]) for d in tr}

    out = {}
    for name, sc in [("V0", v0.to_dict()), ("V3", v3.to_dict()), ("V1", v1), ("V2", v2)]:
        sc = {d: v for d, v in sc.items() if d in tr and np.isfinite(v)}
        if not sc:
            return None
        out[name] = regret_of(max(sc, key=sc.get), tr)
    out["best_fam"] = fam(max(tr, key=tr.get))
    out["n_groups"] = sub.group.nunique()
    return out


for arm in sorted(D.arm.unique()):
    for noise in ([True, False] if arm == "hdbscan" else [True]):
        recs = []
        for ds, sub in D[D.arm == arm].groupby("dataset"):
            r = evaluate(sub, include_noise=noise)
            if r:
                r["dataset"] = ds
                recs.append(r)
        if not recs:
            print(f"\n=== arm={arm} noise={noise}: no usable datasets ==="); continue
        R = pd.DataFrame(recs)
        lbl = f"arm={arm}" + ("" if arm != "hdbscan" else f" noise={'in' if noise else 'out'}")
        print(f"\n=== {lbl}  (datasets={len(R)}, mean groups={R.n_groups.mean():.1f}) ===")
        print(f"  best-family mix: {R.best_fam.value_counts().to_dict()}")
        for v in ["V0", "V3", "V1", "V2"]:
            print(f"  {v} regret = {R[v].mean():.4f}")
        for v in ["V1", "V2"]:
            d = R.V0 - R[v]
            w = wilcoxon(R.V0, R[v]).pvalue if np.any(d != 0) else np.nan
            print(f"  {v} vs V0: delta={d.mean():+.4f} ({100*d.mean()/max(R.V0.mean(),1e-9):+.1f}%) "
                  f"wins={int((d>0).sum())} losses={int((d<0).sum())} ties={int((d==0).sum())} p={w:.3f}")
        gl = R[R.best_fam == "global"]
        if len(gl):
            print(f"  [global-best subset n={len(gl)}] V0={gl.V0.mean():.4f} "
                  f"V1={gl.V1.mean():.4f} V2={gl.V2.mean():.4f}")
        print(R[["dataset", "best_fam", "V0", "V1", "V2"]].round(4).to_string(index=False))
