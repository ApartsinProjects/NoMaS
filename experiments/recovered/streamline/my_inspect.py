# -*- coding: utf-8 -*-
"""Deep inspection of tabular datasets where SPARC loses to EM (oracle high, so anomalies ARE separable).
Per dataset: detector picks (real-best / SPARC a1,a4 / EM), anomaly character, leakage/bug checks,
metric verification, and why the probe misranks the winner."""
import numpy as np, json, glob, os, sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
sys.path.insert(0, r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d\scratchpad")
from hadb_round2_common import _em_auc, _mv_auc
D = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d\scratchpad\streamline"
J = {}
for fn in glob.glob(os.path.join(D, "deepres_v2", "shard_*.jsonl")):
    for ln in open(fn):
        try:
            r = json.loads(ln)
            if "ap" in r: J[r["key"]] = r
        except Exception: pass
t = np.linspace(0, 100, 1000); alpha = np.linspace(0.9, 0.999, 1000)
def key32(A): return np.round(A.astype(np.float32).astype(np.float64), 6)
for key in ["oddbench__AquaticMacroInverts", "ovrbench__RealEstateData", "ovrbench__BoundingBoxClassification", "ovrbench__MalariaDetection"]:
    r = J.get(key); z = np.load(os.path.join(D, "bundle_v2", key + ".npz")); sc = np.load(os.path.join(D, "deepres_v2", f"sc_{key}.npz"))
    Xtr, Xval, Xtn, Xh = [z[k].astype(float) for k in ("Xtr", "Xval", "Xtn", "Xh")]
    names = [n for n in r["ap"] if n in r["a1"] and n in r["a4"]]; ap = r["ap"]; a1 = r["a1"]; a4 = r["a4"]
    print("=" * 90); print(f"{key}   d={Xtr.shape[1]}  n_tr={len(Xtr)} n_tn={len(Xtn)} n_h={len(Xh)}  base_rate={len(Xh)/(len(Xtn)+len(Xh)):.3f}")
    # picks
    rb = max(names, key=lambda v: ap[v]); cls = [n for n in names if not n.startswith("D_")]
    p1 = max(cls, key=lambda v: a1[v]); pe = max(cls, key=lambda v: a4[v])
    # EM/MV
    emv, mvv = {}, {}
    for n in names:
        if f"val__{n}" not in sc.files: continue
        sv = sc[f"val__{n}"].astype(float); su = sc[f"uni__{n}"].astype(float); su = np.nan_to_num(su, nan=float(np.nanmedian(su[np.isfinite(su)])) if np.isfinite(su).any() else 0.0)
        emv[n] = _em_auc(t, 1.0, -su, -sv, 1000); mvv[n] = _mv_auc(alpha, 1.0, -su, -sv, 1000)
    em_pick = max(emv, key=lambda v: emv[v]); mv_pick = min(mvv, key=lambda v: mvv[v])
    top5 = sorted(names, key=lambda v: -ap[v])[:5]
    print("  top-5 by real ap_norm:", [(n, round(ap[n], 2)) for n in top5])
    print(f"  real-best={rb}(ap {ap[rb]:.2f})  |  SPARC a1-best={p1}(ap {ap[p1]:.2f})  a4-best={pe}(ap {ap[pe]:.2f})  |  EM-pick={em_pick}(ap {ap[em_pick]:.2f})  MV-pick={mv_pick}(ap {ap[mv_pick]:.2f})")
    # where does the winner rank on each criterion?
    def rankof(d, v, hi=True): return sorted(names, key=lambda x: -d[x] if hi else d[x]).index(v) + 1
    print(f"  winner {rb} ranks: a1 #{rankof(a1,rb)}/{len(names)}  a4 #{rankof(a4,rb)}  EM #{rankof(emv,rb) if rb in emv else '-'}  ap-real #1  (a1[{rb}]={a1[rb]:.2f} a4[{rb}]={a4[rb]:.2f})")
    # metric verify: recompute ap for winner
    from importlib import import_module
    # anomaly character
    Z = StandardScaler().fit(Xtr); Zh = Z.transform(Xh); Ztn = Z.transform(Xtn)
    marg_h = np.abs(Zh).max(1); k = max(1, int(np.searchsorted(np.cumsum(PCA().fit(Z.transform(Xtr)).explained_variance_ratio_), 0.9) + 1)); pk = PCA(k).fit(Z.transform(Xtr))
    resid = lambda A: np.sqrt(((A - pk.inverse_transform(pk.transform(A))) ** 2).sum(1))
    print(f"  anomaly character: marginal max|z| median={np.median(marg_h):.2f} (normals {np.median(np.abs(Ztn).max(1)):.2f})  off-manifold resid median={np.median(resid(Zh)):.2f} (normals {np.median(resid(Ztn)):.2f})")
    uniq = [len(np.unique(Xtr[:, j])) for j in range(Xtr.shape[1])]
    print(f"  feature cardinality: min={min(uniq)} median={int(np.median(uniq))} max={max(uniq)}  constant-features={sum(1 for u in uniq if u<=1)}  NaN={int(np.isnan(np.vstack([Xtr,Xh])).sum())}")
    # bug/leak checks
    trset = set(map(tuple, key32(Xtr)))
    leak_tn = sum(1 for x in key32(Xtn) if tuple(x) in trset); leak_h = sum(1 for x in key32(Xh) if tuple(x) in trset)
    # val vs test-normal distribution match (mean |standardized-mean difference|)
    dist = np.abs(Z.transform(Xval).mean(0) - Ztn.mean(0)).mean()
    print(f"  LEAK: Xtr-Xtn shared rows={leak_tn}  anomaly=normal rows={leak_h}   val-vs-testnormal mean|z-shift|={dist:.3f}")
