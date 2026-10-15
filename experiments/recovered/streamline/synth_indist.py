# -*- coding: utf-8 -*-
"""Test the user's hypothesis: does our synthetic create IN-DISTRIBUTION (normal-like) points, and do we
then punish the CORRECT detector for correctly not flagging them? For tail datasets we failed: (1) frac
of synthetics whose nearest normal is closer than the normal-normal NN scale (=normal-like); (2) does
FILTERING synthetics to OOD-only (nn-dist > normal scale) move the probe's pick toward the oracle?"""
import os, sys, io, contextlib, warnings
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import average_precision_score, roc_auc_score
warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"; S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
D = os.path.join(S, "scratchpad", "streamline"); sys.path.insert(0, os.path.join(S, "scratchpad")); sys.path.insert(0, os.path.join(ROOT, "src"))
import adrank.pipeline as PP
from dev_common import TAB_POOL
LOCAL = ("LOF", "KNN", "CBLOF"); GLOBAL = ("HBOS", "COPOD", "ECOD", "PCA")
fam = lambda v: "local" if str(v).startswith(LOCAL) else ("global" if str(v).startswith(GLOBAL) else "other")
Q = 0.05
OBJ = {}
for sub in ("adbench", "dami"):
    dd = os.path.join(ROOT, "data", sub)
    if os.path.isdir(dd):
        for ds in PP.load_npz_dir(dd): OBJ[(sub, str(ds.name)[:40])] = (np.nan_to_num(np.asarray(ds.X, float)), np.asarray(ds.y, int).ravel())
def severity(Xtr, X, bins=30):
    n, d = X.shape; sev = np.zeros(n)
    for j in range(d):
        tr = np.sort(Xtr[:, j])
        if tr[-1] - tr[0] < 1e-12: continue
        m = len(tr); fb = np.searchsorted(tr, X[:, j], side="right") / m
        cnt, edges = np.histogram(Xtr[:, j], bins=bins); dens = cnt / max(cnt.sum(), 1); b = np.clip(np.digitize(X[:, j], edges[1:-1]), 0, bins - 1)
        sev = np.maximum(sev, np.maximum(-np.log(np.maximum(2 * np.minimum(fb, 1 - fb), 1.0 / (2 * m))), -np.log(dens[b] + 1e-9)))
    return sev
def harden(Xtr, Xhold, Xa):
    tho = np.quantile(severity(Xtr, Xhold), 1 - Q); so_a = severity(Xtr, Xa)
    try:
        sc = StandardScaler().fit(Xtr); pca = PCA(n_components=0.95, whiten=True, random_state=0).fit(sc.transform(Xtr))
        Ptr, Ph, Pa = pca.transform(sc.transform(Xtr)), pca.transform(sc.transform(Xhold)), pca.transform(sc.transform(Xa))
        thp = np.quantile(severity(Ptr, Ph), 1 - Q); sp_a = severity(Ptr, Pa)
    except Exception:
        sp_a = np.zeros(len(Xa)); thp = np.inf
    return Xa[~((so_a > tho) | (sp_a > thp))]
def _H(Xn, cap=98):
    H = []
    for j in range(Xn.shape[1]):
        col = Xn[:, j]; cnt, edges = np.histogram(col, bins=30); dens = cnt / max(cnt.sum(), 1)
        lo, hi = np.percentile(col, 100 - cap), np.percentile(col, cap); ctr = (edges[:-1] + edges[1:]) / 2
        H.append((edges, dens, (cnt > 0) & (ctr >= lo) & (ctr <= hi)))
    return H
def gen_beta(Xn, ns, beta, frac=0.4, seed=0):
    rng = np.random.default_rng(seed); n, d = Xn.shape; H = _H(Xn); out = np.empty((ns, d))
    for r in range(ns):
        base = Xn[rng.integers(n)].copy()
        for j in rng.choice(d, max(1, int(frac * d)), replace=False):
            edges, dens, allowed = H[j]; w = np.where(allowed, np.maximum(dens, 1e-6) ** beta, 0.0)
            base[j] = Xn[rng.integers(n), j] if w.sum() == 0 else rng.uniform(*edges[[(b := rng.choice(len(w), p=w / w.sum())), b + 1]])
        out[r] = base
    return out
def get3(corp, name):
    if corp in ("oddbench", "ovrbench"):
        d = np.load(os.path.join(ROOT, "data", corp, name + ".npz"), allow_pickle=True)
        X = np.nan_to_num(np.vstack([np.asarray(d["train"], float), np.asarray(d["test"], float)])); y = np.concatenate([np.asarray(d["train_labels"]).ravel(), np.asarray(d["test_labels"]).ravel()]).astype(int); Xn, Xa = X[y == 0], X[y == 1]
    else:
        X, y = OBJ[(corp, name)]; Xn, Xa = X[y == 0], X[y == 1]
    r = np.random.default_rng(0)
    if len(Xn) > 6000: Xn = Xn[r.choice(len(Xn), 6000, replace=False)]
    idx = np.arange(len(Xn)); r.shuffle(idx); a, b = int(0.6 * len(idx)), int(0.8 * len(idx)); return Xn[idx[:a]], Xn[idx[a:b]], Xn[idx[b:]], Xa
TARGETS = [("oddbench", "RacingMotion"), ("ovrbench", "AbstractAnomalyBatch"), ("oddbench", "DiamondClarity"),
           ("oddbench", "TreadmillStatus"), ("ovrbench", "MalariaDetection"), ("oddbench", "ProtestViolence")]
print("=== are synthetics in-distribution? does OOD-filtering fix the probe pick? ===\n")
for corp, name in TARGETS:
    try: Xtr, Xval, Xtn, Xa = get3(corp, name)
    except Exception: continue
    Xh = harden(Xtr, np.vstack([Xval, Xtn]), Xa)
    if len(Xh) < 15: continue
    Xte = np.vstack([Xtn, Xh]); yte = np.r_[np.zeros(len(Xtn)), np.ones(len(Xh))]; base = yte.mean()
    sc = StandardScaler().fit(Xtr); Zv = sc.transform(Xval)
    scale = np.median(NearestNeighbors(n_neighbors=2).fit(Zv).kneighbors(Zv)[0][:, 1]) + 1e-9
    nn = NearestNeighbors(n_neighbors=1).fit(Zv)
    syn1, syn4 = gen_beta(Xval, 300, 1.0), gen_beta(Xval, 300, -4.0)
    d1 = nn.kneighbors(sc.transform(syn1))[0][:, 0]; d4 = nn.kneighbors(sc.transform(syn4))[0][:, 0]
    indist1, indist4 = (d1 <= scale).mean(), (d4 <= scale).mean()
    keep1, keep4 = d1 > scale, d4 > scale                       # OOD-only filter
    ap, sepU, sepF = {}, {}, {}
    yvU = np.r_[np.zeros(len(Xval)), np.ones(len(syn1))]
    yvF = np.r_[np.zeros(len(Xval)), np.ones(int(keep1.sum()))]
    for vn, ct in TAB_POOL:
        try:
            m = ct()
            with contextlib.redirect_stdout(io.StringIO()): m.fit(Xtr)
            st = np.asarray(m.decision_function(Xte), float); sv = np.asarray(m.decision_function(Xval), float); s1 = np.asarray(m.decision_function(syn1), float)
            if not (np.all(np.isfinite(st)) and np.nanstd(st) > 1e-12): continue
            ap[vn] = (average_precision_score(yte, st) - base) / (1 - base + 1e-12)
            sepU[vn] = roc_auc_score(yvU, np.r_[sv, s1]) if np.nanstd(np.r_[sv, s1]) > 1e-12 else 0.5
            if keep1.sum() >= 10: sepF[vn] = roc_auc_score(yvF, np.r_[sv, s1[keep1]]) if np.nanstd(np.r_[sv, s1[keep1]]) > 1e-12 else 0.5
        except Exception: pass
    av = pd.Series(ap); oracle = av.idxmax(); pickU = max(sepU, key=lambda v: sepU[v]); pickF = max(sepF, key=lambda v: sepF[v]) if sepF else pickU
    print(f"{name:22s} n_hard={len(Xh):4d}  in-dist frac: beta1={indist1:.0%} beta-4={indist4:.0%}   (high=many synthetics are normal-like)")
    print(f"   oracle {oracle[:13]:13s}[{fam(oracle)[:3]}] ap={av.max():.2f}")
    print(f"   probe UNFILTERED picks {pickU[:13]:13s}[{fam(pickU)[:3]}] ap={ap.get(pickU,float('nan')):.2f} reg={av.max()-ap.get(pickU,0):.2f}")
    print(f"   probe OOD-FILTERED picks {pickF[:13]:13s}[{fam(pickF)[:3]}] ap={ap.get(pickF,float('nan')):.2f} reg={av.max()-ap.get(pickF,0):.2f}   {'FIXED' if ap.get(pickF,0)>ap.get(pickU,0)+0.1 else ''}\n")
print("KEY: if OOD-filtered pick has much lower regret than unfiltered, in-distribution synthetics were misleading the probe")
