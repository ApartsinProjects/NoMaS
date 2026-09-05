# -*- coding: utf-8 -*-
"""Controlled synthesis test (user's process): make COMBINATION-weird anomalies with ZERO individual
weirdness -> replace k features of a normal with values taken from OTHER real normals (feature
permutation). Every value stays a real typical value; only the joint combination is broken. Sweep k.
Does this make the probe pick the LOCAL oracle on the tail where marginal resampling failed?"""
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
def gen_combo(Xn, ns, frac, seed=0):
    """dependency anomaly: replace frac*d features of a base normal with feature-values sampled from
       OTHER random normals (permutation). Marginals stay exactly real; joint dependency broken."""
    rng = np.random.default_rng(seed); n, d = Xn.shape; out = Xn[rng.integers(n, size=ns)].copy()
    for r in range(ns):
        for j in rng.choice(d, max(1, int(frac * d)), replace=False):
            out[r, j] = Xn[rng.integers(n), j]
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
TARGETS = [("oddbench", "RacingMotion", "local"), ("ovrbench", "AbstractAnomalyBatch", "local"), ("oddbench", "DiamondClarity", "local"),
           ("oddbench", "TreadmillStatus", "local"), ("ovrbench", "MalariaDetection", "global"), ("oddbench", "ProtestViolence", "global"),
           ("ovrbench", "SpectralImaging", "global")]
FRACS = [0.3, 0.6, 1.0]
print("=== CONTROLLED + OOD-FILTER (drop in-cluster synthetics), regret of probe pick (in-dist%) ===")
print(f"  {'dataset':22s} {'fam':6s} {'oracle':>6s} " + " ".join(f"k={f:>4}" for f in FRACS) + "   [oracle-family / pick@k=0.6]")
for corp, name, tf in TARGETS:
    try: Xtr, Xval, Xtn, Xa = get3(corp, name)
    except Exception: continue
    Xh = harden(Xtr, np.vstack([Xval, Xtn]), Xa)
    if len(Xh) < 15: continue
    Xte = np.vstack([Xtn, Xh]); yte = np.r_[np.zeros(len(Xtn)), np.ones(len(Xh))]; base = yte.mean()
    sc = StandardScaler().fit(Xtr); Zv = sc.transform(Xval); scale = np.median(NearestNeighbors(n_neighbors=2).fit(Zv).kneighbors(Zv)[0][:, 1]) + 1e-9; nn = NearestNeighbors(n_neighbors=1).fit(Zv)
    ap, SV = {}, {}
    for vn, ct in TAB_POOL:
        try:
            m = ct()
            with contextlib.redirect_stdout(io.StringIO()): m.fit(Xtr)
            st = np.asarray(m.decision_function(Xte), float); sv = np.asarray(m.decision_function(Xval), float)
            if not (np.all(np.isfinite(st)) and np.nanstd(st) > 1e-12): continue
            ap[vn] = (average_precision_score(yte, st) - base) / (1 - base + 1e-12); SV[vn] = (m, sv)
        except Exception: pass
    if len(ap) < 6: continue
    av = pd.Series(ap); orc = av.max(); cells = []; pick06 = None
    for fr in FRACS:
        syn = gen_combo(Xval, 600, fr); dd = nn.kneighbors(sc.transform(syn))[0][:, 0]
        keep = dd > scale                                          # FILTER: drop synthetics inside the normal cluster
        raw_ind = (dd <= scale).mean()
        if keep.sum() >= 30: syn = syn[keep][:300]
        else: syn = syn[:300]
        ind = raw_ind
        yv = np.r_[np.zeros(len(Xval)), np.ones(len(syn))]; sep = {}
        for vn, (m, sv) in SV.items():
            s1 = np.asarray(m.decision_function(syn), float)
            if np.nanstd(np.r_[sv, s1]) > 1e-12: sep[vn] = roc_auc_score(yv, np.r_[sv, s1])
        pk = max(sep, key=lambda v: sep[v]); cells.append(f"{orc-ap[pk]:.2f}({ind:.0%})")
        if fr == 0.6: pick06 = f"{pk[:11]}[{fam(pk)[:3]}]"
    print(f"  {name[:22]:22s} {tf:6s} {orc:6.2f} " + " ".join(f"{c:>11s}" for c in cells) + f"   [{fam(av.idxmax())[:3]} / {pick06}]")
print("\n  regret DROPS at high k on LOCAL datasets => permutation makes the combination-weird anomalies they need")
