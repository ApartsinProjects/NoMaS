# -*- coding: utf-8 -*-
"""Inspect the HIGH-REGRET tail (co-computed in ONE pass, valid). For the top tabular high-regret
datasets: is the oracle win REAL (a cluster of good detectors we missed) or SPURIOUS (a lone lucky
detector on few anomalies, no detector really better)? Report per-detector ap_norm, cluster size,
best-2nd gap, n_hard, and whether we/MV picked the wrong family."""
import os, sys, io, contextlib, warnings
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score
warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"; S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
D = os.path.join(S, "scratchpad", "streamline"); sys.path.insert(0, os.path.join(S, "scratchpad")); sys.path.insert(0, os.path.join(ROOT, "src"))
import adrank.pipeline as PP
from dev_common import TAB_POOL
from hadb_round2_common import _mv_auc
LOCAL = ("LOF", "KNN", "CBLOF"); GLOBAL = ("HBOS", "COPOD", "ECOD", "PCA")
fam = lambda v: "local" if str(v).startswith(LOCAL) else ("global" if str(v).startswith(GLOBAL) else "other")
Q = 0.05; NGEN = 800
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
rk = pd.read_csv(os.path.join(D, "STREAM_RANK.csv"))
tab = rk[rk.corpus.isin(["oddbench", "ovrbench", "adbench", "dami"])].sort_values("reg_matched", ascending=False).head(8)
t = np.linspace(0, 100, NGEN); alpha = np.linspace(0.9, 0.999, NGEN)
print("=== high-regret tail, co-computed (tabular). Is the oracle real (cluster) or spurious (lone spike)? ===\n")
for _, r in tab.iterrows():
    try: Xtr, Xval, Xtn, Xa = get3(r.corpus, r.dataset)
    except Exception: continue
    Xh = harden(Xtr, np.vstack([Xval, Xtn]), Xa)
    if len(Xh) < 15: continue
    Xte = np.vstack([Xtn, Xh]); yte = np.r_[np.zeros(len(Xtn)), np.ones(len(Xh))]; base = yte.mean()
    rng = np.random.default_rng(0); lo, hi = Xval.min(0), Xval.max(0); U = lo + rng.random((NGEN, Xval.shape[1])) * np.where(hi > lo, hi - lo, 1.0)
    syn1, syn4 = gen_beta(Xval, 200, 1.0), gen_beta(Xval, 200, -4.0); ye1 = np.r_[np.zeros(len(Xval)), np.ones(200)]
    ap, a1, a4, mvv = {}, {}, {}, {}
    for vn, ct in TAB_POOL:
        try:
            m = ct()
            with contextlib.redirect_stdout(io.StringIO()): m.fit(Xtr)
            st = np.asarray(m.decision_function(Xte), float); sv = np.asarray(m.decision_function(Xval), float); su = np.asarray(m.decision_function(U), float)
            if not (np.all(np.isfinite(st)) and np.nanstd(st) > 1e-12): continue
            ap[vn] = (average_precision_score(yte, st) - base) / (1 - base + 1e-12)
            from sklearn.metrics import roc_auc_score
            a1[vn] = roc_auc_score(ye1, np.r_[sv, m.decision_function(syn1)]); a4[vn] = roc_auc_score(np.r_[np.zeros(len(Xval)), np.ones(200)], np.r_[sv, m.decision_function(syn4)])
            su = np.nan_to_num(su, nan=float(np.nanmedian(su[np.isfinite(su)])) if np.isfinite(su).any() else 0.0)
            mvv[vn] = _mv_auc(alpha, 1.0, -su, -sv, NGEN)
        except Exception: pass
    if len(ap) < 6: continue
    av = pd.Series(ap).sort_values(ascending=False); best = av.iloc[0]; second = av.iloc[1]
    p1 = max(a1, key=lambda v: a1[v]); pe = max(a4, key=lambda v: a4[v]); lev = max(a1.values()) - max(a4.values())
    matched = p1 if lev > -0.14 else pe; mvpick = min(mvv, key=lambda v: mvv[v])
    ncl = int((av >= best - 0.05).sum()); ncl10 = int((av >= best - 0.10).sum())
    orfam = fam(av.index[0])
    kind = "ALL-BAD(oracle<.12)" if best < 0.12 else ("LONE-SPIKE" if (best - second) > 0.2 else "CLUSTER")
    print(f"{r.dataset[:24]:26s} truefam={r.truefam:6s} n_hard={len(Xh):4d} base={base:.2f}")
    print(f"   oracle {av.index[0][:14]:14s}[{orfam[:3]}] ap={best:.2f} | 2nd {av.index[1][:12]:12s} ap={second:.2f} | best-2nd={best-second:.2f} | #within.05={ncl} #within.10={ncl10} -> {kind}")
    print(f"   WE(matched) picked {matched[:14]:14s}[{fam(matched)[:3]}] ap={ap.get(matched,float('nan')):.2f} reg={best-ap.get(matched,0):.2f}   MV picked {mvpick[:14]:14s}[{fam(mvpick)[:3]}] ap={ap.get(mvpick,float('nan')):.2f} reg={best-ap.get(mvpick,0):.2f}")
    print(f"   top5: {', '.join(f'{k[:12]}={v:.2f}' for k,v in av.head(5).items())}\n")
print("KEY: CLUSTER = several detectors near oracle (real hard selection); LONE-SPIKE = one lucky detector (maybe spurious); ALL-BAD = oracle itself weak")
