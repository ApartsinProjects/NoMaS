# -*- coding: utf-8 -*-
"""Does the SPARC-vs-fixed headline hold at the SEED-AVERAGED regret (0.114) rather than seed 0 (0.109)?
Per task, average SPARC's classical regret over 10 probe seeds, then paired Wilcoxon vs the fixed detector.
Refit classical locally; grade the dependency probe at 10 seeds; select; regret = oracle - clstd ap[selected]."""
import json, glob, os, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import wilcoxon
S = r"E:/tmp/claude/E--Projects-Submitted-ADRank/236d6247-42ad-4e62-9828-db625dfa055d/scratchpad"
ST = os.path.join(S, "streamline"); BUN = os.path.join(ST, "bundle_v2")
def classical_pool():
    from pyod.models.iforest import IForest; from pyod.models.lof import LOF; from pyod.models.knn import KNN
    from pyod.models.hbos import HBOS; from pyod.models.ecod import ECOD; from pyod.models.copod import COPOD
    from pyod.models.pca import PCA as PPCA; from pyod.models.cblof import CBLOF; from pyod.models.loda import LODA
    p = [("IForest", lambda: IForest(n_estimators=100, random_state=0)), ("LOF", lambda: LOF(n_neighbors=20)),
         ("KNN", lambda: KNN(n_neighbors=5)), ("ECOD", lambda: ECOD()), ("COPOD", lambda: COPOD()),
         ("HBOS", lambda: HBOS(n_bins=10)), ("PCA", lambda: PPCA(n_components=0.5, random_state=0)),
         ("CBLOF", lambda: CBLOF(n_clusters=8, random_state=0)), ("LODA", lambda: LODA(n_bins=10))]
    for k in [3, 10, 35, 100, 200]:
        p.append((f"LOF_k{k}", lambda k=k: LOF(n_neighbors=k))); p.append((f"KNN_k{k}", lambda k=k: KNN(n_neighbors=k)))
    for nb in [5, 20, 50]: p.append((f"HBOS_b{nb}", lambda nb=nb: HBOS(n_bins=nb)))
    for nc in [0.3, 0.7, 0.9]: p.append((f"PCA_c{nc}", lambda nc=nc: PPCA(n_components=nc, random_state=0)))
    return p
def gen_beta(Xn, ns, frac=0.4, seed=0):
    rng = np.random.default_rng(seed); n, d = Xn.shape; out = np.empty((ns, d)); Hs = []
    for j in range(d):
        col = Xn[:, j]; cnt, edges = np.histogram(col, bins=30); dens = cnt / max(cnt.sum(), 1)
        lo, hi = np.percentile(col, 2), np.percentile(col, 98); ctr = (edges[:-1] + edges[1:]) / 2
        allowed = (cnt > 0) & (ctr >= lo) & (ctr <= hi); w = np.where(allowed, np.maximum(dens, 1e-6), 0.0)
        wn = w / w.sum() if w.sum() > 0 else None
        binidx = np.clip(np.digitize(col, edges[1:-1]), 0, len(cnt) - 1); Hs.append((wn, col, binidx))
    for r in range(ns):
        base = Xn[rng.integers(n)].copy()
        for j in rng.choice(d, max(1, int(frac * d)), replace=False):
            wn, col, binidx = Hs[j]
            if wn is None: base[j] = Xn[rng.integers(n), j]
            else:
                b = rng.choice(len(wn), p=wn); cand = col[binidx == b]
                base[j] = cand[rng.integers(len(cand))] if len(cand) else Xn[rng.integers(n), j]
        out[r] = base
    return out
clsap = {json.loads(l)["key"]: json.loads(l)["ap"] for l in open(os.path.join(S, "clsstd_valuni", "ap.jsonl"))}
lb = pd.read_csv(os.path.join(ST, "STREAM_V3_LEADERBOARD.csv"))
keep = lb[(lb.oracle_ap > 0.05) & (lb.reg_random > 0.02) & (lb.n_h >= 100)].set_index("key")
oracle = keep.oracle_ap.to_dict()
files = [f for f in sorted(glob.glob(os.path.join(BUN, "*.npz"))) if os.path.basename(f)[:-4] in keep.index and os.path.basename(f)[:-4] in clsap]
print(f"datasets: {len(files)}", flush=True)
SEEDS = list(range(10)); sparc_mean = []; fixed = []
for fi, f in enumerate(files):
    key = os.path.basename(f)[:-4]; z = np.load(f)
    Xtr = z["Xtr"].astype("float64"); Xval = z["Xval"].astype("float64")
    ss = StandardScaler().fit(Xtr); Ztr = ss.transform(Xtr); Zval = ss.transform(Xval)
    if len(Zval) < 20: continue
    ap = clsap[key]; o = oracle[key]
    sv = {}
    for nm, ctor in classical_pool():
        try: m = ctor(); m.fit(Ztr); sv[nm] = (m, m.decision_function(Zval))
        except Exception: pass
    dets = [d for d in sv if d in ap]
    if len(dets) < 5 or "LOF_k10" not in ap: continue
    rs = []
    for s in SEEDS:
        P = gen_beta(Zval, 1000, seed=s); y = np.r_[np.zeros(len(Zval)), np.ones(len(P))]
        g = {d: roc_auc_score(y, np.r_[sv[d][1], sv[d][0].decision_function(P)]) for d in dets}
        rs.append(o - ap[max(g, key=g.get)])
    sparc_mean.append(np.mean(rs)); fixed.append(o - ap["LOF_k10"])
    if (fi + 1) % 30 == 0: print(f"  {fi+1}/{len(files)}", flush=True)
sm = np.array(sparc_mean); fx = np.array(fixed)
d = sm - fx
print("\n=== seed-averaged SPARC vs fixed (paired over %d tasks) ===" % len(sm))
print(f"  SPARC seed-mean regret {sm.mean():.4f}  vs fixed {fx.mean():.4f}")
print(f"  paired advantage {(-d.mean()):+.4f}  p={wilcoxon(sm, fx).pvalue:.4f}")
print(f"  win/tie/loss {int((d<-1e-9).sum())}/{int((abs(d)<=1e-9).sum())}/{int((d>1e-9).sum())}")
