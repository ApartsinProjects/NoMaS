# -*- coding: utf-8 -*-
"""Tier-3 #3a probe-seed variance + #4 budget-sensitivity (classical, $0 local).
Refit the 25 classical detectors on each dataset; grade them with SPARC's dependency probe at several
RNG seeds and several budgets; select argmax; regret = oracle - published clstd ap[selected].
Reports SPARC-classical regret mean/SD across 10 seeds (and win-rate vs fixed) and across budgets."""
import json, glob, os, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
S = r"E:/tmp/claude/E--Projects-Submitted-ADRank/236d6247-42ad-4e62-9828-db625dfa055d/scratchpad"
ST = os.path.join(S, "streamline"); BUN = os.path.join(ST, "bundle_v2"); DEEP = lambda d: d.startswith("D_")
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
SEEDS = list(range(10)); BUDGETS = [200, 500, 1000, 2000]
seed_reg = {s: [] for s in SEEDS}; seed_win = {s: 0 for s in SEEDS}
budg_reg = {b: [] for b in BUDGETS}
for fi, f in enumerate(files):
    key = os.path.basename(f)[:-4]; z = np.load(f)
    Xtr = z["Xtr"].astype("float64"); Xval = z["Xval"].astype("float64")
    ss = StandardScaler().fit(Xtr); Ztr = ss.transform(Xtr); Zval = ss.transform(Xval)
    if len(Zval) < 20: continue
    ap = clsap[key]; o = oracle[key]
    sv = {}  # classical val scores (fit once)
    for nm, ctor in classical_pool():
        try: m = ctor(); m.fit(Ztr); sv[nm] = (m, m.decision_function(Zval))
        except Exception: pass
    dets = [d for d in sv if d in ap]
    if len(dets) < 5: continue
    fixed_r = o - ap.get("LOF_k10", ap[max(dets, key=lambda d: -ap[d])])
    # seed variance at budget 1000
    for s in SEEDS:
        P = gen_beta(Zval, 1000, seed=s); y = np.r_[np.zeros(len(Zval)), np.ones(len(P))]
        g = {d: roc_auc_score(y, np.r_[sv[d][1], sv[d][0].decision_function(P)]) for d in dets}
        r = o - ap[max(g, key=g.get)]; seed_reg[s].append(r)
        if r < fixed_r - 1e-9: seed_win[s] += 1
    # budget sensitivity at seed 0
    for b in BUDGETS:
        P = gen_beta(Zval, b, seed=0); y = np.r_[np.zeros(len(Zval)), np.ones(len(P))]
        g = {d: roc_auc_score(y, np.r_[sv[d][1], sv[d][0].decision_function(P)]) for d in dets}
        budg_reg[b].append(o - ap[max(g, key=g.get)])
    if (fi + 1) % 30 == 0: print(f"  {fi+1}/{len(files)}", flush=True)
means = np.array([np.mean(seed_reg[s]) for s in SEEDS]); wins = np.array([seed_win[s] for s in SEEDS])
print("\n=== #3a probe-seed variance (classical, 1000 pts, 10 seeds) ===")
print(f"  SPARC-classical regret across seeds: mean {means.mean():.4f}  SD {means.std():.4f}  range [{means.min():.4f}, {means.max():.4f}]")
print(f"  beats fixed detector on {wins.min()}-{wins.max()} of {len(seed_reg[0])} tasks across seeds (mean {wins.mean():.1f})")
print("\n=== #4 budget-sensitivity (classical, seed 0) ===")
for b in BUDGETS: print(f"  {b:5d} points: SPARC-classical regret {np.mean(budg_reg[b]):.4f}")
