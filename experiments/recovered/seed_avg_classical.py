# -*- coding: utf-8 -*-
"""Seed-average ALL classical-pool SPARC numbers over 10 probe seeds, so the paper can headline the
ten-seed mean (0.114) instead of the best seed (0.109). Produces, per variant, the All-tasks mean and
the per-source breakdown (OvrBench / OddBench / TSB-AD/TS), matching Table 6 (classical col), Table 6b,
Table 7 (rows 1-2), and Figures 6-7. Variants: observed-value draw (SPARC) and bin-interpolation.
Classical detectors are refit locally; regret = STREAM_V3 oracle - clstd ap[selected]."""
import json, glob, os, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
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
    """observed-value draw (SPARC): replacement is an actual value the feature takes in the chosen bin."""
    rng = np.random.default_rng(seed); n, d = Xn.shape; out = np.empty((ns, d)); Hs = []
    for j in range(d):
        col = Xn[:, j]; cnt, edges = np.histogram(col, bins=30); dens = cnt / max(cnt.sum(), 1)
        lo, hi = np.percentile(col, 2), np.percentile(col, 98); ctr = (edges[:-1] + edges[1:]) / 2
        allowed = (cnt > 0) & (ctr >= lo) & (ctr <= hi); w = np.where(allowed, np.maximum(dens, 1e-6), 0.0)
        wn = w / w.sum() if w.sum() > 0 else None
        binidx = np.clip(np.digitize(col, edges[1:-1]), 0, len(cnt) - 1); Hs.append((wn, col, binidx, edges))
    for r in range(ns):
        base = Xn[rng.integers(n)].copy()
        for j in rng.choice(d, max(1, int(frac * d)), replace=False):
            wn, col, binidx, edges = Hs[j]
            if wn is None: base[j] = Xn[rng.integers(n), j]
            else:
                b = rng.choice(len(wn), p=wn); cand = col[binidx == b]
                base[j] = cand[rng.integers(len(cand))] if len(cand) else Xn[rng.integers(n), j]
        out[r] = base
    return out

def gen_bininterp(Xn, ns, frac=0.4, seed=0):
    """bin-interpolation: replacement is a uniform continuous value within the chosen bin."""
    rng = np.random.default_rng(seed); n, d = Xn.shape; out = np.empty((ns, d)); Hs = []
    for j in range(d):
        col = Xn[:, j]; cnt, edges = np.histogram(col, bins=30); dens = cnt / max(cnt.sum(), 1)
        lo, hi = np.percentile(col, 2), np.percentile(col, 98); ctr = (edges[:-1] + edges[1:]) / 2
        allowed = (cnt > 0) & (ctr >= lo) & (ctr <= hi); w = np.where(allowed, np.maximum(dens, 1e-6), 0.0)
        wn = w / w.sum() if w.sum() > 0 else None
        Hs.append((wn, edges))
    for r in range(ns):
        base = Xn[rng.integers(n)].copy()
        for j in rng.choice(d, max(1, int(frac * d)), replace=False):
            wn, edges = Hs[j]
            if wn is None: base[j] = Xn[rng.integers(n), j]
            else:
                b = rng.choice(len(wn), p=wn); base[j] = rng.uniform(edges[b], edges[b + 1])
        out[r] = base
    return out

clsap = {json.loads(l)["key"]: json.loads(l)["ap"] for l in open(os.path.join(S, "clsstd_valuni", "ap.jsonl"))}
lb = pd.read_csv(os.path.join(ST, "STREAM_V3_LEADERBOARD.csv"))
keep = lb[(lb.oracle_ap > 0.05) & (lb.reg_random > 0.02) & (lb.n_h >= 100)].set_index("key")
oracle = keep.oracle_ap.to_dict(); corpus = keep.corpus.to_dict()
files = [f for f in sorted(glob.glob(os.path.join(BUN, "*.npz")))
         if os.path.basename(f)[:-4] in keep.index and os.path.basename(f)[:-4] in clsap]
print(f"datasets: {len(files)}", flush=True)

def src_of(c):
    c = str(c).lower()
    if "ovr" in c: return "OvrBench"
    if "odd" in c: return "OddBench"
    if "tsb" in c or c == "ts" or "series" in c: return "TS"
    return "OTHER"  # ADBench/DAMI -> All only

SEEDS = list(range(10)); GENS = {"observed": gen_beta, "bininterp": gen_bininterp}
# regret[variant][key] = list of 10 per-seed regrets
reg = {v: {} for v in GENS}; src = {}
for fi, f in enumerate(files):
    key = os.path.basename(f)[:-4]; z = np.load(f)
    Xtr = z["Xtr"].astype("float64"); Xval = z["Xval"].astype("float64")
    ss = StandardScaler().fit(Xtr); Ztr = ss.transform(Xtr); Zval = ss.transform(Xval)
    if len(Zval) < 20: continue
    ap = clsap[key]; o = oracle[key]
    fitted = {}
    for nm, ctor in classical_pool():
        try:
            m = ctor(); m.fit(Ztr); fitted[nm] = (m, m.decision_function(Zval))
        except Exception: pass
    dets = [d for d in fitted if d in ap]
    if len(dets) < 5: continue
    src[key] = src_of(corpus.get(key, ""))
    for vname, gen in GENS.items():
        rs = []
        for s in SEEDS:
            P = gen(Zval, 1000, seed=s); y = np.r_[np.zeros(len(Zval)), np.ones(len(P))]
            g = {d: roc_auc_score(y, np.r_[fitted[d][1], fitted[d][0].decision_function(P)]) for d in dets}
            rs.append(o - ap[max(g, key=g.get)])
        reg[vname][key] = rs
    if (fi + 1) % 30 == 0: print(f"  {fi+1}/{len(files)}", flush=True)

def report(vname):
    d = reg[vname]; keys = list(d.keys())
    # per-task seed-mean, then average over tasks (matches "mean regret" reporting)
    taskmean = {k: np.mean(d[k]) for k in keys}
    def agg(sub): return np.mean([taskmean[k] for k in sub]) if sub else float("nan")
    allk = keys
    ovr = [k for k in keys if src[k] == "OvrBench"]; odd = [k for k in keys if src[k] == "OddBench"]
    ts = [k for k in keys if src[k] == "TS"]
    # also the seed spread of the All mean, over the 10 seeds
    per_seed_all = [np.mean([d[k][s] for k in keys]) for s in range(10)]
    print(f"\n[{vname}]  All(n={len(allk)}) mean regret = {agg(allk):.4f}   "
          f"(per-seed All range {min(per_seed_all):.4f}-{max(per_seed_all):.4f}, sd {np.std(per_seed_all):.4f})")
    print(f"    OvrBench(n={len(ovr)}) {agg(ovr):.4f}   OddBench(n={len(odd)}) {agg(odd):.4f}   TS(n={len(ts)}) {agg(ts):.4f}")

for v in GENS: report(v)
pd.to_pickle({"reg": reg, "src": src}, os.path.join(S, "SEED_AVG_CLASSICAL.pkl"))
print("\nsaved SEED_AVG_CLASSICAL.pkl")
