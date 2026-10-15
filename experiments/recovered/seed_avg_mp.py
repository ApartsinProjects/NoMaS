# -*- coding: utf-8 -*-
"""Parallel (multiprocessing) seed-average of ALL classical-pool SPARC numbers over 10 probe seeds, both the
observed-value draw (SPARC) and the bin-interpolation variant. Per task: refit classical detectors, grade each
probe seed, record SPARC's pick regret (oracle - clstd ap[pick]). Aggregates All + per-source, and saves
per-task per-seed regret to SEED_AVG_CLASSICAL.pkl for downstream paired tests / cluster bootstrap."""
import json, glob, os, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
S = r"E:/tmp/claude/E--Projects-Submitted-ADRank/236d6247-42ad-4e62-9828-db625dfa055d/scratchpad"
ST = os.path.join(S, "streamline"); BUN = os.path.join(ST, "bundle_v2")
SEEDS = list(range(10))


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


def work(arg):
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    key, f, ap, o = arg
    try:
        z = np.load(f); Xtr = z["Xtr"].astype("float64"); Xval = z["Xval"].astype("float64")
        ss = StandardScaler().fit(Xtr); Ztr = ss.transform(Xtr); Zval = ss.transform(Xval)
        if len(Zval) < 20: return None
        fitted = {}
        for nm, ctor in classical_pool():
            try:
                m = ctor(); m.fit(Ztr); fitted[nm] = (m, m.decision_function(Zval))
            except Exception: pass
        dets = [d for d in fitted if d in ap]
        if len(dets) < 5: return None
        obs, binr = [], []
        for s in SEEDS:
            Po = gen_beta(Zval, 1000, seed=s); Pb = gen_bininterp(Zval, 1000, seed=s)
            y = np.r_[np.zeros(len(Zval)), np.ones(1000)]
            go = {d: roc_auc_score(y, np.r_[fitted[d][1], fitted[d][0].decision_function(Po)]) for d in dets}
            gb = {d: roc_auc_score(y, np.r_[fitted[d][1], fitted[d][0].decision_function(Pb)]) for d in dets}
            obs.append(o - ap[max(go, key=go.get)]); binr.append(o - ap[max(gb, key=gb.get)])
        return (key, obs, binr)
    except Exception as e:
        return ("ERR:" + key, str(e), None)


def src_of(c):
    c = str(c).lower()
    if "ovr" in c: return "OvrBench"
    if "odd" in c: return "OddBench"
    if "tsb" in c or c == "ts" or "series" in c: return "TS"
    return "OTHER"


def main():
    from multiprocessing import Pool
    clsap = {json.loads(l)["key"]: json.loads(l)["ap"] for l in open(os.path.join(S, "clsstd_valuni", "ap.jsonl"))}
    lb = pd.read_csv(os.path.join(ST, "STREAM_V3_LEADERBOARD.csv"))
    keep = lb[(lb.oracle_ap > 0.05) & (lb.reg_random > 0.02) & (lb.n_h >= 100)].set_index("key")
    oracle = keep.oracle_ap.to_dict(); corpus = keep.corpus.to_dict()
    files = [f for f in sorted(glob.glob(os.path.join(BUN, "*.npz")))
             if os.path.basename(f)[:-4] in keep.index and os.path.basename(f)[:-4] in clsap]
    args = [(os.path.basename(f)[:-4], f, clsap[os.path.basename(f)[:-4]], oracle[os.path.basename(f)[:-4]]) for f in files]
    print(f"datasets: {len(args)}  workers: 6", flush=True)
    reg = {"observed": {}, "bininterp": {}}; src = {}; done = 0
    with Pool(6) as pool:
        for r in pool.imap_unordered(work, args, chunksize=1):
            done += 1
            if r is None: continue
            if r[0].startswith("ERR:"): print("  " + r[0], r[1], flush=True); continue
            key, obs, binr = r
            reg["observed"][key] = obs; reg["bininterp"][key] = binr; src[key] = src_of(corpus.get(key, ""))
            if done % 20 == 0: print(f"  {done}/{len(args)}", flush=True)
    pd.to_pickle({"reg": reg, "src": src}, os.path.join(S, "SEED_AVG_CLASSICAL.pkl"))
    keys = list(reg["observed"].keys())
    tm = {k: np.mean(reg["observed"][k]) for k in keys}
    print(f"\nobserved All(n={len(keys)}) seed-mean regret = {np.mean([tm[k] for k in keys]):.4f}")
    print(f"bininterp All seed-mean regret = {np.mean([np.mean(reg['bininterp'][k]) for k in keys]):.4f}")
    print("saved SEED_AVG_CLASSICAL.pkl")


if __name__ == "__main__":
    main()
