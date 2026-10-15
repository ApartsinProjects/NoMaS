# -*- coding: utf-8 -*-
"""Classical re-score with the FIXED dependency/marginal probe generator (_gen_beta). The original drew
the replacement value uniformly WITHIN a histogram bin, which manufactures values that never occur on
discrete/binary columns, letting marginal detectors (ECOD/COPOD/HBOS) spuriously 'detect' the dependency
probe. The fix draws an actually-OBSERVED value from the chosen bin. ap (real test) is unchanged by this
(probe only affects a1/a4); we re-fit and recompute a1/a4 + ap for a full consistent table.
Output -> deepres_v2_clfix. Usage: python rescore_classical_fix.py <bundle_dir> <out_dir>"""
import os, sys, json, glob
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score
BUN = sys.argv[1] if len(sys.argv) > 1 else "bundle_v2"
OUT = sys.argv[2] if len(sys.argv) > 2 else "deepres_v2_clfix"
os.makedirs(OUT, exist_ok=True)


def _classical_pool():
    from pyod.models.iforest import IForest; from pyod.models.lof import LOF; from pyod.models.knn import KNN
    from pyod.models.hbos import HBOS; from pyod.models.ecod import ECOD; from pyod.models.copod import COPOD
    from pyod.models.pca import PCA as PPCA; from pyod.models.cblof import CBLOF; from pyod.models.loda import LODA
    p = [("IForest", lambda: IForest(n_estimators=100, random_state=0)), ("LOF", lambda: LOF(n_neighbors=20)),
         ("KNN", lambda: KNN(n_neighbors=5)), ("ECOD", lambda: ECOD()), ("COPOD", lambda: COPOD()),
         ("HBOS", lambda: HBOS(n_bins=10)), ("PCA", lambda: PPCA(n_components=0.5, random_state=0)),
         ("CBLOF", lambda: CBLOF(n_clusters=8, random_state=0)), ("LODA", lambda: LODA(n_bins=10, random_state=0))]
    for k in [3, 10, 35, 100, 200]:
        p.append((f"LOF_k{k}", lambda k=k: LOF(n_neighbors=k))); p.append((f"KNN_k{k}", lambda k=k: KNN(n_neighbors=k)))
    for nb in [5, 20, 50]:
        p.append((f"HBOS_b{nb}", lambda nb=nb: HBOS(n_bins=nb)))
    for nc in [0.3, 0.7, 0.9]:
        p.append((f"PCA_c{nc}", lambda nc=nc: PPCA(n_components=nc, random_state=0)))
    return p


def _gen_beta(Xn, ns, beta, frac=0.4, seed=0):
    """FIXED: replacement value is an actually-observed value drawn from the chosen histogram bin,
    so discrete/binary columns never receive an off-atom value."""
    rng = np.random.default_rng(seed); n, d = Xn.shape; out = np.empty((ns, d)); Hs = []
    for j in range(d):
        col = Xn[:, j]; cnt, edges = np.histogram(col, bins=30); dens = cnt / max(cnt.sum(), 1)
        lo, hi = np.percentile(col, 2), np.percentile(col, 98); ctr = (edges[:-1] + edges[1:]) / 2
        allowed = (cnt > 0) & (ctr >= lo) & (ctr <= hi); w = np.where(allowed, np.maximum(dens, 1e-6) ** beta, 0.0)
        wn = w / w.sum() if w.sum() > 0 else None
        binidx = np.clip(np.digitize(col, edges[1:-1]), 0, len(cnt) - 1)  # bin of each observed value
        Hs.append((wn, col, binidx))
    for r in range(ns):
        base = Xn[rng.integers(n)].copy()
        for j in rng.choice(d, max(1, int(frac * d)), replace=False):
            wn, col, binidx = Hs[j]
            if wn is None:
                base[j] = Xn[rng.integers(n), j]
            else:
                b = rng.choice(len(wn), p=wn); cand = col[binidx == b]
                base[j] = cand[rng.integers(len(cand))] if len(cand) else Xn[rng.integers(n), j]
        out[r] = base
    return out


def apn(y, s):
    bb = y.mean(); return float((average_precision_score(y, s) - bb) / (1 - bb + 1e-12))


outp = os.path.join(OUT, "shard_local.jsonl"); done = set()
if os.path.exists(outp):
    for ln in open(outp):
        try:
            r = json.loads(ln)
            if "ap" in r: done.add(r["key"])
        except Exception: pass
fo = open(outp, "a")
files = sorted(glob.glob(os.path.join(BUN, "*.npz")))
for i, f in enumerate(files):
    key = os.path.basename(f)[:-4]
    if key in done: continue
    z = np.load(f); Xtr, Xval, Xtn, Xh = [z[k].astype("float64") for k in ("Xtr", "Xval", "Xtn", "Xh")]
    _ss = StandardScaler().fit(Xtr); Xtr, Xval, Xtn, Xh = _ss.transform(Xtr), _ss.transform(Xval), _ss.transform(Xtn), _ss.transform(Xh)
    Xte = np.vstack([Xtn, Xh]); yte = np.r_[np.zeros(len(Xtn)), np.ones(len(Xh))]
    syn1, syn4 = _gen_beta(Xval, 200, 1.0), _gen_beta(Xval, 200, -4.0)
    y1 = np.r_[np.zeros(len(Xval)), np.ones(200)]
    res = {"key": key, "ap": {}, "a1": {}, "a4": {}}
    for nm, ctor in _classical_pool():
        try:
            m = ctor(); m.fit(Xtr)
            sv = np.asarray(m.decision_function(Xval), float); s1 = np.asarray(m.decision_function(syn1), float)
            s4 = np.asarray(m.decision_function(syn4), float); st = np.asarray(m.decision_function(Xte), float)
            if np.all(np.isfinite(st)) and np.nanstd(st) > 1e-12: res["ap"][nm] = apn(yte, st)
            if np.nanstd(np.r_[sv, s1]) > 1e-12: res["a1"][nm] = float(roc_auc_score(y1, np.r_[sv, s1]))
            if np.nanstd(np.r_[sv, s4]) > 1e-12: res["a4"][nm] = float(roc_auc_score(y1, np.r_[sv, s4]))
        except Exception: pass
    fo.write(json.dumps(res) + "\n"); fo.flush()
    if i % 20 == 0: print(f"  ..{i}/{len(files)} {key}: {len(res['ap'])} det", flush=True)
fo.close(); print(f"done -> {OUT}")
