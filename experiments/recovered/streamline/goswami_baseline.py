# -*- coding: utf-8 -*-
"""Goswami et al. (ICLR 2023) injection-based unsupervised model selection, as a baseline for ADReal.
Faithful adaptation to our windowed-feature / tabular representation: for each task we re-fit the SAME
classical detector pool as rescore_classical_std.py on standardized train normals, generate synthetic
anomalies of several canonical injection TYPES from the validation normals (Goswami's synthetic-anomaly
surrogate), score each detector by how well it separates injected from normal (ROC-AUC per type), and
aggregate detector ranks across injection types by Borda count. The selected detector is the Borda winner.
Candidates are the classical detectors (deep models are not re-fittable here and SPARC also restricts
probe candidates to classical, so this is apples-to-apples). Writes gos_<key> rows incrementally.
Usage: python goswami_baseline.py <bundle_dir> <out_jsonl>"""
import os, sys, json, glob
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

BUN = sys.argv[1] if len(sys.argv) > 1 else "bundle_v2"
OUT = sys.argv[2] if len(sys.argv) > 2 else "GOSWAMI_INJ.jsonl"


def _classical_pool():
    from pyod.models.iforest import IForest; from pyod.models.lof import LOF; from pyod.models.knn import KNN
    from pyod.models.hbos import HBOS; from pyod.models.ecod import ECOD; from pyod.models.copod import COPOD
    from pyod.models.pca import PCA as PPCA; from pyod.models.cblof import CBLOF; from pyod.models.loda import LODA
    p = [("IForest", lambda: IForest(n_estimators=100, random_state=0)), ("LOF", lambda: LOF(n_neighbors=20, n_jobs=1)),
         ("KNN", lambda: KNN(n_neighbors=5, n_jobs=1)), ("ECOD", lambda: ECOD()), ("COPOD", lambda: COPOD()),
         ("HBOS", lambda: HBOS(n_bins=10)), ("PCA", lambda: PPCA(n_components=0.5, random_state=0)),
         ("CBLOF", lambda: CBLOF(n_clusters=8, random_state=0)), ("LODA", lambda: LODA(n_bins=10, random_state=0))]
    for k in [3, 10, 35, 100, 200]:
        p.append((f"LOF_k{k}", lambda k=k: LOF(n_neighbors=k, n_jobs=1))); p.append((f"KNN_k{k}", lambda k=k: KNN(n_neighbors=k, n_jobs=1)))
    for nb in [5, 20, 50]:
        p.append((f"HBOS_b{nb}", lambda nb=nb: HBOS(n_bins=nb)))
    for nc in [0.3, 0.7, 0.9]:
        p.append((f"PCA_c{nc}", lambda nc=nc: PPCA(n_components=nc, random_state=0)))
    return p


def inject(Xn, ns=200, seed=0):
    """Goswami-style synthetic anomalies of several canonical TYPES, built from normal rows Xn
    (standardized). Returns dict type -> array (ns, d). Types: spike (extreme coord), noise (additive
    Gaussian), scale (multiplicative blow-up of a subset), contextual (swap a subset with a distant
    row's values), cutoff (flatten a subset to zero/mean)."""
    rng = np.random.default_rng(seed); n, d = Xn.shape; out = {}
    k = max(1, int(0.2 * d))

    def base_idx(m): return rng.integers(n, size=m)
    # spike: push k coords to a large magnitude (standardized units)
    A = Xn[base_idx(ns)].copy()
    for r in range(ns):
        js = rng.choice(d, k, replace=False); A[r, js] = rng.choice([-1, 1], k) * rng.uniform(4, 8, k)
    out["spike"] = A
    # noise: additive Gaussian on all coords
    out["noise"] = Xn[base_idx(ns)] + rng.normal(0, 1.5, (ns, d))
    # scale: multiply k coords by a large factor
    A = Xn[base_idx(ns)].copy()
    for r in range(ns):
        js = rng.choice(d, k, replace=False); A[r, js] = A[r, js] * rng.uniform(3, 6, k)
    out["scale"] = A
    # contextual: replace k coords with those of a randomly chosen distant row
    A = Xn[base_idx(ns)].copy(); D = Xn[base_idx(ns)]
    for r in range(ns):
        js = rng.choice(d, k, replace=False); A[r, js] = D[r, js]
    out["contextual"] = A
    # cutoff: flatten k coords to zero (the standardized mean)
    A = Xn[base_idx(ns)].copy()
    for r in range(ns):
        js = rng.choice(d, k, replace=False); A[r, js] = 0.0
    out["cutoff"] = A
    return out


def main():
    done = set()
    if os.path.exists(OUT):
        for ln in open(OUT):
            try:
                r = json.loads(ln)
                if "sel" in r:
                    done.add(r["key"])
            except Exception:
                pass
    fo = open(OUT, "a")
    files = sorted(glob.glob(os.path.join(BUN, "*.npz")))
    pool = _classical_pool()
    for i, f in enumerate(files):
        key = os.path.basename(f)[:-4]
        if key in done:
            continue
        z = np.load(f); Xtr, Xval = z["Xtr"].astype("float64"), z["Xval"].astype("float64")
        # cap the fit sample so high-dimensional / long-series tasks re-fit quickly; the injection
        # ranking is stable to this subsample and the selected detector's ap comes from the full-data
        # cached score, so this only speeds the local re-fit.
        if len(Xtr) > 4000:
            Xtr = Xtr[np.random.default_rng(0).choice(len(Xtr), 4000, replace=False)]
        ss = StandardScaler().fit(Xtr); Xtr, Xval = ss.transform(Xtr), ss.transform(Xval)
        inj = inject(Xval, ns=200, seed=0)
        types = list(inj.keys())
        # per-type detection AUC for each detector
        aucs = {nm: {} for nm, _ in pool}
        names = []
        for nm, ctor in pool:
            try:
                m = ctor(); m.fit(Xtr)
                sv = np.asarray(m.decision_function(Xval), float)
                ok = True
                per = {}
                for t in types:
                    sa = np.asarray(m.decision_function(inj[t]), float)
                    s = np.r_[sv, sa]; y = np.r_[np.zeros(len(sv)), np.ones(len(sa))]
                    if np.nanstd(s) < 1e-12 or not np.all(np.isfinite(s)):
                        ok = False; break
                    per[t] = float(roc_auc_score(y, s))
                if ok:
                    aucs[nm] = per; names.append(nm)
            except Exception:
                pass
        if len(names) < 6:
            fo.write(json.dumps({"key": key, "sel": None, "sel_c": None}) + "\n"); fo.flush(); continue
        # Borda aggregation of per-type ranks (higher AUC = better) across injection types
        borda = {nm: 0.0 for nm in names}
        for t in types:
            order = sorted(names, key=lambda nm: aucs[nm].get(t, 0.0))  # ascending
            for rank, nm in enumerate(order):
                borda[nm] += rank  # higher rank index = better
        # mean-AUC winner too (alternative aggregation), for robustness reporting
        meanauc = {nm: float(np.mean([aucs[nm][t] for t in types])) for nm in names}
        sel = max(names, key=lambda nm: borda[nm])
        sel_mean = max(names, key=lambda nm: meanauc[nm])
        fo.write(json.dumps({"key": key, "sel": sel, "sel_mean": sel_mean,
                             "borda": borda, "meanauc": meanauc}) + "\n"); fo.flush()
        if i % 10 == 0:
            print(f"  ..{i}/{len(files)} {key}: sel={sel}", flush=True)
    fo.close(); print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
