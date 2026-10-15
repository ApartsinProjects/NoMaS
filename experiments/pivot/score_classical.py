# -*- coding: utf-8 -*-
"""Substrate for pivot smoke tests: refit the 25 classical detectors on every task bundle (standardized
features, all seeds fixed incl. LODA) and save per-point scores on the validation normals and the test set,
plus test labels. Resumable: skips tasks whose output exists. Output: experiments/pivot/scores/<key>.npz
with arrays val__<det>, te__<det>, y_te. Also writes ap.jsonl (ap_norm per detector) for invariant checks."""
import glob, os, sys, json, warnings
import numpy as np
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
BUN = os.path.join(HERE, "..", "..", "artifacts", "modal", "bv2")
OUT = os.path.join(HERE, "scores"); os.makedirs(OUT, exist_ok=True)


def classical_pool():
    from pyod.models.iforest import IForest; from pyod.models.lof import LOF; from pyod.models.knn import KNN
    from pyod.models.hbos import HBOS; from pyod.models.ecod import ECOD; from pyod.models.copod import COPOD
    from pyod.models.pca import PCA as PPCA; from pyod.models.cblof import CBLOF; from pyod.models.loda import LODA
    def loda(**kw):
        np.random.seed(0); return LODA(**kw)          # LODA draws projections from the global RNG
    p = [("IForest", lambda: IForest(n_estimators=100, random_state=0)), ("LOF", lambda: LOF(n_neighbors=20)),
         ("KNN", lambda: KNN(n_neighbors=5)), ("ECOD", lambda: ECOD()), ("COPOD", lambda: COPOD()),
         ("HBOS", lambda: HBOS(n_bins=10)), ("PCA", lambda: PPCA(n_components=0.5, random_state=0)),
         ("CBLOF", lambda: CBLOF(n_clusters=8, random_state=0)), ("LODA", lambda: loda(n_bins=10))]
    for k in [3, 10, 35, 100, 200]:
        p.append((f"LOF_k{k}", lambda k=k: LOF(n_neighbors=k))); p.append((f"KNN_k{k}", lambda k=k: KNN(n_neighbors=k)))
    for nb in [5, 20, 50]: p.append((f"HBOS_b{nb}", lambda nb=nb: HBOS(n_bins=nb)))
    for nc in [0.3, 0.7, 0.9]: p.append((f"PCA_c{nc}", lambda nc=nc: PPCA(n_components=nc, random_state=0)))
    return p


def work(f):
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import average_precision_score
    key = os.path.basename(f)[:-4]; dst = os.path.join(OUT, key + ".npz")
    if os.path.exists(dst): return (key, None)
    z = np.load(f)
    Xtr, Xval, Xtn, Xh = (z[k].astype("float64") for k in ("Xtr", "Xval", "Xtn", "Xh"))
    ss = StandardScaler().fit(Xtr); Ztr, Zval = ss.transform(Xtr), ss.transform(Xval)
    Zte = ss.transform(np.vstack([Xtn, Xh])); y = np.r_[np.zeros(len(Xtn)), np.ones(len(Xh))]; r = y.mean()
    out, ap = {"y_te": y.astype(np.int8)}, {}
    for nm, ctor in classical_pool():
        try:
            m = ctor(); m.fit(Ztr)
            st = np.asarray(m.decision_function(Zte), np.float32); sv = np.asarray(m.decision_function(Zval), np.float32)
            if not (np.all(np.isfinite(st)) and np.std(st) > 0): continue
            out[f"te__{nm}"] = st; out[f"val__{nm}"] = sv
            ap[nm] = float((average_precision_score(y, st) - r) / (1 - r))
        except Exception:
            pass
    np.savez_compressed(dst, **out)
    return (key, ap)


if __name__ == "__main__":
    from multiprocessing import Pool
    files = sorted(glob.glob(os.path.join(BUN, "*.npz")))
    print(f"{len(files)} tasks", flush=True)
    with Pool(int(os.environ.get("NW", "3"))) as pool, open(os.path.join(HERE, "ap_classical.jsonl"), "a") as fo:
        for i, (key, ap) in enumerate(pool.imap_unordered(work, files, chunksize=1)):
            if ap is not None: fo.write(json.dumps({"key": key, "ap": ap}) + "\n"); fo.flush()
            if (i + 1) % 20 == 0: print(f"  {i+1}/{len(files)}", flush=True)
    print("done", flush=True)
