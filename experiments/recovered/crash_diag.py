# -*- coding: utf-8 -*-
"""Reproduce the shard crash: fit each of the 32 detectors ONE AT A TIME on the two datasets that killed
shards 10/11, logging (flushed) before/after each fit. A hard crash (segfault/OOM) leaves the last
'fitting X' line as the culprit. Classical only first (fast); deep can be added if classical survives."""
import sys, os, warnings, numpy as np, time
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
BUN = r"E:/tmp/claude/E--Projects-Submitted-ADRank/236d6247-42ad-4e62-9828-db625dfa055d/scratchpad/streamline/bundle_v2"
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
def deep_pool():
    M=[]
    for nm,mk in [("D_DeepSVDD", lambda: __import__("deepod.models",fromlist=["DeepSVDD"]).DeepSVDD(epochs=40,device="cpu",random_state=0)),
                  ("D_ICL", lambda: __import__("deepod.models",fromlist=["ICL"]).ICL(epochs=40,device="cpu",random_state=0)),
                  ("D_RCA", lambda: __import__("deepod.models",fromlist=["RCA"]).RCA(epochs=40,device="cpu",random_state=0)),
                  ("D_SLAD", lambda: __import__("deepod.models",fromlist=["SLAD"]).SLAD(epochs=40,device="cpu",random_state=0)),
                  ("D_DIF", lambda: __import__("deepod.models",fromlist=["DeepIsolationForest"]).DeepIsolationForest(epochs=1,device="cpu",random_state=0)),
                  ("D_AE", lambda: __import__("pyod.models.auto_encoder",fromlist=["AutoEncoder"]).AutoEncoder(epoch_num=50,verbose=0)),
                  ("D_VAE", lambda: __import__("pyod.models.vae",fromlist=["VAE"]).VAE(epoch_num=50,verbose=0))]:
        M.append((nm,mk))
    return M
key = sys.argv[1]; do_deep = len(sys.argv) > 2 and sys.argv[2] == "deep"
z = np.load(os.path.join(BUN, key + ".npz")); Xtr = z["Xtr"].astype("float64"); Xval = z["Xval"].astype("float64")
ss = StandardScaler().fit(Xtr); Ztr = ss.transform(Xtr); Zval = ss.transform(Xval)
print(f"=== {key}  Xtr{Xtr.shape} Xval{Xval.shape} ===", flush=True)
pool = classical_pool() + (deep_pool() if do_deep else [])
for nm, ctor in pool:
    print(f"  fitting {nm} ...", end=" ", flush=True); t=time.time()
    try:
        m = ctor(); m.fit(Ztr); sv = m.decision_function(Zval)
        print(f"OK ({time.time()-t:.1f}s, val_std={np.std(sv):.3g})", flush=True)
    except Exception as e:
        print(f"EXC {type(e).__name__}: {e}", flush=True)
print("=== survived all fits ===", flush=True)
