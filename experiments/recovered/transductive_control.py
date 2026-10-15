# -*- coding: utf-8 -*-
"""Tier-3 #7 transductive control ($0 local). The agreement selectors (consensus, model-centrality, HITS)
are run two ways on the classical pool: NORMALS-ONLY (criterion on the validation normals, as deployed)
and TRANSDUCTIVE (criterion on the test split, as they are usually reported). If they collapse only under
normals-only, transductive regret should be much lower. Refit classical locally; score val and test."""
import json, glob, os, sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, r"E:/Projects/Submitted/zenodo_build/code/lib")
from label_free_criteria import consensus_scores, model_centrality_scores, hits_authority_scores
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
clsap = {json.loads(l)["key"]: json.loads(l)["ap"] for l in open(os.path.join(S, "clsstd_valuni", "ap.jsonl"))}
lb = pd.read_csv(os.path.join(ST, "STREAM_V3_LEADERBOARD.csv"))
keep = lb[(lb.oracle_ap > 0.05) & (lb.reg_random > 0.02) & (lb.n_h >= 100)].set_index("key")
oracle = keep.oracle_ap.to_dict()
files = [f for f in sorted(glob.glob(os.path.join(BUN, "*.npz"))) if os.path.basename(f)[:-4] in keep.index and os.path.basename(f)[:-4] in clsap]
print(f"datasets: {len(files)}", flush=True)
CRIT = [("consensus", consensus_scores), ("model-centrality", model_centrality_scores), ("HITS", hits_authority_scores)]
res = {c: {"normals": [], "transductive": []} for c, _ in CRIT}
for fi, f in enumerate(files):
    key = os.path.basename(f)[:-4]; z = np.load(f)
    Xtr = z["Xtr"].astype("float64"); Xval = z["Xval"].astype("float64"); Xtn = z["Xtn"].astype("float64"); Xh = z["Xh"].astype("float64")
    ss = StandardScaler().fit(Xtr); Ztr = ss.transform(Xtr); Zval = ss.transform(Xval); Zte = ss.transform(np.vstack([Xtn, Xh]))
    ap = clsap[key]; o = oracle[key]
    dets, val_cols, te_cols = [], [], []
    for nm, ctor in classical_pool():
        if nm not in ap: continue
        try:
            m = ctor(); m.fit(Ztr); dets.append(nm); val_cols.append(m.decision_function(Zval)); te_cols.append(m.decision_function(Zte))
        except Exception: pass
    if len(dets) < 5: continue
    Sval = np.column_stack(val_cols); Ste = np.column_stack(te_cols)
    for cn, fn in CRIT:
        res[cn]["normals"].append(o - ap[dets[int(np.argmax(fn(Sval)))]])
        res[cn]["transductive"].append(o - ap[dets[int(np.argmax(fn(Ste)))]])
    if (fi + 1) % 30 == 0: print(f"  {fi+1}/{len(files)}", flush=True)
print("\n=== #7 transductive control (classical pool) ===")
print(f"  {'criterion':<16} {'normals-only':>13} {'transductive':>13}  (fixed detector 0.130, random 0.219)")
for cn, _ in CRIT:
    n = np.array(res[cn]["normals"]); t = np.array(res[cn]["transductive"])
    print(f"  {cn:<16} {n.mean():>13.4f} {t.mean():>13.4f}")
