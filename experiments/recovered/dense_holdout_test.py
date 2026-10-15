# -*- coding: utf-8 -*-
"""Dense-void hold-out probe (user refinement): hold out COMPACT DENSE clusters (large-K seed balls in
the densest regions) so removing them creates ACTUAL distribution voids, not just thinning. This should
(a) break the LOF_k3 collapse and (b) make the pseudo-anomalies resemble genuine between-mode voids.
Test on FAILING (deep-loses) datasets, small scale. VERIFY a real void forms before trusting the grade.
"""
import json, glob, os, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
S = r"E:/tmp/claude/E--Projects-Submitted-ADRank/236d6247-42ad-4e62-9828-db625dfa055d/scratchpad"
BUN = os.path.join(S, "streamline", "bundle_v2"); DEEP = lambda d: d.startswith("D_")
NEIGH = lambda d: d.startswith(("LOF", "KNN"))

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

def deep_ctors():
    out = []
    try:
        from pyod.models.auto_encoder import AutoEncoder; out.append(("D_AE", lambda: AutoEncoder(epoch_num=50, verbose=0)))
    except Exception: pass
    try:
        from deepod.models import DeepSVDD; out.append(("D_DeepSVDD", lambda: DeepSVDD(epochs=40, random_state=0, device="cpu")))
    except Exception: pass
    return out

def dense_void_partition(Ztr, K=30, n_voids=6, seed=0):
    n = len(Ztr); K = max(5, min(K, n // 4))
    nn = NearestNeighbors(n_neighbors=K + 1).fit(Ztr); dist, idx = nn.kneighbors(Ztr)
    radius = dist[:, -1]                                    # kNN radius; small = dense
    mask = np.zeros(n, bool); chosen = 0
    for s in np.argsort(radius):                            # densest seeds first
        if chosen >= n_voids: break
        ball = idx[s]
        if mask[ball].any(): continue                      # keep voids disjoint
        mask[ball] = True; chosen += 1
    return mask

DB = {}
for fn in glob.glob(os.path.join(S, "streamline", "debias_results", "**", "shard_*.jsonl"), recursive=True):
    for ln in open(fn):
        try: r = json.loads(ln)
        except: continue
        if r.get("ap"): DB[r["key"]] = r["ap"]
lb = pd.read_csv(os.path.join(S, "streamline", "STREAM_V3_LEADERBOARD.csv"))
keep = set(lb[(lb.oracle_ap > 0.05) & (lb.reg_random > 0.02) & (lb.n_h >= 100)].key)
scored = []
for k, ap in DB.items():
    if k not in keep: continue
    dc = [v for d, v in ap.items() if not DEEP(d)]; dd = [v for d, v in ap.items() if DEEP(d)]
    if dc and dd: scored.append((k, max(dd) - max(dc)))
scored.sort(key=lambda x: x[1])
picks = [k for k, _ in scored[:8]]                         # 8 worst deep-loses (FAILING datasets)
print(f"FAILING (deep-loses) datasets: {len(picks)}\n", flush=True)

rows = []
for key in picks:
    f = os.path.join(BUN, key + ".npz")
    if not os.path.exists(f) or key not in DB: continue
    z = np.load(f); Xtr = z["Xtr"].astype("float64"); Xval = z["Xval"].astype("float64")
    ss = StandardScaler().fit(Xtr); Ztr = ss.transform(Xtr); Zval = ss.transform(Xval)
    if len(Ztr) < 80: continue
    mask = dense_void_partition(Ztr); R = Ztr[~mask]; P = Ztr[mask]
    if len(P) < 15 or len(R) < 40: continue
    # VERIFY void: P-points' nearest-normal distance in R vs typical normal spacing
    nnR = NearestNeighbors(n_neighbors=1).fit(R)
    dnn_med = np.median(nnR.kneighbors(Zval, return_distance=True)[0][:, 0]) + 1e-9
    void_ratio = float(np.median(nnR.kneighbors(P, return_distance=True)[0][:, 0]) / dnn_med)
    y = np.r_[np.zeros(len(Zval)), np.ones(len(P))]
    ho_auc = {}
    for nm, ctor in classical_pool() + deep_ctors():
        try:
            import contextlib, io
            m = ctor()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()): m.fit(R)
            sv = m.decision_function(Zval); sp = m.decision_function(P)
            if np.nanstd(np.r_[sv, sp]) > 1e-12: ho_auc[nm] = roc_auc_score(y, np.r_[sv, sp])
        except Exception: pass
    ap = DB[key]; dets = [d for d in ho_auc if d in ap]
    if len(dets) < 5: continue
    orc = max(ap[d] for d in dets); best_cls = max([d for d in dets if not DEEP(d)], key=lambda d: ap[d])
    ho_pick = max(dets, key=lambda d: ho_auc[d])
    deep_ho = max([ho_auc[d] for d in dets if DEEP(d)], default=np.nan)
    neigh_ho = max([ho_auc[d] for d in dets if NEIGH(d)], default=np.nan)
    rows.append(dict(key=key[:28], nP=int(mask.sum()), void=round(void_ratio, 1),
                     oracle=round(orc, 3), ho_pick=ho_pick[:10], reg_ho=round(orc - ap[ho_pick], 3),
                     reg_bestcls=round(orc - ap[best_cls], 3),
                     deep_ho=round(deep_ho, 2), neigh_ho=round(neigh_ho, 2)))
    print("done", key[:30], "void=%.1f" % void_ratio, "pick", ho_pick, flush=True)
df = pd.DataFrame(rows)
print("\n=== DENSE-VOID hold-out probe on FAILING datasets ===")
print("void = P-point dist to nearest R-normal / typical normal spacing (>1.5 = genuine void)")
print(df.to_string(index=False))
if len(df):
    print(f"\nmean regret dense-void probe = {df.reg_ho.mean():.3f}   (best-classical-oracle = {df.reg_bestcls.mean():.3f})")
    print(f"genuine void formed (>1.5) on {(df.void>1.5).sum()}/{len(df)} datasets")
    print(f"picks deep on {(df.ho_pick.str.startswith('D_')).sum()}/{len(df)} ; distinct picks: {sorted(df.ho_pick.unique())}")
    df.to_csv(os.path.join(S, "DENSE_HOLDOUT_TEST.csv"), index=False)
