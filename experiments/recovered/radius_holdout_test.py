# -*- coding: utf-8 -*-
"""Fixed-RADIUS void hold-out probe (user refinement 2): at a dense seed, delete ALL points within
radius rho = mult * median-nn-dist -> empties a ball of controlled depth, so a genuine void of depth
~rho forms (unlike kNN, which empties a tiny ball in dense regions). Verify void depth, then grade.
Also reports whether the FAILING datasets are even multimodal enough for a void to form at all.
Sweeps rho multiplier for the void geometry (cheap), fits the pool once at the depth that best forms a void."""
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

def radius_void(Ztr, mult, n_voids=6):
    """Delete all points within rho=mult*median-nn of the densest-by-count seeds; keep voids disjoint.
    Returns mask and the realized void depth (median P-point dist to nearest surviving normal / spacing)."""
    n = len(Ztr)
    nn2 = NearestNeighbors(n_neighbors=2).fit(Ztr); d1 = nn2.kneighbors(Ztr, return_distance=True)[0][:, 1]
    med = np.median(d1) + 1e-12; rho = mult * med
    rn = NearestNeighbors(radius=rho).fit(Ztr); within = rn.radius_neighbors(Ztr, return_distance=False)
    dens = np.array([len(w) for w in within])
    mask = np.zeros(n, bool); chosen = 0
    for s in np.argsort(-dens):
        if chosen >= n_voids: break
        ball = within[s]
        if mask[ball].any(): continue
        mask[ball] = True; chosen += 1
    if mask.sum() < 10 or (~mask).sum() < 30: return mask, np.nan
    nnR = NearestNeighbors(n_neighbors=1).fit(Ztr[~mask])
    depth = np.median(nnR.kneighbors(Ztr[mask], return_distance=True)[0][:, 0]) / med
    return mask, float(depth)

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
picks = [k for k, _ in scored[:8]]
print(f"FAILING (deep-loses) datasets: {len(picks)}\n", flush=True)

rows = []
for key in picks:
    f = os.path.join(BUN, key + ".npz")
    if not os.path.exists(f) or key not in DB: continue
    z = np.load(f); Xtr = z["Xtr"].astype("float64"); Xval = z["Xval"].astype("float64")
    ss = StandardScaler().fit(Xtr); Ztr = ss.transform(Xtr); Zval = ss.transform(Xval)
    if len(Ztr) < 80: continue
    # cheap geometry sweep: which rho gives the deepest void?
    depths = {m: radius_void(Ztr, m)[1] for m in [1.5, 2.5, 4.0, 6.0]}
    best_m = max([m for m in depths if not np.isnan(depths[m])], key=lambda m: depths[m], default=2.5)
    mask, depth = radius_void(Ztr, best_m)
    if mask.sum() < 15 or np.isnan(depth):
        print(f"  {key[:30]}: no void formable (max depth {max([d for d in depths.values() if not np.isnan(d)] or [0]):.1f})", flush=True)
        rows.append(dict(key=key[:26], best_rho=best_m, void=round(max([d for d in depths.values() if not np.isnan(d)] or [0]),1), nP=int(mask.sum()), ho_pick="(no void)", reg_ho=None, reg_bestcls=None, deep_ho=None, neigh_ho=None)); continue
    R = Ztr[~mask]; P = Ztr[mask]; y = np.r_[np.zeros(len(Zval)), np.ones(len(P))]
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
    rows.append(dict(key=key[:26], best_rho=best_m, void=round(depth, 1), nP=int(mask.sum()),
                     ho_pick=ho_pick[:10], reg_ho=round(orc - ap[ho_pick], 3),
                     reg_bestcls=round(orc - ap[best_cls], 3),
                     deep_ho=round(deep_ho, 2), neigh_ho=round(neigh_ho, 2)))
    print("done", key[:28], "void=%.1f" % depth, "pick", ho_pick, flush=True)
df = pd.DataFrame(rows)
print("\n=== FIXED-RADIUS void hold-out probe on FAILING datasets ===")
print("void = realized void depth (P-pt dist to nearest surviving normal / spacing; >1.5 = genuine)")
print(df.to_string(index=False))
vg = df[df.void > 1.5]
print(f"\ngenuine void formed (>1.5) on {(df.void>1.5).sum()}/{len(df)} FAILING datasets")
if len(vg) and vg.reg_ho.notna().any():
    print(f"  on those: mean regret probe = {vg.reg_ho.dropna().mean():.3f}  best-classical-oracle = {vg.reg_bestcls.dropna().mean():.3f}")
    print(f"  distinct picks where void formed: {sorted(vg[vg.ho_pick!='(no void)'].ho_pick.unique())}")
df.to_csv(os.path.join(S, "RADIUS_HOLDOUT_TEST.csv"), index=False)
