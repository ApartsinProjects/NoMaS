# -*- coding: utf-8 -*-
"""NO-SYNTHETIC probe (user's idea): partition train normals into P = sampled seeds + their kNN
(held-out local clusters) and R = the rest. Train each detector on R; grade it by AUC separating P
from fresh validation normals. Hope: a detector that flags the held-out local hole is sensitive to
local structure and generalizes to real local anomalies; deep bridges the hole (over-generalizes) so
scores LOW -> the probe de-biases deep for free. Compare to SPARC's dependency probe.
Real ap (deploy = trained on ALL normals) reused from the debias run for a matched regret."""
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

def holdout_partition(Ztr, frac=0.12, m=8, seed=0):
    """P = union of (seed + m kNN); cap |P| ~ frac*n. Returns index masks."""
    rng = np.random.default_rng(seed); n = len(Ztr)
    nn = NearestNeighbors(n_neighbors=m + 1).fit(Ztr); idx = nn.kneighbors(Ztr, return_distance=False)
    target = int(frac * n); Pset = set(); seeds = rng.permutation(n)
    for s in seeds:
        if len(Pset) >= target: break
        Pset.update(idx[s].tolist())
    P = np.array(sorted(Pset)); mask = np.zeros(n, bool); mask[P] = True
    return mask

# --- real ap from debias run ---
DB = {}
for fn in glob.glob(os.path.join(S, "streamline", "debias_results", "**", "shard_*.jsonl"), recursive=True):
    for ln in open(fn):
        try: r = json.loads(ln)
        except: continue
        if r.get("ap"): DB[r["key"]] = r["ap"]
lb = pd.read_csv(os.path.join(S, "streamline", "STREAM_V3_LEADERBOARD.csv"))
keep = set(lb[(lb.oracle_ap > 0.05) & (lb.reg_random > 0.02) & (lb.n_h >= 100)].key)
# stratify by deep gap
scored = []
for k, ap in DB.items():
    if k not in keep: continue
    dc = [v for d, v in ap.items() if not DEEP(d)]; dd = [v for d, v in ap.items() if DEEP(d)]
    if dc and dd: scored.append((k, max(dd) - max(dc)))
scored.sort(key=lambda x: x[1])
picks = [k for k, _ in scored[:8]] + [k for k, _ in scored[len(scored)//2-2:len(scored)//2+2]] + [k for k, _ in scored[-4:]]
picks = list(dict.fromkeys(picks))
print(f"testing {len(picks)} datasets (deep-loses / mid / deep-wins)\n", flush=True)

rows = []
for key in picks:
    f = os.path.join(BUN, key + ".npz")
    if not os.path.exists(f) or key not in DB: continue
    z = np.load(f); Xtr = z["Xtr"].astype("float64"); Xval = z["Xval"].astype("float64")
    ss = StandardScaler().fit(Xtr); Ztr = ss.transform(Xtr); Zval = ss.transform(Xval)
    if len(Ztr) < 60: continue
    mask = holdout_partition(Ztr); R = Ztr[~mask]; P = Ztr[mask]
    if len(P) < 15 or len(R) < 30: continue
    y = np.r_[np.zeros(len(Zval)), np.ones(len(P))]                 # normal=0, held-out hole=1
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
    orc = max(ap[d] for d in dets)
    ho_pick = max(dets, key=lambda d: ho_auc[d])
    deep_ho = max([ho_auc[d] for d in dets if DEEP(d)], default=np.nan)
    neigh_ho = max([ho_auc[d] for d in dets if NEIGH(d)], default=np.nan)
    rows.append(dict(key=key[:30], nP=int(mask.sum()), oracle=round(orc, 3),
                     deep_is_orc=int(DEEP(max(dets, key=lambda d: ap[d]))),
                     ho_pick=ho_pick[:10], reg_ho=round(orc - ap[ho_pick], 3),
                     deep_ho=round(deep_ho, 2), neigh_ho=round(neigh_ho, 2),
                     debias=("neigh>deep" if neigh_ho > deep_ho else "deep>=neigh")))
    print("done", key[:34], "ho_pick", ho_pick, flush=True)
df = pd.DataFrame(rows)
print("\n=== hold-out-cluster probe (no synthetic) ===")
print("deep_ho/neigh_ho = best deep vs best neighbor AUC on the hold-out probe (de-bias if neigh>deep)")
print(df.to_string(index=False))
if len(df):
    print(f"\nmean regret hold-out probe = {df.reg_ho.mean():.3f}  (SPARC-classical published 0.113; fixed 0.130)")
    print(f"de-biases deep (neigh>deep) on {(df.debias=='neigh>deep').sum()}/{len(df)} datasets")
    dl = df[df.deep_is_orc == 0]; dw = df[df.deep_is_orc == 1]
    print(f"  deep-LOSES tasks n={len(dl)}: mean reg={dl.reg_ho.mean():.3f}, picks deep on {(dl.ho_pick.str.startswith('D_')).sum()}")
    print(f"  deep-WINS  tasks n={len(dw)}: mean reg={dw.reg_ho.mean():.3f}, picks deep on {(dw.ho_pick.str.startswith('D_')).sum()}")
    df.to_csv(os.path.join(S, "HOLDOUT_PROBE_TEST.csv"), index=False)
