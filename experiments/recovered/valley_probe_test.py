# -*- coding: utf-8 -*-
"""kNN density-valley probing (user's recommended method), tested as a SELECTION probe.
Generator (per validation normals):
  1. kNN neighborhoods N_i, local scale r_i = d(x_i, x_i^(k)), density rho(x_i)=1/mean_k-dist.
  2. probe z = x_i + eps*r_i*u; newly exposed D(z)=N(z) minus N_i; representative newly exposed x_j.
  3. density valley along y(t)=(1-t)x_i+t x_j; accept z* = argmin rho if rho(z*)<alpha*min(rho_i,rho_j).
  4. dedup near-duplicates; rank by valley depth.
Then fit the classical pool + AE + DeepSVDD; per detector compute real ap, dependency-probe AUC,
valley-probe AUC. Compare which probe's top pick has lower regret and whether valley DE-BIASES deep.
"""
import json, glob, os, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import average_precision_score, roc_auc_score
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

def valley_probe(Z, n_target=200, k=10, eps=1.6, n_dir=6, alpha=0.6, seed=0):
    rng = np.random.default_rng(seed); n, d = Z.shape
    nn = NearestNeighbors(n_neighbors=min(k + 1, n)).fit(Z)
    dist, idx = nn.kneighbors(Z); r = dist[:, -1]; Ni = [set(row[1:]) for row in idx]
    rho_orig = 1.0 / (dist[:, 1:].mean(1) + 1e-9)
    def rho(pts):
        dd = nn.kneighbors(pts, n_neighbors=k, return_distance=True)[0]; return 1.0 / (dd.mean(1) + 1e-9)
    out, depths, seps = [], [], []
    for i in rng.permutation(n):
        if len(out) >= n_target: break
        for _ in range(n_dir):
            u = rng.normal(size=d); u /= np.linalg.norm(u) + 1e-12
            z = Z[i] + eps * r[i] * u
            nbrs = nn.kneighbors(z[None], n_neighbors=k, return_distance=False)[0]
            newn = list(set(nbrs) - Ni[i] - {i})
            if not newn: continue
            j = min(newn, key=lambda jj: np.linalg.norm(Z[jj] - z))
            ts = np.linspace(0.15, 0.85, 8); seg = (1 - ts)[:, None] * Z[i] + ts[:, None] * Z[j]
            rr = rho(seg); m = int(rr.argmin())
            if rr[m] < alpha * min(rho_orig[i], rho_orig[j]):
                out.append(seg[m]); depths.append(min(rho_orig[i], rho_orig[j]) / (rr[m] + 1e-9))
                seps.append(np.linalg.norm(Z[j] - Z[i]))
    if not out: return np.empty((0, d)), 0.0, 0.0
    P = np.array(out)
    # dedup: greedy, drop points within 0.25*median local scale of an accepted one
    thr = 0.25 * np.median(r); keep = []; used = np.zeros(len(P), bool)
    order = np.argsort(-np.array(depths))
    for oi in order:
        if used[oi]: continue
        keep.append(oi); dv = np.linalg.norm(P - P[oi], axis=1); used |= (dv < thr)
    P = P[keep]
    return P, float(np.mean(depths)), float(np.median(seps))

def apn(y, s):
    bb = y.mean(); return float((average_precision_score(y, s) - bb) / (1 - bb + 1e-12))

# ---- stratified dataset sample: deep-loses + deep-wins ----
J = {}
for fn in glob.glob(os.path.join(S, "streamline", "debias_results", "**", "shard_*.jsonl"), recursive=True):
    for ln in open(fn):
        try: r = json.loads(ln)
        except: continue
        if r.get("ap"): J[r["key"]] = r
lb = pd.read_csv(os.path.join(S, "streamline", "STREAM_V3_LEADERBOARD.csv"))
keep = set(lb[(lb.oracle_ap > 0.05) & (lb.reg_random > 0.02) & (lb.n_h >= 100)].key)
scored = []
for k, r in J.items():
    if k not in keep: continue
    ap = r["ap"]; dc = [v for d, v in ap.items() if not DEEP(d)]; dd = [v for d, v in ap.items() if DEEP(d)]
    if dc and dd: scored.append((k, max(dd) - max(dc)))       # >0 = deep wins, <0 = deep loses
scored.sort(key=lambda x: x[1])
picks = [k for k, _ in scored[:8]] + [k for k, _ in scored[-6:]]   # 8 deep-loses + 6 deep-wins
print(f"testing {len(picks)} datasets (deep-loses + deep-wins)\n")

rows = []
for key in picks:
    f = os.path.join(BUN, key + ".npz")
    if not os.path.exists(f): continue
    z = np.load(f); Xtr = z["Xtr"].astype("float64"); Xval = z["Xval"].astype("float64")
    Xtn = z["Xtn"].astype("float64"); Xh = z["Xh"].astype("float64")
    ss = StandardScaler().fit(Xtr); Ztr = ss.transform(Xtr); Zval = ss.transform(Xval)
    Zte = np.vstack([ss.transform(Xtn), ss.transform(Xh)]); yte = np.r_[np.zeros(len(Xtn)), np.ones(len(Xh))]
    Pv, depth, sep = valley_probe(Zval)
    if len(Pv) < 20: print(f"  {key[:34]}: only {len(Pv)} valley pts, skip"); continue
    # dependency probe (SPARC's): resample 40% of features from observed values
    rng = np.random.default_rng(0); Pd = Zval[rng.integers(len(Zval), size=200)].copy()
    for rrow in Pd:
        for jj in rng.choice(Zval.shape[1], max(1, int(0.4 * Zval.shape[1])), replace=False):
            rrow[jj] = Zval[rng.integers(len(Zval)), jj]
    yv = np.r_[np.zeros(len(Zval)), np.ones(len(Pv))]; yd = np.r_[np.zeros(len(Zval)), np.ones(len(Pd))]
    # nearest-normal dist of valley pts (normal-median units)
    nn0 = NearestNeighbors(n_neighbors=1).fit(Ztr)
    dnn_med = np.median(nn0.kneighbors(Zval, return_distance=True)[0][:, 0]) + 1e-9
    dnn_valley = float(np.median(nn0.kneighbors(Pv, return_distance=True)[0][:, 0]) / dnn_med)
    ap_, valley_auc, dep_auc = {}, {}, {}
    pool = classical_pool()
    for nm, ctor in pool:
        try: m = ctor(); m.fit(Ztr)
        except Exception: continue
        sv = m.decision_function(Zval); st = m.decision_function(Zte)
        if np.nanstd(st) > 1e-12: ap_[nm] = apn(yte, st)
        valley_auc[nm] = roc_auc_score(yv, np.r_[sv, m.decision_function(Pv)])
        dep_auc[nm] = roc_auc_score(yd, np.r_[sv, m.decision_function(Pd)])
    for nm, ctor, tag in [("D_AE", None, "ae"), ("D_DeepSVDD", None, "svdd")]:
        try:
            import contextlib, io
            if tag == "ae":
                from pyod.models.auto_encoder import AutoEncoder; m = AutoEncoder(epoch_num=50, verbose=0)
            else:
                from deepod.models import DeepSVDD; m = DeepSVDD(epochs=40, random_state=0, device="cpu")
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()): m.fit(Ztr)
            sv = m.decision_function(Zval); st = m.decision_function(Zte)
            if np.nanstd(st) > 1e-12: ap_[nm] = apn(yte, st)
            valley_auc[nm] = roc_auc_score(yv, np.r_[sv, m.decision_function(Pv)])
            dep_auc[nm] = roc_auc_score(yd, np.r_[sv, m.decision_function(Pd)])
        except Exception as e: print(f"   {nm} failed: {e}")
    dets = [d for d in ap_ if d in valley_auc and d in dep_auc]
    if len(dets) < 5: continue
    orc = max(ap_[d] for d in dets)
    dep_pick = max(dets, key=lambda d: dep_auc[d]); val_pick = max(dets, key=lambda d: valley_auc[d])
    deep_va = max([valley_auc[d] for d in dets if DEEP(d)], default=np.nan)
    neigh_va = max([valley_auc[d] for d in dets if NEIGH(d)], default=np.nan)
    deep_da = max([dep_auc[d] for d in dets if DEEP(d)], default=np.nan)
    neigh_da = max([dep_auc[d] for d in dets if NEIGH(d)], default=np.nan)
    rows.append(dict(key=key[:30], nP=len(Pv), depth=round(depth, 1), dnn=round(dnn_valley, 1),
                     oracle=round(orc, 3), deep_is_orc=int(DEEP(max(dets, key=lambda d: ap_[d]))),
                     dep_pick=dep_pick[:9], reg_dep=round(orc - ap_[dep_pick], 3),
                     val_pick=val_pick[:9], reg_val=round(orc - ap_[val_pick], 3),
                     dep_deepVneigh=f"{deep_da:.2f}/{neigh_da:.2f}", val_deepVneigh=f"{deep_va:.2f}/{neigh_va:.2f}"))
    print("done", key[:34])
df = pd.DataFrame(rows)
print("\n=== valley-probe vs dependency-probe as a SELECTION criterion ===")
print("depth=valley depth ratio; dnn=valley-pt dist to nearest normal (normal-median units)")
print("deepVneigh = best-deep-AUC / best-neighbor-AUC on that probe (probe favors deep if left>right)")
print(df.to_string(index=False))
if len(df):
    print(f"\nmean regret  dependency-probe = {df.reg_dep.mean():.3f}   valley-probe = {df.reg_val.mean():.3f}")
    dl = df[df.deep_is_orc == 0]; dw = df[df.deep_is_orc == 1]
    print(f"  on deep-LOSES tasks (n={len(dl)}): dep={dl.reg_dep.mean():.3f}  valley={dl.reg_val.mean():.3f}")
    print(f"  on deep-WINS  tasks (n={len(dw)}): dep={dw.reg_dep.mean():.3f}  valley={dw.reg_val.mean():.3f}")
    df.to_csv(os.path.join(S, "VALLEY_PROBE_TEST.csv"), index=False)
