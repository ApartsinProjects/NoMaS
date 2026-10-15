# -*- coding: utf-8 -*-
"""Recompute the three remaining UOMS baselines on the CURRENT basis (bundle_v2, STREAM_V3 oracle,
151 keep tasks), using the 32-detector validation scores saved by modal_valscores.py.
 - IREOS (Marques et al.): argmax chance-adjusted RBF-SVM separability of a detector's top-flagged val
   points. Pool selector -> compute on classical(25)/deep(7)/full(32).  [ireos_leaderboard.py logic]
 - IFOREST-R (Ma et al.): select the detector whose val ranking best correlates (Spearman) with a
   random-iForest reference. Pool selector -> 3 pools.
 - UDR: defined only on the stochastic iForest sub-pool (deterministic detectors have trivial perfect
   stability); recompute cross-seed ranking stability on iForest configs x seeds -> ONE value, not per-pool.
Regret = oracle - ap[pick]. Also per-task vectors for a Wilcoxon vs the fixed detector (LOF_k10)."""
import json, glob, os, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import IsolationForest
from scipy.stats import spearmanr, wilcoxon
S = r"E:/tmp/claude/E--Projects-Submitted-ADRank/236d6247-42ad-4e62-9828-db625dfa055d/scratchpad"
ST = os.path.join(S, "streamline"); BUN = os.path.join(ST, "bundle_v2")
DEEP = lambda d: d.startswith("D_")
VS = {}
for fn in glob.glob(os.path.join(ST, "valscores_results", "shard_*.jsonl")):
    for ln in open(fn):
        try: r = json.loads(ln)
        except Exception: continue
        if r.get("vs"): VS[r["key"]] = r
lb = pd.read_csv(os.path.join(ST, "STREAM_V3_LEADERBOARD.csv"))
keep = lb[(lb.oracle_ap > 0.05) & (lb.reg_random > 0.02) & (lb.n_h >= 100)].set_index("key")
oracle = keep.oracle_ap.to_dict(); corpus = keep.corpus.to_dict()
keys = [k for k in keep.index if k in VS]
print(f"datasets with val scores: {len(keys)} / {len(keep)}", flush=True)

def zval_of(key):
    z = np.load(os.path.join(BUN, key + ".npz")); Xtr = z["Xtr"].astype("float64"); Xval = z["Xval"].astype("float64")
    ss = StandardScaler().fit(Xtr); Ztr = ss.transform(Xtr); Zval = ss.transform(Xval)
    if len(Zval) > 800:
        rng = np.random.default_rng(0); sel = np.sort(rng.choice(len(Zval), 800, replace=False)); Zval = Zval[sel]
    return Ztr, Zval

def sep(Z, i, osub):
    X = np.vstack([Z[i][None], Z[osub]]); y = np.r_[1.0, np.zeros(len(osub))]
    try:
        clf = SVC(kernel="rbf", gamma="scale", C=100).fit(X, y)
        return float(1.0 / (1.0 + np.exp(-clf.decision_function(Z[i][None])[0])))
    except Exception: return 0.5

K, NSUB = 4, 150
def ireos_pick(Z, vs, names):
    n = len(Z); allidx = np.arange(n); rng = np.random.default_rng(0)
    rnd = rng.choice(allidx, K, replace=False)
    osub_c = rng.choice(np.setdiff1d(allidx, rnd), min(NSUB, n - K), replace=False)
    chance = np.mean([sep(Z, i, osub_c) for i in rnd])
    sc = {}
    for nm in names:
        sv = np.asarray(vs[nm], float)
        top = allidx[np.argsort(sv)[-K:]]
        osub = rng.choice(np.setdiff1d(allidx, top), min(NSUB, n - K), replace=False)
        sc[nm] = np.mean([sep(Z, i, osub) for i in top]) - chance
    return max(names, key=lambda v: sc[v])

def ifr_pick(Ztr, Z, vs, names, nref=5):
    ref = np.zeros(len(Z))
    for s in range(nref):
        m = IsolationForest(n_estimators=100, random_state=s).fit(Ztr)
        ref += -m.score_samples(Z)                              # higher = more anomalous
    best, bnm = -2, names[0]
    for nm in names:
        rho = spearmanr(np.asarray(vs[nm], float), ref).statistic
        if np.isfinite(rho) and rho > best: best, bnm = rho, nm
    return bnm

# ---- IREOS + IFOREST-R on 3 pools ----
POOLS = {"classical": lambda d: not DEEP(d), "deep": DEEP, "full": lambda d: True}
rec = {p: {"ireos": [], "ifr": []} for p in POOLS}
fixed = []
for ki, key in enumerate(keys):
    ap = VS[key]["ap"]; vs = VS[key]["vs"]; o = oracle[key]
    if "LOF_k10" not in ap: continue
    Ztr, Z = zval_of(key)
    if len(Z) != len(next(iter(vs.values()))): continue          # alignment guard
    fixed.append(o - ap["LOF_k10"])
    for p, filt in POOLS.items():
        names = [d for d in vs if filt(d) and d in ap]
        if len(names) < 5: continue
        rec[p]["ireos"].append(o - ap[ireos_pick(Z, vs, names)])
        rec[p]["ifr"].append(o - ap[ifr_pick(Ztr, Z, vs, names)])
    if (ki + 1) % 25 == 0: print(f"  {ki+1}/{len(keys)} done", flush=True)
print("\n=== IREOS / IFOREST-R regret by pool (current basis) ===")
for p in POOLS:
    ir = np.array(rec[p]["ireos"]); ifr = np.array(rec[p]["ifr"])
    print(f"  {p:9s}: IREOS {ir.mean():.4f} (n={len(ir)})   IFOREST-R {ifr.mean():.4f} (n={len(ifr)})")
print(f"  fixed detector (LOF_k10) = {np.mean(fixed):.4f}")
pd.to_pickle(rec, os.path.join(S, "UOMS_RECOMPUTE.pkl"))
print("saved UOMS_RECOMPUTE.pkl")
