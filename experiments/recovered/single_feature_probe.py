# -*- coding: utf-8 -*-
"""Single-feature (MARGINAL) probe: move ONE feature of a normal into its low-probability tail (drawn
from that feature's observed tail values), keeping the rest. Grade classical detectors by AUC vs val
normals. Does it, alone or COMBINED/ROUTED with SPARC's dependency probe, beat SPARC (0.113)?
INVARIANT (stated in advance): the locally-computed dependency probe must reproduce ~0.113, else the
pipeline is wrong and marginal numbers are untrustworthy. Classical pool only (matched to SPARC).
"""
import glob, os, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata, wilcoxon
S = r"E:/tmp/claude/E--Projects-Submitted-ADRank/236d6247-42ad-4e62-9828-db625dfa055d/scratchpad"
BUN = os.path.join(S, "streamline", "bundle_v2")
CLS = os.path.join(S, "clsstd_valuni")

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
DIST = lambda d: d.startswith(("HBOS", "ECOD", "COPOD"))

def dep_probe(Zval, ns=200, frac=0.4, seed=0):
    rng = np.random.default_rng(seed); n, d = Zval.shape; out = Zval[rng.integers(n, size=ns)].copy()
    for row in out:
        for j in rng.choice(d, max(1, int(frac * d)), replace=False): row[j] = Zval[rng.integers(n), j]
    return out

def marg_probe(Zval, ns=200, seed=1):
    """Move ONE random feature into its low-prob tail (observed value beyond the 5th/95th pct)."""
    rng = np.random.default_rng(seed); n, d = Zval.shape; out = Zval[rng.integers(n, size=ns)].copy()
    tails = []
    for j in range(d):
        col = Zval[:, j]; lo, hi = np.percentile(col, 5), np.percentile(col, 95)
        low = col[col <= lo]; high = col[col >= hi]; tails.append((low, high))
    for row in out:
        j = rng.integers(d); low, high = tails[j]
        pool = high if rng.random() < 0.5 else low
        if len(pool): row[j] = pool[rng.integers(len(pool))]
    return out

clsap = {}
import json
for l in open(os.path.join(CLS, "ap.jsonl")):
    r = json.loads(l); clsap[r["key"]] = r["ap"]
lb = pd.read_csv(os.path.join(S, "streamline", "STREAM_V3_LEADERBOARD.csv"))
keep = lb[(lb.oracle_ap > 0.05) & (lb.reg_random > 0.02) & (lb.n_h >= 100)].set_index("key")
oracle = keep.oracle_ap.to_dict()
files = [f for f in sorted(glob.glob(os.path.join(BUN, "*.npz"))) if os.path.basename(f)[:-4] in keep.index and os.path.basename(f)[:-4] in clsap]
print(f"datasets: {len(files)}", flush=True)

rec = []
for f in files:
    key = os.path.basename(f)[:-4]
    z = np.load(f); Xtr = z["Xtr"].astype("float64"); Xval = z["Xval"].astype("float64")
    ss = StandardScaler().fit(Xtr); Ztr = ss.transform(Xtr); Zval = ss.transform(Xval)
    if len(Zval) < 20: continue
    Pdep = dep_probe(Zval); Pmarg = marg_probe(Zval)
    ydep = np.r_[np.zeros(len(Zval)), np.ones(len(Pdep))]; ymarg = np.r_[np.zeros(len(Zval)), np.ones(len(Pmarg))]
    depg, margg = {}, {}
    for nm, ctor in classical_pool():
        try: m = ctor(); m.fit(Ztr)
        except Exception: continue
        sv = m.decision_function(Zval)
        depg[nm] = roc_auc_score(ydep, np.r_[sv, m.decision_function(Pdep)])
        margg[nm] = roc_auc_score(ymarg, np.r_[sv, m.decision_function(Pmarg)])
    ap = clsap[key]; dets = [d for d in depg if d in ap and d in margg]
    if len(dets) < 5: continue
    o = oracle[key]
    reg = lambda pick: o - ap[pick]
    dep_pick = max(dets, key=lambda d: depg[d]); marg_pick = max(dets, key=lambda d: margg[d])
    # combined: Borda over the two grades
    rd = rankdata([depg[d] for d in dets]); rm = rankdata([margg[d] for d in dets])
    comb_pick = dets[int(np.argmax(rd + rm))]
    # oracle router upper bound (per task best of the two picks)
    true_fam_marg = DIST(max(dets, key=lambda d: ap[d]))     # is this task a distribution-oracle task?
    rec.append(dict(key=key, reg_dep=reg(dep_pick), reg_marg=reg(marg_pick), reg_comb=reg(comb_pick),
                    reg_router=min(reg(dep_pick), reg(marg_pick)), dist_oracle=int(true_fam_marg),
                    marg_picks_dist=int(DIST(marg_pick))))
df = pd.DataFrame(rec)
print(f"\ntasks: {len(df)}")
print(f"INVARIANT dependency-probe regret = {df.reg_dep.mean():.4f}   (must be ~0.113)")
print(f"marginal-probe regret            = {df.reg_marg.mean():.4f}")
print(f"combined (Borda) regret          = {df.reg_comb.mean():.4f}   vs dep p={wilcoxon(df.reg_comb, df.reg_dep).pvalue:.3f}")
print(f"oracle dep-vs-marg router (UB)    = {df.reg_router.mean():.4f}   (headroom {df.reg_dep.mean()-df.reg_router.mean():.4f})")
dd = df[df.dist_oracle == 1]
print(f"\non the {len(dd)} DISTRIBUTION-oracle tasks (where dependency probe is blind):")
print(f"  dependency regret={dd.reg_dep.mean():.4f}  marginal regret={dd.reg_marg.mean():.4f}  combined={dd.reg_comb.mean():.4f}")
print(f"  marginal probe correctly picks a distribution detector on {dd.marg_picks_dist.sum()}/{len(dd)}")
df.to_csv(os.path.join(S, "SINGLE_FEATURE_PROBE.csv"), index=False)
