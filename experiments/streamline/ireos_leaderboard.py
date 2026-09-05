# -*- coding: utf-8 -*-
"""IREOS baseline (leak-free): score each detector's solution by the chance-adjusted separability of its
top-flagged VAL points (RBF-SVM max-margin per candidate). Pick argmax IREOS; regret vs oracle. Uses
bundle Xval + saved val scores (deepres3) + ap (jsonl). Incremental per-dataset CSV + resume."""
import os, sys, json, glob, warnings
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
warnings.filterwarnings("ignore")
S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
D = os.path.join(S, "scratchpad", "streamline"); SRC = os.path.join(D, "deepres3"); BUN = os.path.join(D, "bundle")
LOCAL = ("LOF", "KNN", "CBLOF"); GLOBAL = ("HBOS", "COPOD", "ECOD", "PCA")
fam = lambda v: "local" if str(v).startswith(LOCAL) else ("global" if str(v).startswith(GLOBAL) else "other")
OUT = os.path.join(D, "STREAM_IREOS.csv"); K = 4; NSUB = 150
J = {}
for fn in glob.glob(os.path.join(SRC, "shard_*.jsonl")):
    for ln in open(fn):
        ln = ln.strip()
        if ln:
            r = json.loads(ln)
            if "ap" in r and len(r.get("ap", {})) >= 6: J[r["key"]] = r
def sep(Z, i, osub):
    X = np.vstack([Z[i][None], Z[osub]]); y = np.r_[1.0, np.zeros(len(osub))]
    try:
        clf = SVC(kernel="rbf", gamma="scale", C=100).fit(X, y)
        return float(1.0 / (1.0 + np.exp(-clf.decision_function(Z[i][None])[0])))
    except Exception: return 0.5
done = set(pd.read_csv(OUT).dataset) if os.path.exists(OUT) else set(); fo = open(OUT, "a")
if not done: fo.write("key,truefam,reg_ireos,reg_oracle\n"); fo.flush()
for key, r in J.items():
    ds = key.split("__", 1)[1] if "__" in key else key
    if ds in done: continue
    scf = os.path.join(SRC, f"sc_{key}.npz"); bf = os.path.join(BUN, key + ".npz")
    if not (os.path.exists(scf) and os.path.exists(bf)): continue
    z = np.load(scf); Xval = np.load(bf)["Xval"].astype(float)
    names = [n for n in r["ap"] if f"val__{n}" in z.files]
    if len(names) < 6: continue
    Z = StandardScaler().fit_transform(Xval); n = len(Z)
    if n > 600: rng0 = np.random.default_rng(0); keep = rng0.choice(n, 600, replace=False); Z = Z[keep]; smap = {i: j for j, i in enumerate(keep)}
    else: keep = np.arange(n); smap = {i: i for i in range(n)}
    rng = np.random.default_rng(0); allidx = np.arange(len(Z))
    rnd = rng.choice(allidx, K, replace=False); osub_c = rng.choice(np.setdiff1d(allidx, rnd), min(NSUB, len(allidx) - K), replace=False)
    chance = np.mean([sep(Z, i, osub_c) for i in rnd])                    # detector-independent chance, once per dataset
    ap = r["ap"]; best = max(ap.values()); ir = {}
    for nm in names:
        sv = z[f"val__{nm}"].astype(float)[keep] if len(keep) < n else z[f"val__{nm}"].astype(float)
        top = allidx[np.argsort(sv)[-K:]]; osub = rng.choice(np.setdiff1d(allidx, top), min(NSUB, len(allidx) - K), replace=False)
        ir[nm] = np.mean([sep(Z, i, osub) for i in top]) - chance
    pick = max(names, key=lambda v: ir[v])
    loc = [ap[v] for v in names if fam(v) == "local"]; glo = [ap[v] for v in names if fam(v) == "global"]
    tf = "local" if (loc and (not glo or max(loc) > max(glo))) else "global"
    fo.write(f"{key},{tf},{best-ap[pick]:.4f},0.0\n"); fo.flush()
    print(f"  {ds[:24]} ireos_pick={pick[:12]} reg={best-ap[pick]:.3f}", flush=True)
fo.close()
df = pd.read_csv(OUT)
def macro(c): return np.mean([df[df.truefam == fa][c].mean() for fa in ["local", "global"] if (df.truefam == fa).any()])
# compare against the final leaderboard MV
fl = pd.read_csv(os.path.join(D, "STREAM_FINAL_LEADERBOARD.csv"))
print(f"\n=== IREOS (leak-free) over full pool ({len(df)} datasets) ===")
print(f"  IREOS          micro {df.reg_ireos.mean():.3f}  macro {macro('reg_ireos'):.3f}  median {df.reg_ireos.median():.3f}")
print(f"  (for reference from final leaderboard: MV macro {np.mean([fl[fl.truefam==fa].reg_mv.mean() for fa in ['local','global']]):.3f}, combined 0.156, random {np.mean([fl[fl.truefam==fa].reg_random.mean() for fa in ['local','global']]):.3f})")
print("saved streamline/STREAM_IREOS.csv")
