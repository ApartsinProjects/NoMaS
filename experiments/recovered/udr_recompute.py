# -*- coding: utf-8 -*-
"""UDR on the CURRENT basis (bundle_v2, STREAM_V3 oracle, 151 keep tasks). UDR is defined only on the
STOCHASTIC sub-pool: iForest configs (n_estimators x max_features) x seeds. Pick the config with the
highest cross-seed ranking STABILITY (median pairwise Spearman of val-normal scores); its ap_norm = mean
over its seeds on the test set. Regret vs the FULL-pool oracle (UDR can only pick an iForest config)."""
import glob, os, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score
from scipy.stats import spearmanr
S = r"E:/tmp/claude/E--Projects-Submitted-ADRank/236d6247-42ad-4e62-9828-db625dfa055d/scratchpad"
ST = os.path.join(S, "streamline"); BUN = os.path.join(ST, "bundle_v2")
lb = pd.read_csv(os.path.join(ST, "STREAM_V3_LEADERBOARD.csv"))
keep = lb[(lb.oracle_ap > 0.05) & (lb.reg_random > 0.02) & (lb.n_h >= 100)].set_index("key")
oracle = keep.oracle_ap.to_dict()
NEST, MAXF, SEEDS = [50, 100, 200], [0.5, 0.75, 1.0], [0, 1, 2]
def apn(y, s):
    bb = y.mean(); return float((average_precision_score(y, s) - bb) / (1 - bb + 1e-12))
regs, fixed = [], []
keys = [k for k in keep.index if os.path.exists(os.path.join(BUN, k + ".npz"))]
print(f"datasets: {len(keys)}", flush=True)
for ki, key in enumerate(keys):
    z = np.load(os.path.join(BUN, key + ".npz"))
    Xtr, Xval, Xtn, Xh = z["Xtr"].astype("float64"), z["Xval"].astype("float64"), z["Xtn"].astype("float64"), z["Xh"].astype("float64")
    ss = StandardScaler().fit(Xtr); Ztr, Zval = ss.transform(Xtr), ss.transform(Xval)
    Zte = ss.transform(np.vstack([Xtn, Xh])); yte = np.r_[np.zeros(len(Xtn)), np.ones(len(Xh))]
    best_stab, best_ap = -2, None
    for ne in NEST:
        for mf in MAXF:
            vs_seeds, ap_seeds = [], []
            for sd in SEEDS:
                try:
                    m = IsolationForest(n_estimators=ne, max_features=mf, random_state=sd).fit(Ztr)
                    vs_seeds.append(-m.score_samples(Zval)); ap_seeds.append(apn(yte, -m.score_samples(Zte)))
                except Exception: pass
            if len(vs_seeds) < 2: continue
            st = np.median([spearmanr(vs_seeds[a], vs_seeds[b]).statistic
                            for a in range(len(vs_seeds)) for b in range(a + 1, len(vs_seeds))])
            if np.isfinite(st) and st > best_stab: best_stab, best_ap = st, float(np.mean(ap_seeds))
    if best_ap is None: continue
    regs.append(oracle[key] - best_ap)
    if (ki + 1) % 25 == 0: print(f"  {ki+1}/{len(keys)}", flush=True)
regs = np.array(regs)
print(f"\n=== UDR (stochastic iForest sub-pool, current basis) ===")
print(f"  UDR regret = {regs.mean():.4f}  (n={len(regs)}; median {np.median(regs):.4f}; fixed detector 0.130)")
np.save(os.path.join(S, "UDR_REGRETS.npy"), regs)
