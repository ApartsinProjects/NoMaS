# -*- coding: utf-8 -*-
"""Are the benchmark anomalies TRIVIAL, and does that distort the detector ranking?

User's hypothesis: many real anomalies are single-value extremes (one feature far out of
range). Every detector catches those, so the true ranking is dominated by easy cases and
the "which detector is best" question is measured on the wrong population. Ranking should
be scored on NON-TRIVIAL anomalies only.

Triviality is label-based, which is legitimate here: it redefines the EVALUATION TARGET
(the true ranking is already label-based). The NoMaS method itself never sees any of this.

Definitions of trivial (per anomaly point):
  maxz    max |z-score| over features >= 3   -> extreme in at least one feature
  hbos    flagged by a per-feature histogram detector in its top-k
  consensus  every detector in the panel scores it above the 90th pct of normals

Outputs per dataset:
  frac_trivial            how many anomalies are trivial
  spread_all / spread_hard  max-minus-mean true AUC over detectors, before/after excluding
                            trivial anomalies (bigger spread = the choice matters more)
  best_all / best_hard      which detector wins, and its family, before/after
KEY TEST: is triviality HIGHER on datasets where a GLOBAL detector wins?
"""
import os, sys, glob, time, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"
sys.path.insert(0, os.path.join(ROOT, "src"))
from adrank.pipeline import fit_and_score  # noqa: E402

DETS = ["IForest", "LOF", "KNN", "ECOD", "COPOD", "HBOS", "PCA", "CBLOF", "LODA"]
LOCAL = {"LOF", "KNN", "CBLOF"}
GLOBAL = {"HBOS", "COPOD", "ECOD", "PCA"}
fam = lambda d: "local" if d in LOCAL else ("global" if d in GLOBAL else "other")
MAX_N = 6000
OUT = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d\trivial_anom.csv"

SETS = []
for sub in ("dami", "adbench"):
    for p in sorted(glob.glob(os.path.join(ROOT, "data", sub, "*.npz"))):
        SETS.append((sub, os.path.splitext(os.path.basename(p))[0]))

rows, t0 = [], time.time()
for sub, name in SETS:
    a = np.load(os.path.join(ROOT, "data", sub, name + ".npz"))
    X, y = np.asarray(a["X"], float), np.asarray(a["y"], int).ravel()
    rng = np.random.RandomState(0)
    if len(X) > MAX_N:
        keep = rng.choice(len(X), MAX_N, replace=False)
        X, y = X[keep], y[keep]
    if len(X) < 200 or y.sum() < 20:
        continue

    # score every point ONCE per detector (no refitting for the subset analysis)
    S = {}
    for d in DETS:
        s = fit_and_score(d, X, X)
        if s is not None:
            S[d] = s
    if len(S) < 6:
        continue

    Xn = X[y == 0]
    mu, sd = Xn.mean(0), Xn.std(0) + 1e-9
    Z = np.abs((X - mu) / sd)
    maxz = Z.max(1)                       # how extreme is the point in its MOST extreme feature
    anom = np.where(y == 1)[0]
    trivial = maxz[anom] >= 3.0           # single-feature extreme => trivial
    frac_triv = float(trivial.mean())

    norm_idx = np.where(y == 0)[0]
    hard = anom[~trivial]
    if len(hard) < 10:
        rows.append(dict(dataset=name, n=len(X), d=X.shape[1], n_anom=len(anom),
                         frac_trivial=frac_triv, n_hard=len(hard), status="too_few_hard"))
        print(f"{name}: frac_trivial={frac_triv:.2f} hard={len(hard)} SKIP", flush=True)
        continue

    def auc_on(idx_anom):
        yy = np.r_[np.ones(len(idx_anom)), np.zeros(len(norm_idx))]
        out = {}
        for d, s in S.items():
            out[d] = roc_auc_score(yy, np.r_[s[idx_anom], s[norm_idx]])
        return out

    all_auc, hard_auc = auc_on(anom), auc_on(hard)
    ba, bh = max(all_auc, key=all_auc.get), max(hard_auc, key=hard_auc.get)
    rows.append(dict(
        dataset=name, n=len(X), d=X.shape[1], n_anom=len(anom), n_hard=len(hard),
        frac_trivial=frac_triv, status="ok",
        best_all=ba, fam_all=fam(ba), best_hard=bh, fam_hard=fam(bh),
        auc_best_all=all_auc[ba], auc_best_hard=hard_auc[bh],
        spread_all=all_auc[ba] - np.mean(list(all_auc.values())),
        spread_hard=hard_auc[bh] - np.mean(list(hard_auc.values())),
        mean_auc_all=float(np.mean(list(all_auc.values()))),
        mean_auc_hard=float(np.mean(list(hard_auc.values()))),
    ))
    print(f"{name}: trivial={frac_triv:.2f} best_all={ba}({fam(ba)}) "
          f"best_hard={bh}({fam(bh)}) spread {all_auc[ba]-np.mean(list(all_auc.values())):.3f}"
          f"->{hard_auc[bh]-np.mean(list(hard_auc.values())):.3f} | {time.time()-t0:.0f}s", flush=True)

pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"\nwrote {OUT} rows={len(rows)} elapsed={time.time()-t0:.0f}s")
