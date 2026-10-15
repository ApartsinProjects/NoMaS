# -*- coding: utf-8 -*-
"""Validate the cycle-1 triage before reporting it (verify-before-report; negatives validated hardest).

C1  Portfolio null: fraction of test anomalies whose combined -log p sits at the calibration floor
    (p = 1/(n+1)) -> ties; rerun the LOF_k10+COPOD portfolio with a deterministic tie-break
    (max -log p, then mean of validation-standardized raw scores).
A1  Few-shot selection: absolute regret per source at k=5 (pick vs fixed vs classical oracle).
A2  Time-series representativeness: in TS bundles anomaly windows overlap (window 64, stride 16), so
    k random anomaly rows tend to come from one contiguous segment. Measure segment coverage of the
    sampled anomalies and re-score TS with SEGMENT-STRATIFIED draws (one window per distinct run of
    consecutive anomaly rows, as far as possible).
"""
import os, numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
HERE = os.path.dirname(os.path.abspath(__file__)); SC = os.path.join(HERE, "scores")
rows = pd.read_pickle(os.path.join(HERE, "triage_cycle1.pkl"))
FIX = "LOF_k10"

def apn(y, s): r = y.mean(); return (average_precision_score(y, s) - r) / (1 - r)
def tailp(cal, x):
    cs = np.sort(cal); n = len(cs); return (1 + n - np.searchsorted(cs, x, "left")) / (n + 1)
def frac_below(pos, val):
    vs = np.sort(val); lo = np.searchsorted(vs, pos, "left"); hi = np.searchsorted(vs, pos, "right")
    return float(np.mean((lo + 0.5 * (hi - lo)) / len(vs)))

# ---------- A1 ----------
df = pd.DataFrame([{"key": r["key"], "src": r["source"], "R5": np.mean(r["A_k5"]), "F5": np.mean(r["A_fix_k5"])} for r in rows])
print("A1  k=5 absolute regret by source (pick / fixed / tasks):")
for s, g in df.groupby("src"):
    print(f"    {s:9s} pick {g.R5.mean():.4f}  fixed {g.F5.mean():.4f}  n={len(g)}  worse>0.05: {int((g.R5-g.F5>0.05).sum())}  better>0.05: {int((g.F5-g.R5>0.05).sum())}")
tab = df[df.src != "TS"]
print(f"    TABULAR only: G5 = {1 - tab.R5.mean()/tab.F5.mean():.3f} (n={len(tab)})")

# ---------- C1 + A2 ----------
c1 = []; a2 = []
for r in rows:
    key = r["key"]; z = np.load(os.path.join(SC, key + ".npz")); y = z["y_te"].astype(int)
    dets = [k[4:] for k in z.files if k.startswith("te__")]
    TE = {d: z["te__" + d].astype(float) for d in dets}; VA = {d: z["val__" + d].astype(float) for d in dets}
    nv = len(VA[FIX]); idx = np.random.default_rng(0).permutation(nv); v1 = idx[: nv // 2]
    if "COPOD" in dets:
        nl = [-np.log(tailp(VA[d][v1], TE[d])) for d in (FIX, "COPOD")]
        comb = np.max(nl, 0); floor = -np.log(1 / (len(v1) + 1))
        zs = np.mean([(TE[d] - VA[d][v1].mean()) / (VA[d][v1].std() + 1e-12) for d in (FIX, "COPOD")], 0)
        tie = comb + 1e-6 * (zs - zs.min()) / (zs.max() - zs.min() + 1e-12)
        ap = {d: apn(y, TE[d]) for d in dets}; o = max(ap.values())
        c1.append(dict(src=r["source"], sat=float(np.mean(np.isclose(comb[y == 1], floor))),
                       reg_fix=o - ap[FIX], reg_port=o - apn(y, comb), reg_port_tb=o - apn(y, tie),
                       reg_zmean=o - apn(y, zs)))
    if r["source"] == "TS":
        pos = np.where(y == 1)[0]
        runs = np.split(pos, np.where(np.diff(pos) > 1)[0] + 1)          # consecutive anomaly rows
        rng = np.random.default_rng(1); cov = []; rs = []; rf = []
        for _ in range(30):
            s = rng.choice(pos, 5, replace=False)
            cov.append(len({next(i for i, ru in enumerate(runs) if v in ru) for v in s}))
            # segment-stratified draw: one window from each of up to 5 distinct runs
            order = rng.permutation(len(runs))[:5]
            ss = np.array([rng.choice(runs[i]) for i in order])
            if len(ss) < 5: ss = np.r_[ss, rng.choice(np.setdiff1d(pos, ss), 5 - len(ss), replace=False)]
            keep = np.ones(len(y), bool); keep[ss] = False
            apk = {d: apn(y[keep], TE[d][keep]) for d in dets}; ok = max(apk.values())
            Q = {d: frac_below(TE[d][ss], VA[d]) for d in dets}; b = max(Q.values())
            pk = FIX if Q[FIX] >= b - 1e-12 else max(Q, key=Q.get)
            rs.append(ok - apk[pk]); rf.append(ok - apk[FIX])
        a2.append(dict(key=r["key"], n_runs=len(runs), n_anom=len(pos), mean_cov=np.mean(cov),
                       R5_strat=np.mean(rs), F5=np.mean(rf), R5_random=np.mean(r["A_k5"])))
c1 = pd.DataFrame(c1); a2 = pd.DataFrame(a2)
print("\nC1  portfolio LOF_k10+COPOD: anomalies at the p-value floor (ties):",
      f"mean {c1.sat.mean():.0%}, median {c1.sat.median():.0%}")
print(f"    regret: fixed {c1.reg_fix.mean():.4f} | max -log p {c1.reg_port.mean():.4f} | same + tie-break {c1.reg_port_tb.mean():.4f}"
      f" | mean of standardized scores {c1.reg_zmean.mean():.4f}")
print("\nA2  time series (n=%d): distinct anomaly runs per task median %d; random 5-draw covers %.1f runs on average"
      % (len(a2), a2.n_runs.median(), a2.mean_cov.mean()))
print(f"    k=5 regret: random draws {a2.R5_random.mean():.4f} | segment-stratified draws {a2.R5_strat.mean():.4f} | fixed {a2.F5.mean():.4f}")
print(f"    tasks with a single anomaly run: {int((a2.n_runs == 1).sum())} of {len(a2)}")
