# -*- coding: utf-8 -*-
"""Stress tests for the 'k labeled anomalies + logistic regression beats the unsupervised oracle' lead.
Tabular tasks, k=5, 10 seeded draws per task. Regret vs the oracle among the 25 unsupervised detectors on
the SAME evaluation set (negative = beats the oracle).

S1 UNSEEN ANOMALY TYPE: cluster each task's test anomalies into 3 subtypes (k-means on standardized
   features, seed 0). Draw the k labels from ONE subtype; evaluate only on the OTHER subtypes' anomalies
   plus all test normals. Methods: FIX (LOF_k10), SEL (few-shot selection), LR, and HYB = rank-average of
   LR and the fixed detector (a hedge). Pre-stated reading: if LR loses its edge here while FIX does not,
   supervised few-label training generalizes only to SEEN anomaly types.
S2 REALISTIC PREVALENCE: labels drawn as in the main run; evaluation set = all test normals plus a random
   subsample of the remaining anomalies at 2% prevalence (10 subsamples per draw). Same methods.
"""
import os, numpy as np, pandas as pd, warnings
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); SC = os.path.join(HERE, "scores")
BUN = os.path.join(HERE, "..", "..", "artifacts", "modal", "bv2")
FIX = "LOF_k10"; K = 5; NDRAW = 10


def apn(y, s): r = y.mean(); return (average_precision_score(y, s) - r) / (1 - r)
def frac_below(pos, val):
    vs = np.sort(val); lo = np.searchsorted(vs, pos, "left"); hi = np.searchsorted(vs, pos, "right")
    return float(np.mean((lo + 0.5 * (hi - lo)) / len(vs)))
def rk(s): return np.argsort(np.argsort(s)) / max(len(s) - 1, 1)


def evaluate(y, mask, TE, picks, scores):
    yk = y[mask]; ap = {d: apn(yk, TE[d][mask]) for d in TE}; o = max(ap.values())
    out = {m: o - ap[d] for m, d in picks.items()}
    out.update({m: o - apn(yk, s[mask]) for m, s in scores.items()})
    return out


def task(arg):
    key, src = arg
    from sklearn.linear_model import LogisticRegression
    from sklearn.cluster import KMeans
    z = np.load(os.path.join(SC, key + ".npz")); y = z["y_te"].astype(int)
    b = np.load(os.path.join(BUN, key + ".npz"))
    ss = StandardScaler().fit(b["Xtr"]); Ztr = ss.transform(b["Xtr"]); Zte = ss.transform(np.vstack([b["Xtn"], b["Xh"]]))
    dets = [k[4:] for k in z.files if k.startswith("te__")]
    TE = {d: z["te__" + d].astype(float) for d in dets}; VA = {d: z["val__" + d].astype(float) for d in dets}
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    rng = np.random.default_rng(int.from_bytes(key.encode()[:8].ljust(8, b"\0"), "little") % (2**32) + 23)
    trs = Ztr[rng.choice(len(Ztr), min(2000, len(Ztr)), replace=False)]
    def fit_scores(s):
        X = np.vstack([trs, Zte[s]]); t = np.r_[np.zeros(len(trs)), np.ones(len(s))]
        lr = LogisticRegression(class_weight="balanced", max_iter=2000).fit(X, t).decision_function(Zte)
        Q = {d: frac_below(TE[d][s], VA[d]) for d in dets}; bq = max(Q.values())
        pk = FIX if Q[FIX] >= bq - 1e-12 else max(Q, key=Q.get)
        return lr, pk
    res = []
    # ---- S1 unseen type ----
    lab = KMeans(3, n_init=10, random_state=0).fit_predict(Zte[pos])
    sizes = np.bincount(lab, minlength=3)
    for c in range(3):
        if sizes[c] < K + 5 or (len(pos) - sizes[c]) < 20: continue
        for _ in range(max(1, NDRAW // 3)):
            s = rng.choice(pos[lab == c], K, replace=False)
            lr, pk = fit_scores(s)
            m = np.zeros(len(y), bool); m[neg] = True; m[pos[lab != c]] = True
            r = evaluate(y, m, TE, {"FIX": FIX, "SEL": pk}, {"LR": lr, "HYB": rk(lr) + rk(TE[FIX])})
            r.update(test="S1_unseen", key=key, src=src); res.append(r)
            m2 = np.ones(len(y), bool); m2[s] = False                       # same draw, SEEN types too
            r = evaluate(y, m2, TE, {"FIX": FIX, "SEL": pk}, {"LR": lr, "HYB": rk(lr) + rk(TE[FIX])})
            r.update(test="S1_all", key=key, src=src); res.append(r)
    # ---- S2 realistic prevalence ----
    for _ in range(NDRAW):
        s = rng.choice(pos, K, replace=False); lr, pk = fit_scores(s)
        rest = np.setdiff1d(pos, s); na = max(2, int(round(0.02 / 0.98 * len(neg))))
        for _ in range(10):
            sub = rng.choice(rest, min(na, len(rest)), replace=False)
            m = np.zeros(len(y), bool); m[neg] = True; m[sub] = True
            r = evaluate(y, m, TE, {"FIX": FIX, "SEL": pk}, {"LR": lr, "HYB": rk(lr) + rk(TE[FIX])})
            r.update(test="S2_prev2pct", key=key, src=src); res.append(r)
    return res


if __name__ == "__main__":
    from multiprocessing import Pool
    rows = pd.read_pickle(os.path.join(HERE, "triage_cycle1.pkl"))
    args = [(r["key"], r["source"]) for r in rows if r["source"] != "TS"]
    with Pool(3) as p:
        res = [x for lst in p.map(task, args) for x in lst]
    df = pd.DataFrame(res); df.to_pickle(os.path.join(HERE, "stress_test.pkl"))
    for t, g in df.groupby("test"):
        tk = g.groupby("key")[["FIX", "SEL", "LR", "HYB"]].mean(); f = tk.FIX.mean()
        print(f"{t:12s} tasks={len(tk):3d}  regret FIX {f:.4f} SEL {tk.SEL.mean():.4f} LR {tk.LR.mean():.4f} HYB {tk.HYB.mean():.4f}"
              f" | G: SEL {1-tk.SEL.mean()/f:5.2f} LR {1-tk.LR.mean()/f:5.2f} HYB {1-tk.HYB.mean()/f:5.2f} | LR worse than FIX on {int((tk.LR > tk.FIX).sum())} tasks")
