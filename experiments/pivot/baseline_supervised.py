# -*- coding: utf-8 -*-
"""Decisive incumbent check for the few-shot-selection lead: with the SAME k labeled anomalies, does using
them to TRAIN a model beat using them to CHOOSE among the 25 unsupervised detectors?

Tabular tasks only (time series has one anomaly event per task, see validate_cycle1). For each task and
draw (10 seeded draws, k in {5, 10}): remove the k anomalies from the test set; then score
  SEL     few-shot selection (pick argmax_j of mean fraction of validation normals the k anomalies outrank)
  FIX     LOF_k10
  LR      logistic regression on train normals (<=2000 sampled) vs the k anomalies, class-balanced
  RF      random forest (200 trees, balanced) on the same data
  KNNPOS  negative distance to the nearest labeled anomaly (a one-line supervised baseline)
  SELxRF  rank-average of SEL's detector and RF (does the selected detector add to a trained model?)
All on the reduced test set; regret against the classical-detector oracle on that set (trained models can
beat the oracle -> negative regret). Same draws for every method.
"""
import os, glob, numpy as np, pandas as pd, warnings
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); SC = os.path.join(HERE, "scores")
BUN = os.path.join(HERE, "..", "..", "artifacts", "modal", "bv2")
FIX = "LOF_k10"; KS = [5, 10]; NDRAW = 10


def apn(y, s): r = y.mean(); return (average_precision_score(y, s) - r) / (1 - r)
def frac_below(pos, val):
    vs = np.sort(val); lo = np.searchsorted(vs, pos, "left"); hi = np.searchsorted(vs, pos, "right")
    return float(np.mean((lo + 0.5 * (hi - lo)) / len(vs)))
def rk(s): return np.argsort(np.argsort(s)) / max(len(s) - 1, 1)


def task(key):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    z = np.load(os.path.join(SC, key + ".npz")); y = z["y_te"].astype(int)
    b = np.load(os.path.join(BUN, key + ".npz"))
    ss = StandardScaler().fit(b["Xtr"]); Ztr = ss.transform(b["Xtr"]); Zte = ss.transform(np.vstack([b["Xtn"], b["Xh"]]))
    dets = [k[4:] for k in z.files if k.startswith("te__")]
    TE = {d: z["te__" + d].astype(float) for d in dets}; VA = {d: z["val__" + d].astype(float) for d in dets}
    pos = np.where(y == 1)[0]
    rng = np.random.default_rng(int.from_bytes(key.encode()[:8].ljust(8, b"\0"), "little") % (2**32) + 7)
    trs = Ztr[rng.choice(len(Ztr), min(2000, len(Ztr)), replace=False)]
    out = []
    for k in KS:
        for _ in range(NDRAW):
            s = rng.choice(pos, k, replace=False); keep = np.ones(len(y), bool); keep[s] = False
            yk = y[keep]; ap = {d: apn(yk, TE[d][keep]) for d in dets}; o = max(ap.values())
            Q = {d: frac_below(TE[d][s], VA[d]) for d in dets}; bq = max(Q.values())
            pk = FIX if Q[FIX] >= bq - 1e-12 else max(Q, key=Q.get)
            X = np.vstack([trs, Zte[s]]); t = np.r_[np.zeros(len(trs)), np.ones(k)]
            lr = LogisticRegression(class_weight="balanced", max_iter=2000).fit(X, t).decision_function(Zte[keep])
            rf = RandomForestClassifier(200, class_weight="balanced", min_samples_leaf=1, random_state=0, n_jobs=1).fit(X, t).predict_proba(Zte[keep])[:, 1]
            dpos = -np.min(np.linalg.norm(Zte[keep][:, None, :] - Zte[s][None, :, :], axis=2), 1)
            out.append(dict(key=key, k=k, SEL=o - ap[pk], FIX=o - ap[FIX], LR=o - apn(yk, lr), RF=o - apn(yk, rf),
                            KNNPOS=o - apn(yk, dpos), SELxRF=o - apn(yk, rk(TE[pk][keep]) + rk(rf))))
    return out


if __name__ == "__main__":
    from multiprocessing import Pool
    rows = pd.read_pickle(os.path.join(HERE, "triage_cycle1.pkl"))
    keys = [r["key"] for r in rows if r["source"] != "TS"]
    print(f"tabular tasks: {len(keys)}", flush=True)
    with Pool(3) as p:
        res = [x for lst in p.map(task, keys) for x in lst]
    df = pd.DataFrame(res); df.to_pickle(os.path.join(HERE, "baseline_supervised.pkl"))
    for k, g in df.groupby("k"):
        f = g.FIX.mean(); print(f"\nk={k}  (regret; G = gap closed vs fixed)")
        for m in ["FIX", "SEL", "LR", "RF", "KNNPOS", "SELxRF"]:
            t = g.groupby("key")[m].mean(); print(f"  {m:7s} regret {g[m].mean():7.4f}   G {1 - g[m].mean()/f:6.3f}")
        tk = g.groupby("key")[["SEL", "RF", "LR", "SELxRF"]].mean()
        print(f"  per-task: SEL better than RF on {int((tk.SEL < tk.RF).sum())}/{len(tk)}; SELxRF better than SEL on {int((tk.SELxRF < tk.SEL).sum())}/{len(tk)}")
