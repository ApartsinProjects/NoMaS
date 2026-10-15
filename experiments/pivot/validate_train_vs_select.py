# -*- coding: utf-8 -*-
"""Break the 'k labeled anomalies + logistic regression beats the unsupervised oracle' result before
believing it. Tabular tasks, 10 seeded draws per (task, k), k in {1,2,3,5}.
  SEL      few-shot selection among the 25 detectors (as in triage)
  LR       logistic regression: train normals (<=2000) vs the k labeled anomalies, class-balanced
  LR_FAKE  NEGATIVE CONTROL: identical, but the k 'anomalies' are k random TEST NORMALS (removed from the
           test set the same way). Must NOT beat the fixed detector; if it does, the pipeline is broken.
Invariant stated in advance: LR_FAKE regret >= FIX regret (no information in fake labels).
Reports regret, gap closed G = 1 - R/R_fix, and the per-source breakdown at k=5.
"""
import os, numpy as np, pandas as pd, warnings
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); SC = os.path.join(HERE, "scores")
BUN = os.path.join(HERE, "..", "..", "artifacts", "modal", "bv2")
FIX = "LOF_k10"; KS = [1, 2, 3, 5]; NDRAW = 10


def apn(y, s): r = y.mean(); return (average_precision_score(y, s) - r) / (1 - r)
def frac_below(pos, val):
    vs = np.sort(val); lo = np.searchsorted(vs, pos, "left"); hi = np.searchsorted(vs, pos, "right")
    return float(np.mean((lo + 0.5 * (hi - lo)) / len(vs)))


def task(arg):
    key, src = arg
    from sklearn.linear_model import LogisticRegression
    z = np.load(os.path.join(SC, key + ".npz")); y = z["y_te"].astype(int)
    b = np.load(os.path.join(BUN, key + ".npz"))
    ss = StandardScaler().fit(b["Xtr"]); Ztr = ss.transform(b["Xtr"]); Zte = ss.transform(np.vstack([b["Xtn"], b["Xh"]]))
    dets = [k[4:] for k in z.files if k.startswith("te__")]
    TE = {d: z["te__" + d].astype(float) for d in dets}; VA = {d: z["val__" + d].astype(float) for d in dets}
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    rng = np.random.default_rng(int.from_bytes(key.encode()[:8].ljust(8, b"\0"), "little") % (2**32) + 11)
    trs = Ztr[rng.choice(len(Ztr), min(2000, len(Ztr)), replace=False)]
    def lr_fit(P):
        X = np.vstack([trs, P]); t = np.r_[np.zeros(len(trs)), np.ones(len(P))]
        return LogisticRegression(class_weight="balanced", max_iter=2000).fit(X, t)
    out = []
    for k in KS:
        for _ in range(NDRAW):
            s = rng.choice(pos, k, replace=False); keep = np.ones(len(y), bool); keep[s] = False
            yk = y[keep]; ap = {d: apn(yk, TE[d][keep]) for d in dets}; o = max(ap.values())
            Q = {d: frac_below(TE[d][s], VA[d]) for d in dets}; bq = max(Q.values())
            pk = FIX if Q[FIX] >= bq - 1e-12 else max(Q, key=Q.get)
            lr = lr_fit(Zte[s]).decision_function(Zte[keep])
            # negative control: k test normals posing as anomalies, removed from the test set
            f = rng.choice(neg, k, replace=False); kf = np.ones(len(y), bool); kf[f] = False
            yf = y[kf]; apf = {d: apn(yf, TE[d][kf]) for d in dets}; of = max(apf.values())
            lrf = lr_fit(Zte[f]).decision_function(Zte[kf])
            out.append(dict(key=key, src=src, k=k, FIX=o - ap[FIX], SEL=o - ap[pk], LR=o - apn(yk, lr),
                            FIX_f=of - apf[FIX], LR_FAKE=of - apn(yf, lrf)))
    return out


if __name__ == "__main__":
    from multiprocessing import Pool
    rows = pd.read_pickle(os.path.join(HERE, "triage_cycle1.pkl"))
    args = [(r["key"], r["source"]) for r in rows if r["source"] != "TS"]
    with Pool(3) as p:
        res = [x for lst in p.map(task, args) for x in lst]
    df = pd.DataFrame(res); df.to_pickle(os.path.join(HERE, "validate_train_vs_select.pkl"))
    print(f"{len(args)} tabular tasks, {NDRAW} draws each\n")
    print(f"{'k':>2s} {'FIX':>7s} {'SEL':>7s} {'LR':>8s} {'G_SEL':>6s} {'G_LR':>6s} | {'FIX_f':>7s} {'LR_FAKE':>8s}  invariant LR_FAKE>=FIX_f")
    for k, g in df.groupby("k"):
        print(f"{k:>2d} {g.FIX.mean():7.4f} {g.SEL.mean():7.4f} {g.LR.mean():8.4f} {1-g.SEL.mean()/g.FIX.mean():6.3f} {1-g.LR.mean()/g.FIX.mean():6.3f} |"
              f" {g.FIX_f.mean():7.4f} {g.LR_FAKE.mean():8.4f}  {'PASS' if g.LR_FAKE.mean() >= g.FIX_f.mean() else 'FAIL'}")
    g5 = df[df.k == 5]
    print("\nk=5 by source:")
    for s, g in g5.groupby("src"):
        t = g.groupby("key")[["FIX", "SEL", "LR"]].mean()
        print(f"  {s:9s} n={len(t):3d}  FIX {t.FIX.mean():.4f}  SEL {t.SEL.mean():.4f}  LR {t.LR.mean():.4f}  G_LR {1-t.LR.mean()/t.FIX.mean():.2f}"
              f"  LR beats oracle on {int((t.LR < 0).sum())}/{len(t)} tasks")
