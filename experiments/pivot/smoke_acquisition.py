# -*- coding: utf-8 -*-
"""Smoke test of the most likely sinking objection (label-acquisition realism) with the strategist's design
fixes, on the 112 development tabular tasks. Pre-registered expectation (second opinion): the gain over the
fixed default SHRINKS when labels are the anomalies the incumbent already caught.

Design (per task, 10 seeded repetitions):
- Split test anomalies 50/50 into a LABELING POOL and an EVALUATION set; split test normals 50/50 the same
  way. The evaluation set (eval anomalies + eval normals) is FIXED across methods and acquisition modes.
- Acquisition modes (labels come only from the labeling pool):
    IID5      5 random pool anomalies
    CAUGHT20  all pool anomalies inside LOF_k10's top-20 alerts over (pool anomalies + pool normals)
    CAUGHT50  same with top-50
    MISSED5   5 random pool anomalies from the half LOF_k10 ranks LOWEST among pool anomalies
- Methods: FIX LOF_k10; SEL few-shot selection; LR class-balanced logistic regression (train normals <=2000
  vs labels); HYBi INDUCTIVE hedge = mean of empirical CDFs of LR and LOF_k10, each estimated on the
  validation normals (no test-batch information). If a mode yields 0 labels, LR/SEL/HYBi fall back to FIX.
- Regret against the oracle over the 25 unsupervised detectors on the fixed evaluation set; gap closed
  G = 1 - mean R / mean R_FIX. Also mean number of labels obtained.
"""
import os, numpy as np, pandas as pd, warnings
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); SC = os.path.join(HERE, "scores")
BUN = os.path.join(HERE, "..", "..", "artifacts", "modal", "bv2")
FIX = "LOF_k10"; NREP = 10


def apn(y, s): r = y.mean(); return (average_precision_score(y, s) - r) / (1 - r)
def frac_below(pos, val):
    vs = np.sort(val); lo = np.searchsorted(vs, pos, "left"); hi = np.searchsorted(vs, pos, "right")
    return float(np.mean((lo + 0.5 * (hi - lo)) / len(vs)))
def ecdf(cal, x):
    cs = np.sort(cal); return np.searchsorted(cs, x, "right") / len(cs)


def task(arg):
    key, src = arg
    from sklearn.linear_model import LogisticRegression
    z = np.load(os.path.join(SC, key + ".npz")); y = z["y_te"].astype(int)
    b = np.load(os.path.join(BUN, key + ".npz"))
    ss = StandardScaler().fit(b["Xtr"]); Ztr = ss.transform(b["Xtr"]); Zval = ss.transform(b["Xval"])
    Zte = ss.transform(np.vstack([b["Xtn"], b["Xh"]]))
    dets = [k[4:] for k in z.files if k.startswith("te__")]
    TE = {d: z["te__" + d].astype(float) for d in dets}; VA = {d: z["val__" + d].astype(float) for d in dets}
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    rng = np.random.default_rng(int.from_bytes(key.encode()[:8].ljust(8, b"\0"), "little") % (2**32) + 31)
    trs = Ztr[rng.choice(len(Ztr), min(2000, len(Ztr)), replace=False)]
    res = []
    for _ in range(NREP):
        pp = rng.permutation(pos); pn = rng.permutation(neg)
        poolA, evalA = pp[: len(pp) // 2], pp[len(pp) // 2:]; poolN, evalN = pn[: len(pn) // 2], pn[len(pn) // 2:]
        ev = np.r_[evalN, evalA]; ye = y[ev]
        ap = {d: apn(ye, TE[d][ev]) for d in dets}; o = max(ap.values())
        lof_pool = TE[FIX]; cand = np.r_[poolA, poolN]; order = cand[np.argsort(-lof_pool[cand])]
        modes = {"IID5": rng.choice(poolA, min(5, len(poolA)), replace=False),
                 "CAUGHT20": np.intersect1d(order[:20], poolA), "CAUGHT50": np.intersect1d(order[:50], poolA),
                 "MISSED5": rng.choice(poolA[np.argsort(lof_pool[poolA])][: max(5, len(poolA) // 2)], 5, replace=False)}
        for mode, lab in modes.items():
            r = dict(key=key, src=src, mode=mode, k=len(lab), FIX=o - ap[FIX])
            if len(lab) == 0:
                r.update(SEL=r["FIX"], LR=r["FIX"], HYBi=r["FIX"]); res.append(r); continue
            Q = {d: frac_below(TE[d][lab], VA[d]) for d in dets}; bq = max(Q.values())
            pk = FIX if Q[FIX] >= bq - 1e-12 else max(Q, key=Q.get)
            X = np.vstack([trs, Zte[lab]]); t = np.r_[np.zeros(len(trs)), np.ones(len(lab))]
            m = LogisticRegression(class_weight="balanced", max_iter=2000).fit(X, t)
            lr_ev = m.decision_function(Zte[ev]); lr_val = m.decision_function(Zval)
            hyb = 0.5 * (ecdf(lr_val, lr_ev) + ecdf(VA[FIX], TE[FIX][ev]))
            r.update(SEL=o - ap[pk], LR=o - apn(ye, lr_ev), HYBi=o - apn(ye, hyb)); res.append(r)
    return res


if __name__ == "__main__":
    from multiprocessing import Pool
    rows = pd.read_pickle(os.path.join(HERE, "triage_cycle1.pkl"))
    args = [(r["key"], r["source"]) for r in rows if r["source"] != "TS"]
    with Pool(3) as p:
        res = [x for lst in p.map(task, args) for x in lst]
    df = pd.DataFrame(res); df.to_pickle(os.path.join(HERE, "smoke_acquisition.pkl"))
    print(f"{len(args)} tabular tasks x {NREP} reps; fixed evaluation set per rep\n")
    print(f"{'mode':9s} {'mean k':>6s} {'FIX':>7s} {'SEL':>7s} {'LR':>8s} {'HYBi':>8s} | {'G_SEL':>6s} {'G_LR':>6s} {'G_HYBi':>6s} | HYBi worse than FIX (tasks)")
    for mode in ["IID5", "CAUGHT20", "CAUGHT50", "MISSED5"]:
        g = df[df["mode"] == mode]; t = g.groupby("key")[["FIX", "SEL", "LR", "HYBi"]].mean(); f = t.FIX.mean()
        print(f"{mode:9s} {g.k.mean():6.1f} {f:7.4f} {t.SEL.mean():7.4f} {t.LR.mean():8.4f} {t.HYBi.mean():8.4f} |"
              f" {1-t.SEL.mean()/f:6.2f} {1-t.LR.mean()/f:6.2f} {1-t.HYBi.mean()/f:6.2f} | {int((t.HYBi > t.FIX + 1e-9).sum())}/{len(t)}")
