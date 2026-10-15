# -*- coding: utf-8 -*-
"""ResearchPivot cycle-1 triage (pre-registered by the strategist; thresholds fixed BEFORE running).

Substrate: per-point scores of the 25 classical detectors (scores/<key>.npz), tasks = the rebuilt keep set.
Candidate pool = classical detectors present for the task; fixed detector = LOF_k10; oracle = best classical
detector ON THE SAME EVALUATION SET.

ARM A (few labeled anomalies): for k in {1,2,3,5,10}, draw k distinct test anomalies (30 seeded draws),
remove them from the test set, pick argmax_j Q_j where Q_j = mean over the k anomalies of the fraction of
validation normals they outrank (ties 0.5). Ties between detectors break toward the fixed detector.
Regret of pick and of the fixed detector are both computed on the reduced test set. Gap recovery
G_k = 1 - mean R_k / mean R_fixed (pooled). Pre-registered (k=5): positive G5 >= 0.40 AND >= 70% of draws
improve on fixed (mean over tasks); kill G5 < 0.20 OR < 55% of draws improve.
Also an ABSTAIN variant (exploratory): switch from LOF_k10 only if Q_pick - Q_fixed >= 0.05.

ARM C (calibrated portfolio, no labels): LOF_k10 + COPOD; empirical normal-tail p-values from half the
validation normals (V1), score = max_j -log p_j; alert threshold = 95th percentile of the combined score on
the other half (V2). ap_norm on the full test set; regret vs the classical oracle (may be negative). Recall
at the V2-calibrated 5% budget vs LOF_k10 at its own V2 threshold; also the realized test-normal FPR.
Pre-registered: positive G_port >= 0.35 AND recall gain >= 20%; kill G_port < 0.10 OR recall gain < 5%.
Exploratory extras: LOF_k10+HBOS, LOF_k10+COPOD+PCA, mean-of-ranks LOF_k10+COPOD.
"""
import glob, os, json
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score

HERE = os.path.dirname(os.path.abspath(__file__))
SC = os.path.join(HERE, "scores")
KEEP = pd.read_csv(os.path.join(HERE, "..", "..", "artifacts", "LEADERBOARD_REBUILT_probe1000.csv"))
KEEP = KEEP[(KEEP.oracle > 0.05) & (KEEP.reg_random > 0.02) & (KEEP.n_h >= 100)]
SRC = dict(zip(KEEP.key, KEEP.source))
FIX = "LOF_k10"; KS = [1, 2, 3, 5, 10]; NDRAW = 30


def apn(y, s):
    r = y.mean(); return (average_precision_score(y, s) - r) / (1 - r)


def frac_below(pos, val):
    """mean over positives of P(val < pos) + 0.5 P(val == pos)."""
    vs = np.sort(val)
    lo = np.searchsorted(vs, pos, "left"); hi = np.searchsorted(vs, pos, "right")
    return float(np.mean((lo + 0.5 * (hi - lo)) / len(vs)))


def tailp(cal, x):
    cs = np.sort(cal); n = len(cs)
    ge = n - np.searchsorted(cs, x, "left")            # #cal >= x
    return (1 + ge) / (n + 1)


def task(key):
    z = np.load(os.path.join(SC, key + ".npz"))
    y = z["y_te"].astype(int)
    dets = [k[4:] for k in z.files if k.startswith("te__")]
    if FIX not in dets: return None
    TE = {d: z["te__" + d].astype(float) for d in dets}; VA = {d: z["val__" + d].astype(float) for d in dets}
    pos = np.where(y == 1)[0]
    out = {"key": key, "source": SRC.get(key, "?")}
    # ---------- ARM A ----------
    rng = np.random.default_rng(                              # deterministic per task (no builtin hash())
int.from_bytes(key.encode()[:8].ljust(8, b"\0"), "little") % (2**32))
    for k in KS:
        rs, rf, ra = [], [], []
        for _ in range(NDRAW):
            pick_idx = rng.choice(pos, size=min(k, len(pos) - 10), replace=False)
            keep = np.ones(len(y), bool); keep[pick_idx] = False
            ap = {d: apn(y[keep], TE[d][keep]) for d in dets}
            o = max(ap.values())
            Q = {d: frac_below(TE[d][pick_idx], VA[d]) for d in dets}
            best = max(Q.values()); cands = [d for d in dets if Q[d] >= best - 1e-12]
            pk = FIX if FIX in cands else cands[0]
            pa = pk if Q[pk] - Q[FIX] >= 0.05 else FIX
            rs.append(o - ap[pk]); rf.append(o - ap[FIX]); ra.append(o - ap[pa])
        out[f"A_k{k}"] = rs; out[f"A_fix_k{k}"] = rf; out[f"A_abst_k{k}"] = ra
    # ---------- ARM C ----------
    ap_full = {d: apn(y, TE[d]) for d in dets}; o_full = max(ap_full.values())
    nv = len(VA[FIX]); idx = np.random.default_rng(0).permutation(nv); v1, v2 = idx[: nv // 2], idx[nv // 2:]
    def port(members, mode="max"):
        if mode == "max":
            te = np.max([-np.log(tailp(VA[d][v1], TE[d])) for d in members], 0)
            va2 = np.max([-np.log(tailp(VA[d][v1], VA[d][v2])) for d in members], 0)
        else:  # mean of empirical tail ranks
            te = np.mean([-np.log(tailp(VA[d][v1], TE[d])) for d in members], 0)
            va2 = np.mean([-np.log(tailp(VA[d][v1], VA[d][v2])) for d in members], 0)
        return te, va2
    def rec_fpr(te, va2):
        thr = np.quantile(va2, 0.95)
        return float(np.mean(te[y == 1] > thr)), float(np.mean(te[y == 0] > thr))
    out["C_fix_reg"] = o_full - ap_full[FIX]
    out["C_fix_rec"], out["C_fix_fpr"] = rec_fpr(TE[FIX], VA[FIX][v2])
    for name, mem, mode in [("LC", [FIX, "COPOD"], "max"), ("LH", [FIX, "HBOS"], "max"),
                            ("LCP", [FIX, "COPOD", "PCA"], "max"), ("LCmean", [FIX, "COPOD"], "mean")]:
        if not all(m in dets for m in mem): continue
        te, va2 = port(mem, mode)
        out[f"C_{name}_reg"] = o_full - apn(y, te)
        out[f"C_{name}_rec"], out[f"C_{name}_fpr"] = rec_fpr(te, va2)
    return out


def main():
    keys = [k for k in KEEP.key if os.path.exists(os.path.join(SC, k + ".npz"))]
    print(f"tasks with scores: {len(keys)} of {len(KEEP)} kept", flush=True)
    from multiprocessing import Pool
    with Pool(3) as p:
        rows = [r for r in p.map(task, keys) if r]
    pd.to_pickle(rows, os.path.join(HERE, "triage_cycle1.pkl"))
    n = len(rows); print(f"analysed {n} tasks\n")
    print("ARM A: few labeled anomalies used only to choose a detector")
    print(f"{'k':>3s} {'R_k':>7s} {'R_fix':>7s} {'G_k':>6s} {'draws>fix':>9s} {'abstain R':>9s} {'abstain G':>9s}")
    for k in [1, 2, 3, 5, 10]:
        Rk = np.array([r[f"A_k{k}"] for r in rows]); Rf = np.array([r[f"A_fix_k{k}"] for r in rows]); Ra = np.array([r[f"A_abst_k{k}"] for r in rows])
        G = 1 - Rk.mean() / Rf.mean(); Ga = 1 - Ra.mean() / Rf.mean()
        imp = np.mean(Rk.mean(0) < Rf.mean(0))
        print(f"{k:>3d} {Rk.mean():7.4f} {Rf.mean():7.4f} {G:6.3f} {imp:9.0%} {Ra.mean():9.4f} {Ga:9.3f}")
    R5 = np.array([np.mean(r["A_k5"]) for r in rows]); F5 = np.array([np.mean(r["A_fix_k5"]) for r in rows])
    print("  k=5 by source (G):", {s: round(1 - R5[[r['source'] == s for r in rows]].mean() / F5[[r['source'] == s for r in rows]].mean(), 3) for s in sorted({r['source'] for r in rows})})
    print("  k=5 per-task: tasks worse than fixed by > 0.05:", int(np.sum(R5 - F5 > 0.05)), " better by > 0.05:", int(np.sum(F5 - R5 > 0.05)))
    print("\nARM C: calibrated no-label portfolio vs LOF_k10")
    fr = np.mean([r["C_fix_reg"] for r in rows]); frc = np.mean([r["C_fix_rec"] for r in rows]); ffp = np.mean([r["C_fix_fpr"] for r in rows])
    print(f"  LOF_k10        regret {fr:.4f}  recall@5% {frc:.3f}  test FPR {ffp:.3f}")
    for nm in ["LC", "LH", "LCP", "LCmean"]:
        rr = [r for r in rows if f"C_{nm}_reg" in r]
        if not rr: continue
        g = np.mean([r[f"C_{nm}_reg"] for r in rr]); rc = np.mean([r[f"C_{nm}_rec"] for r in rr]); fp = np.mean([r[f"C_{nm}_fpr"] for r in rr])
        print(f"  {nm:14s} regret {g:.4f}  G {1 - g / fr:6.3f}  recall@5% {rc:.3f} (gain {rc / frc - 1:+.1%})  test FPR {fp:.3f}  n={len(rr)}")


if __name__ == "__main__":
    main()
