# -*- coding: utf-8 -*-
"""Pre-registered few-label study harness (experiments/PREREGISTRATION_fewlabel.md). Run in .venv-baselines.

Per task: standardized bundle + precomputed per-point scores of the 25 unsupervised detectors on the test
set and validation normals. For each repetition r (seeded): test anomalies and test normals are split 50/50
into a LABELING POOL and a fixed EVALUATION set; validation normals are split 50/50 into V_fit (normal
class for score-feature learners) and V_cal (empirical-CDF calibration for every hedge). For each CELL
(acquisition mode, k, training contamination) labels are drawn ONLY from the pool (IID draws are nested:
the first k of one permutation), learners are fit, and every method scores the same evaluation set.
Metrics are computed at several evaluation prevalences (anomaly subsampling, 10 subsamples, averaged):
ap_norm, ROC-AUC, recall at 1% and 5% FPR with thresholds set on V_cal (inductive). Also recorded per
prevalence: the default D (LOF_k10) and the oracle O (best of the 25 detectors on that evaluation set).

Methods (group "cheap"): SEL (label-based selection), LR, RF, XGB, XGBOD (XGB on raw features + the 25
standardized detector scores; normal class = V_fit), KNNPOS, label-free picks (consensus, model-centrality,
HITS from V_cal scores; optional EM/MV/MetaOD picks from a picks file), and the hedge H(m) for every
learner m. Group "deep": DevNet, PReNet, DeepSAD, FeaWAD, RoSAS (deepod), TabPFN, and their hedges.
Output: one JSON line per (task, rep, cell). Resumable (skips (task, rep, cell) already written).
"""
import os, sys, json, glob, argparse, time, warnings
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
FIX = "LOF_k10"


# ------------------------------------------------------------------ metrics
def fast_ap(y, s):
    """Average precision identical to sklearn.average_precision_score (tie-aware step function), without
    sklearn's per-call validation overhead (profiled: it dominated the harness)."""
    o = np.argsort(-s, kind="mergesort"); ys = y[o]; ss = s[o]
    last = np.r_[np.where(np.diff(ss) != 0)[0], len(ss) - 1]          # last index of each distinct score
    tp = np.cumsum(ys)[last]; fp = (last + 1) - tp; P = ys.sum()
    prec = tp / (tp + fp); rec = tp / P
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def fast_auc(y, s):
    """ROC-AUC via the Mann-Whitney statistic with average ranks (ties count 0.5), = sklearn.roc_auc_score."""
    from scipy.stats import rankdata
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def apn(y, s):
    r = y.mean(); return float((fast_ap(y, s) - r) / (1 - r))


def ecdf(cal, x):
    cs = np.sort(cal); return np.searchsorted(cs, x, "right") / len(cs)


def _metrics(ye, sc, cal):
    """ap_norm, ROC-AUC, recall at 1% / 5% FPR with thresholds from calibration normals (inductive)."""
    th1, th5 = np.quantile(cal, 0.99), np.quantile(cal, 0.95)
    return (apn(ye, sc), fast_auc(ye, sc), float(np.mean(sc[ye == 1] > th1)), float(np.mean(sc[ye == 1] > th5)))


def rank_detectors_from_val(V):
    """consensus / model-centrality / HITS on a [n, m] validation score matrix (label-free)."""
    from scipy.stats import rankdata
    R = np.apply_along_axis(rankdata, 0, V); Rc = R - R.mean(0); den = np.sqrt((Rc ** 2).sum(0)); den[den == 0] = 1
    C = np.clip((Rc.T @ Rc) / np.outer(den, den), -1, 1); np.fill_diagonal(C, 1)
    cons = rankdata(R.mean(1)); cc = cons - cons.mean()
    consensus = (Rc * cc[:, None]).sum(0) / (den * np.sqrt((cc ** 2).sum()) + 1e-12)
    M = C.copy(); np.fill_diagonal(M, 0); mc = M.sum(1) / max(M.shape[0] - 1, 1)
    A = np.clip(M, 0, None); a = np.ones(A.shape[0])
    for _ in range(100):
        a = A.T @ (A @ a); n = np.linalg.norm(a)
        if n == 0: break
        a = a / n
    return {"CONS": int(np.argmax(consensus)), "MC": int(np.argmax(mc)), "HITS": int(np.argmax(a))}


# ------------------------------------------------------------------ learners
def fit_cheap(name, Xn, Xa, seed):
    """Return a scoring function f(X)->anomaly score, trained on normals Xn (label 0) and anomalies Xa (1)."""
    X = np.vstack([Xn, Xa]); t = np.r_[np.zeros(len(Xn)), np.ones(len(Xa))]
    if name == "LR":
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(class_weight="balanced", max_iter=2000).fit(X, t); return m.decision_function
    if name == "RF":
        from sklearn.ensemble import RandomForestClassifier
        m = RandomForestClassifier(200, class_weight="balanced", random_state=seed, n_jobs=1).fit(X, t)
        return lambda Z: m.predict_proba(Z)[:, 1]
    if name in ("XGB", "XGBOD"):
        from xgboost import XGBClassifier
        m = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.9,
                          scale_pos_weight=len(Xn) / max(len(Xa), 1), random_state=seed, n_jobs=1, verbosity=0).fit(X, t)
        return lambda Z: m.predict_proba(Z)[:, 1]
    if name == "KNNPOS":
        return lambda Z: -np.min(np.linalg.norm(Z[:, None, :] - Xa[None, :, :], axis=2), 1)
    raise ValueError(name)


def fit_deep(name, Xn, Xa, seed, device):
    X = np.vstack([Xn, Xa]); t = np.r_[np.zeros(len(Xn)), np.ones(len(Xa))]
    if name == "TABPFN":
        from tabpfn import TabPFNClassifier
        m = TabPFNClassifier(device=device, random_state=seed).fit(X, t); return lambda Z: m.predict_proba(Z)[:, 1]
    import deepod.models as dm
    M = {"DEVNET": dm.DevNet, "PRENET": dm.PReNet, "DEEPSAD": dm.DeepSAD, "FEAWAD": dm.FeaWAD, "ROSAS": dm.RoSAS}[name]
    m = M(device=device, random_state=seed, verbose=0); m.fit(X, t); return m.decision_function


# ------------------------------------------------------------------ cells
def cells(phase):
    if phase == "core":
        return [("IID", k, 0.0) for k in (1, 2, 3, 5, 10, 20, 50)]
    if phase == "acq":
        return [("CAUGHT20", None, 0.0), ("CAUGHT50", None, 0.0), ("MISSED", 5, 0.0)]
    if phase == "deep":
        return [("IID", 5, 0.0), ("CAUGHT20", None, 0.0)]
    raise ValueError(phase)


def run_task(args, key, done):
    rng_task = np.random.default_rng(int.from_bytes(key.encode()[:8].ljust(8, b"\0"), "little") % (2 ** 32))
    b = np.load(os.path.join(args.bundles, key + ".npz")); s = np.load(os.path.join(args.scores, key + ".npz"))
    y = s["y_te"].astype(int)
    ss = StandardScaler().fit(b["Xtr"]); Ztr = ss.transform(b["Xtr"]); Zval = ss.transform(b["Xval"])
    Zte = ss.transform(np.vstack([b["Xtn"], b["Xh"]]))
    dets = sorted(k[4:] for k in s.files if k.startswith("te__"))
    if FIX not in dets: return []
    TE = np.column_stack([s["te__" + d] for d in dets]).astype(float); VA = np.column_stack([s["val__" + d] for d in dets]).astype(float)
    iD = dets.index(FIX); VS = np.sort(VA, axis=0)
    picks_free = json.load(open(args.picks)).get(key, {}) if args.picks and os.path.exists(args.picks) else {}
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    Ztr_s = Ztr[rng_task.choice(len(Ztr), min(2000, len(Ztr)), replace=False)]
    out = []
    for r in range(args.reps):
        rr = np.random.default_rng([r, 1009]); pp = rr.permutation(pos); pn = rr.permutation(neg)
        poolA, evalA = pp[: len(pp) // 2], pp[len(pp) // 2:]; poolN, evalN = pn[: len(pn) // 2], pn[len(pn) // 2:]
        vperm = rr.permutation(len(Zval)); vfit, vcal = vperm[: len(vperm) // 2], vperm[len(vperm) // 2:]
        # standardized detector scores (for XGBOD features), fitted on V_fit normals
        mu, sd = VA[vfit].mean(0), VA[vfit].std(0) + 1e-9
        # evaluation subsets per prevalence (shared by all cells/methods in this rep)
        evsets = {"native": [np.r_[evalN, evalA]]}
        for p in args.prev:
            na = int(round(p / (1 - p) * len(evalN)))
            if 2 <= na <= len(evalA):
                evsets[f"p{p}"] = [np.r_[evalN, rr.choice(evalA, na, replace=False)] for _ in range(10)]
        lab_perm = rr.permutation(poolA); lof_rank = poolA[np.argsort(TE[poolA, iD])]   # ascending LOF score
        cand = np.r_[poolA, poolN]; top = cand[np.argsort(-TE[cand, iD])]
        # label-free picks (independent of labels; computed once per rep)
        lf = rank_detectors_from_val(VA[vcal]); lf.update({k: dets.index(v) for k, v in picks_free.items() if v in dets})
        # label-independent quantities, computed ONCE per rep (oracle, default, label-free picks)
        S0, C0 = {"D": TE[:, iD]}, {"D": VA[vcal, iD]}
        for nm, j in lf.items(): S0["LF_" + nm], C0["LF_" + nm] = TE[:, j], VA[vcal, j]
        base = {}
        for pname, subsets in evsets.items():
            acc = {"O": []}
            for ev in subsets:
                ye = y[ev]; acc["O"].append(max(apn(ye, TE[ev, j]) for j in range(len(dets))))
                for nm in S0: acc.setdefault(nm, []).append(_metrics(ye, S0[nm][ev], C0[nm]))
            base[pname] = {"O": float(np.mean(acc.pop("O")))}
            base[pname].update({nm: [round(float(x), 5) for x in np.array(v).mean(0)] for nm, v in acc.items()})
        for (mode, k, cont) in cells(args.phase):
            tag = f"{key}|{r}|{mode}|{k}|{cont}|{args.group}"
            if tag in done: continue
            if mode == "IID": L = lab_perm[:k]
            elif mode.startswith("CAUGHT"): L = np.intersect1d(top[: int(mode[6:])], poolA)
            elif mode == "MISSED": L = rr.choice(lof_rank[: max(k, len(lof_rank) // 2)], k, replace=False)
            if len(L) == 0: continue
            # method scores on the full test set index space (we only read eval indices) and on V_cal
            S, C = {}, {}
            if args.group == "cheap":
                Q = np.empty(len(dets))      # mean fraction of validation normals each labeled anomaly outranks
                for j in range(len(dets)):
                    lo = np.searchsorted(VS[:, j], TE[L, j], "left"); hi = np.searchsorted(VS[:, j], TE[L, j], "right")
                    Q[j] = np.mean((lo + 0.5 * (hi - lo)) / len(VS))
                jsel = iD if Q[iD] >= Q.max() - 1e-12 else int(np.argmax(Q))
                S["SEL"], C["SEL"] = TE[:, jsel], VA[vcal, jsel]
                for nm in ("LR", "RF", "XGB", "KNNPOS"):
                    f = fit_cheap(nm, Ztr_s, Zte[L], r); S[nm], C[nm] = f(Zte), f(Zval[vcal])
                F_te = np.hstack([Zte, (TE - mu) / sd]); F_vf = np.hstack([Zval[vfit], (VA[vfit] - mu) / sd]); F_vc = np.hstack([Zval[vcal], (VA[vcal] - mu) / sd])
                f = fit_cheap("XGBOD", F_vf, F_te[L], r); S["XGBOD"], C["XGBOD"] = f(F_te), f(F_vc)
                learners = ("SEL", "LR", "RF", "XGB", "KNNPOS", "XGBOD")
            else:
                learners = []
                for nm in args.deep_methods:
                    t0 = time.time()
                    try:
                        f = fit_deep(nm, Ztr_s, Zte[L], r, args.device); S[nm], C[nm] = f(Zte), f(Zval[vcal]); learners.append(nm)
                    except Exception as e:
                        print(f"  {key} {nm} failed: {str(e)[:100]}", flush=True)
            for nm in learners:
                S["H_" + nm] = 0.5 * (ecdf(C[nm], S[nm]) + ecdf(C0["D"], S0["D"]))
                C["H_" + nm] = 0.5 * (ecdf(C[nm], C[nm]) + ecdf(C0["D"], C0["D"]))
            res = {}
            for pname, subsets in evsets.items():
                acc = {}
                for ev in subsets:
                    ye = y[ev]
                    for nm in S: acc.setdefault(nm, []).append(_metrics(ye, S[nm][ev], C[nm]))
                res[pname] = dict(base[pname])
                res[pname].update({nm: [round(float(x), 5) for x in np.array(v).mean(0)] for nm, v in acc.items()})
            out.append({"tag": tag, "key": key, "rep": r, "mode": mode, "k": int(len(L)), "cont": cont, "group": args.group, "res": res})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles", required=True); ap.add_argument("--scores", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--phase", default="core"); ap.add_argument("--group", default="cheap")
    ap.add_argument("--reps", type=int, default=30); ap.add_argument("--prev", type=float, nargs="*", default=[0.1, 0.02, 0.005])
    ap.add_argument("--picks", default=None); ap.add_argument("--keys", nargs="*", default=None)
    ap.add_argument("--deep_methods", nargs="*", default=["DEVNET", "PRENET", "DEEPSAD", "FEAWAD", "ROSAS", "TABPFN"])
    ap.add_argument("--device", default="cpu"); ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    keys = args.keys or sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(args.scores, "*.npz")))
    keys = [k for k in keys if os.path.exists(os.path.join(args.bundles, k + ".npz"))][args.shard::args.nshards]
    done = set()
    if os.path.exists(args.out):
        for ln in open(args.out, encoding="utf-8"):
            try: done.add(json.loads(ln)["tag"])
            except Exception: pass
    print(f"{len(keys)} tasks | phase {args.phase} | group {args.group} | reps {args.reps} | already done {len(done)}", flush=True)
    t0 = time.time()
    if args.workers > 1:
        from multiprocessing import Pool
        from functools import partial
        with Pool(args.workers) as p, open(args.out, "a", encoding="utf-8") as fo:
            for i, rows in enumerate(p.imap_unordered(partial(_safe, args, done=done), keys, chunksize=1)):
                for row in rows: fo.write(json.dumps(row) + "\n")
                fo.flush(); print(f"  {i+1}/{len(keys)}  {time.time()-t0:.0f}s", flush=True)
    else:
        with open(args.out, "a", encoding="utf-8") as fo:
            for i, k in enumerate(keys):
                for row in _safe(args, k, done=done): fo.write(json.dumps(row) + "\n")
                fo.flush(); print(f"  {i+1}/{len(keys)}  {time.time()-t0:.0f}s", flush=True)


def _safe(args, key, done):
    try: return run_task(args, key, done)
    except Exception as e:
        print(f"  {key} FAILED: {type(e).__name__}: {str(e)[:150]}", flush=True); return []


if __name__ == "__main__":
    main()
