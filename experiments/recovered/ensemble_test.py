# -*- coding: utf-8 -*-
"""Add 3 ensembles (local-biased / global-biased / neutral) to the pool and rank them too.
Everything construct-matched in ONE pass: fit on TRAIN, ap_norm on TEST, synth-rank on VAL.

Questions:
  1. Fixed-choice: is 'always neutral ensemble' a low-regret robust baseline vs EM/beta1/best-individual?
  2. Does adding ensembles to the candidate pool LOWER the achievable oracle (are they ever best)?
  3. Does the beta=1 selector, with ensembles in the pool, pick them and cut regret?
"""
import os, sys, io, contextlib, zipfile, re, warnings
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"; S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
sys.path.insert(0, os.path.join(S, "scratchpad"))
from dev_common import TAB_POOL, TS_POOL
from hadb_ts_final import W, STRIDE, MAX_LEN, block_split3
from adrank.ts import _window_features, _window_labels
LOCAL = ("LOF", "KNN", "CBLOF"); GLOBAL = ("HBOS", "COPOD", "ECOD", "PCA")
fam = lambda v: "local" if str(v).startswith(LOCAL) else ("global" if str(v).startswith(GLOBAL) else "other")
CSVMAP = {"oddbench": "hadb_oddbench.csv", "ovrbench": "hadb_ovrbench.csv", "ucr": "hadb_ts_ucr.csv", "tsbad_u": "hadb_ts_tsbad.csv"}


def tab_load(corpus, name):
    d = np.load(os.path.join(ROOT, "data", corpus, name + ".npz"), allow_pickle=True)
    X = np.vstack([np.asarray(d["train"], float), np.asarray(d["test"], float)]); y = np.concatenate([np.asarray(d["train_labels"]).ravel(), np.asarray(d["test_labels"]).ravel()]).astype(int); X = np.nan_to_num(X)
    r = np.random.default_rng(0); ni = np.where(y == 0)[0]; ai = np.where(y == 1)[0]
    if len(ni) > 5000: ni = r.choice(ni, 5000, replace=False)
    if len(ai) > 400: ai = r.choice(ai, 400, replace=False)
    idx = np.arange(len(ni)); r.shuffle(idx); ni = ni[idx]
    Xtr = X[ni[:int(0.6 * len(ni))]]; Xval = X[ni[int(0.6 * len(ni)):int(0.8 * len(ni))]]; Xten = X[ni[int(0.8 * len(ni)):]]
    Xte = np.vstack([Xten, X[ai]]); yte = np.r_[np.zeros(len(Xten)), np.ones(len(ai))]
    return Xtr, Xval, Xte, yte


def uni_load(name, zipname):
    z = zipfile.ZipFile(os.path.join(ROOT, "data", zipname))
    cand = [n for n in z.namelist() if os.path.basename(n) == name] or [n for n in z.namelist() if n.lower().endswith((".txt", ".csv")) and name.split("_")[0] + "_" in os.path.basename(n)]
    fn = cand[0]
    if fn.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(z.read(fn))); x = np.nan_to_num(df.iloc[:, 0].to_numpy(float)); lab = df.iloc[:, 1].to_numpy(int)
    else:
        m = re.search(r"_(\d+)_(\d+)_(\d+)\.txt$", fn); x = np.array([float(v) for v in z.read(fn).decode("utf-8", "replace").split() if v.strip()]); a0, a1 = int(m.group(2)), int(m.group(3)); lab = np.zeros(len(x), int); lab[a0:min(a1 + 1, len(x))] = 1
    if len(x) > MAX_LEN:
        a = np.where(lab == 1)[0]; c = int((a[0] + a[-1]) // 2) if len(a) else MAX_LEN // 2; lo = max(0, min(c - MAX_LEN // 2, len(x) - MAX_LEN)); x, lab = x[lo:lo + MAX_LEN], lab[lo:lo + MAX_LEN]
    Xw, _ = _window_features(x, w=W, stride=STRIDE); st = np.arange(0, len(x) - W + 1, STRIDE); yw = _window_labels(lab, st, w=W, min_count=1); Xw = np.nan_to_num(Xw)
    pos = np.arange(len(Xw)); tr, va, te = block_split3(yw, pos, 0)
    Xtr = Xw[tr][yw[tr] == 0]; Xval = Xw[va][yw[va] == 0]
    Xten = Xw[te][yw[te] == 0]; Xan = Xw[yw == 1]
    if len(Xten) > 2000: Xten = Xten[:2000]
    if len(Xan) > 400: Xan = Xan[:400]
    Xte = np.vstack([Xten, Xan]); yte = np.r_[np.zeros(len(Xten)), np.ones(len(Xan))]
    return Xtr, Xval, Xte, yte


def _H(Xn, cap=98):
    H = []
    for j in range(Xn.shape[1]):
        col = Xn[:, j]; cnt, edges = np.histogram(col, bins=30); dens = cnt / max(cnt.sum(), 1)
        lo, hi = np.percentile(col, 100 - cap), np.percentile(col, cap); ctr = (edges[:-1] + edges[1:]) / 2
        H.append((edges, dens, (cnt > 0) & (ctr >= lo) & (ctr <= hi)))
    return H


def gen_beta(Xn, ns, beta, frac=0.4, seed=0):
    rng = np.random.default_rng(seed); n, d = Xn.shape; H = _H(Xn); out = np.empty((ns, d))
    for r in range(ns):
        base = Xn[rng.integers(n)].copy()
        for j in rng.choice(d, max(1, int(frac * d)), replace=False):
            edges, dens, allowed = H[j]; w = np.where(allowed, np.maximum(dens, 1e-6) ** beta, 0.0)
            base[j] = Xn[rng.integers(n), j] if w.sum() == 0 else rng.uniform(*edges[[(b := rng.choice(len(w), p=w / w.sum())), b + 1]])
        out[r] = base
    return out


def apn(y, s):
    b = y.mean()
    return (average_precision_score(y, s) - b) / (1 - b + 1e-12)


M = pd.read_csv(os.path.join(S, "HADB_MANIFEST.csv")); inc = M[M.include & M.corpus.isin(CSVMAP)].copy()
samp = pd.concat([g.sort_values("spread_ap_norm").iloc[np.linspace(0, len(g) - 1, min(11, len(g))).astype(int)]
                  for c, g in inc.groupby("corpus")])
rows = []
for _, r in samp.iterrows():
    try:
        if r.corpus in ("oddbench", "ovrbench"): Xtr, Xval, Xte, yte = tab_load(r.corpus, r.dataset); pool = TAB_POOL
        elif r.corpus == "ucr": Xtr, Xval, Xte, yte = uni_load(r.dataset, "ucr/UCR_TimeSeriesAnomalyDatasets2021.zip"); pool = TS_POOL
        else: Xtr, Xval, Xte, yte = uni_load(r.dataset, "tsbad/TSB-AD-U.zip"); pool = TS_POOL
    except Exception: continue
    if len(Xtr) < 40 or len(Xval) < 20 or yte.sum() < 3 or (yte == 0).sum() < 3: continue
    syn1 = gen_beta(Xval, 200, 1.0)
    # per-detector: z-normalized scores on val, test, synth  (z via val-normal stats)
    Zval, Zte, Zsy, names = {}, {}, {}, []
    for vn, ct in pool:
        try:
            m = ct()
            with contextlib.redirect_stdout(io.StringIO()): m.fit(Xtr)
            sv = np.asarray(m.decision_function(Xval), float); st = np.asarray(m.decision_function(Xte), float); ss = np.asarray(m.decision_function(syn1), float)
            mu, sd = sv.mean(), sv.std() + 1e-9
            if np.all(np.isfinite(sv)) and np.all(np.isfinite(st)) and sd > 1e-9:
                Zval[vn] = (sv - mu) / sd; Zte[vn] = (st - mu) / sd; Zsy[vn] = (ss - mu) / sd; names.append(vn)
        except Exception: pass
    if len(names) < 8: continue
    locs = [v for v in names if fam(v) == "local"]; globs = [v for v in names if fam(v) == "global"]
    if not locs or not globs: continue
    # individual ap_norm (test) and synth separation a1
    ap = {v: apn(yte, Zte[v]) for v in names}
    a1 = {v: roc_auc_score(np.r_[np.zeros(len(Xval)), np.ones(len(syn1))], np.r_[Zval[v], Zsy[v]]) for v in names}
    # ensembles
    def comb(members, how):
        f = (lambda A: np.mean(A, 0)) if how == "mean" else (lambda A: np.max(A, 0))
        return f([Zte[v] for v in members]), f([Zval[v] for v in members]), f([Zsy[v] for v in members])
    def ens(members, tag, how="mean"):
        te, va, sy = comb(members, how)
        ap[tag] = apn(yte, te)
        a1[tag] = roc_auc_score(np.r_[np.zeros(len(Xval)), np.ones(len(syn1))], np.r_[va, sy])
    ens(locs, "ENS_local"); ens(globs, "ENS_global"); ens(names, "ENS_neutral")
    ens(names, "ENS_max", "max")                       # union: max z over all detectors
    ens(locs + globs, "ENS_maxLG", "max")              # max over local+global only
    ENS = ["ENS_local", "ENS_global", "ENS_neutral", "ENS_max", "ENS_maxLG"]
    indiv = names
    best_ind = max(ap[v] for v in indiv); best_all = max(ap.values())
    p1_ind = max(indiv, key=lambda v: a1[v]); p1_all = max(indiv + ENS, key=lambda v: a1[v])
    rows.append({"dataset": r.dataset, "corpus": r.corpus, "best_ind": best_ind, "best_all": best_all,
                 "ap_ENSneutral": ap["ENS_neutral"], "ap_ENSlocal": ap["ENS_local"], "ap_ENSglobal": ap["ENS_global"],
                 "ap_ENSmax": ap["ENS_max"], "ap_ENSmaxLG": ap["ENS_maxLG"],
                 "reg_beta1_noens": best_all - ap[p1_ind], "reg_beta1_withens": best_all - ap[p1_all],
                 "reg_fixed_neutral": best_all - ap["ENS_neutral"], "reg_fixed_max": best_all - ap["ENS_max"],
                 "reg_oracle_ind": best_all - best_ind,
                 "ens_is_best": int(max(ENS, key=lambda v: ap[v]) == max(ap, key=ap.get)),
                 "pick_withens_is_ens": int(p1_all in ENS)})
df = pd.DataFrame(rows); df.to_csv(os.path.join(S, "ENSEMBLE_TEST.csv"), index=False)
print(f"=== ensembles in the pool ({len(df)} datasets, smoke test) ===")
print(f"\n1) FIXED-choice mean regret (vs oracle over full pool incl. ensembles):")
print(f"   always ENS_neutral : {df.reg_fixed_neutral.mean():.4f}")
print(f"   always ENS_max     : {df.reg_fixed_max.mean():.4f}")
print(f"   beta1 (no ens)     : {df.reg_beta1_noens.mean():.4f}")
print(f"   beta1 (with ens)   : {df.reg_beta1_withens.mean():.4f}")
print(f"   oracle-individuals : {df.reg_oracle_ind.mean():.4f}  (how much ensembles add on top of best individual)")
print(f"\n2) mean ap_norm achieved:  ENS_neutral {df.ap_ENSneutral.mean():.3f}  ENS_local {df.ap_ENSlocal.mean():.3f}  "
      f"ENS_global {df.ap_ENSglobal.mean():.3f}  ENS_max {df.ap_ENSmax.mean():.3f}  ENS_maxLG {df.ap_ENSmaxLG.mean():.3f}  best_individual {df.best_ind.mean():.3f}")
print(f"\n3) an ensemble is the single best in the full pool on {df.ens_is_best.mean()*100:.0f}% of datasets")
print(f"   beta1-with-ens actually picks an ensemble on {df.pick_withens_is_ens.mean()*100:.0f}% of datasets")
print("\nsaved ENSEMBLE_TEST.csv")
