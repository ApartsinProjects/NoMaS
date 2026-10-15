# -*- coding: utf-8 -*-
"""Elbow-beta detector characterization. Sweep beta from HARD (typical/multivariate, +2) to EASY
(marginal-extreme, -4). Each detector's separation AUC(beta) rises as the problem gets easier; the
ELBOW is the beta where it crosses from hard(low AUC) to easy(high AUC). Hypothesis: local detectors
elbow at HIGH beta (succeed even on hard multivariate synths), global detectors elbow at LOW beta
(only succeed once anomalies are marginal extremes). Use elbow-beta to characterize detectors."""
import os, sys, io, contextlib, zipfile, re, warnings
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"; S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
sys.path.insert(0, os.path.join(S, "scratchpad"))
from dev_common import TAB_POOL, TS_POOL, true_apnorm
from hadb_ts_final import W, STRIDE, MAX_LEN, block_split3
from adrank.ts import _window_features, _window_labels
LOCAL = ("LOF", "KNN", "CBLOF"); GLOBAL = ("HBOS", "COPOD", "ECOD", "PCA")
fam = lambda v: "local" if str(v).startswith(LOCAL) else ("global" if str(v).startswith(GLOBAL) else "other")
CSVMAP = {"oddbench": "hadb_oddbench.csv", "ovrbench": "hadb_ovrbench.csv", "ucr": "hadb_ts_ucr.csv", "tsbad_u": "hadb_ts_tsbad.csv"}
BETAS = [2.0, 1.0, 0.5, 0.0, -1.0, -2.0, -4.0]        # hard -> easy


def tab_tv(corpus, name):
    d = np.load(os.path.join(ROOT, "data", corpus, name + ".npz"), allow_pickle=True)
    X = np.vstack([np.asarray(d["train"], float), np.asarray(d["test"], float)]); y = np.concatenate([np.asarray(d["train_labels"]).ravel(), np.asarray(d["test_labels"]).ravel()]).astype(int); X = np.nan_to_num(X)
    if len(X) > 5000:
        r = np.random.RandomState(0); k = r.choice(len(X), 5000, replace=False); X, y = X[k], y[k]
    ni = np.where(y == 0)[0]; g = np.random.default_rng(0); idx = np.arange(len(ni)); g.shuffle(idx)
    return X[ni[idx[:int(0.6 * len(idx))]]], X[ni[idx[int(0.6 * len(idx)):int(0.8 * len(idx))]]]


def uni_tv(name, zipname):
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
    pos = np.arange(len(Xw)); tr, va, te = block_split3(yw, pos, 0); return Xw[tr], Xw[va]


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


def sauc(Xtr, Xval, syn, pool):
    Xe = np.vstack([Xval, syn]); ye = np.r_[np.zeros(len(Xval)), np.ones(len(syn))]; o = {}
    for vn, ct in pool:
        try:
            m = ct()
            with contextlib.redirect_stdout(io.StringIO()): m.fit(Xtr)
            s = np.asarray(m.decision_function(Xe), float)
            if np.all(np.isfinite(s)) and np.nanstd(s) > 1e-12: o[vn] = float(roc_auc_score(ye, s))
        except Exception: pass
    return o


def elbow(curve):
    """curve = AUC at BETAS (hard->easy). Elbow beta = where AUC first reaches its midpoint."""
    a = np.array(curve); lo, hi = a.min(), a.max()
    if hi - lo < 0.05: return np.nan                    # flat -> no elbow
    mid = lo + 0.5 * (hi - lo)
    for i in range(1, len(a)):
        if a[i] >= mid:                                  # crossed midpoint between i-1 and i
            b0, b1, a0, a1 = BETAS[i - 1], BETAS[i], a[i - 1], a[i]
            if a1 == a0: return b1
            return b0 + (b1 - b0) * (mid - a0) / (a1 - a0)
    return BETAS[-1]


M = pd.read_csv(os.path.join(S, "HADB_MANIFEST.csv")); inc = M[M.include & M.corpus.isin(CSVMAP)].copy()
samp = pd.concat([g.sort_values("spread_ap_norm").iloc[np.linspace(0, len(g) - 1, min(11, len(g))).astype(int)]
                  for c, g in inc.groupby("corpus")])
det_elbow = {}          # detector -> list of elbow betas
ds_rows = []
for _, r in samp.iterrows():
    try:
        if r.corpus in ("oddbench", "ovrbench"): Xtr, Xval = tab_tv(r.corpus, r.dataset); pool = TAB_POOL
        elif r.corpus == "ucr": Xtr, Xval = uni_tv(r.dataset, "ucr/UCR_TimeSeriesAnomalyDatasets2021.zip"); pool = TS_POOL
        else: Xtr, Xval = uni_tv(r.dataset, "tsbad/TSB-AD-U.zip"); pool = TS_POOL
    except Exception: continue
    if len(Xtr) < 40 or len(Xval) < 30: continue
    mat = {b: sauc(Xtr, Xval, gen_beta(Xval, 150, b), pool) for b in BETAS}
    common = [v for v, _ in pool if all(v in mat[b] for b in BETAS)]
    if len(common) < 6: continue
    for v in common:
        e = elbow([mat[b][v] for b in BETAS])
        if e == e: det_elbow.setdefault(v, []).append(e)
    # dataset-level hard->easy switch: beta where pool-mean AUC crosses 0.7; and max-spread beta
    meanauc = {b: np.mean([mat[b][v] for v in common]) for b in BETAS}
    spread = {b: np.std([mat[b][v] for v in common]) for b in BETAS}
    switch = elbow([meanauc[b] for b in BETAS])          # reuse midpoint-crossing on pool mean
    ds_rows.append({"dataset": r.dataset, "switch_beta": switch, "maxspread_beta": max(spread, key=spread.get)})

print(f"=== elbow-beta detector characterization ({sum(len(v) for v in det_elbow.values())} detector-obs) ===")
print(f"\n1) per-detector median elbow-beta by family (HIGH beta = succeeds on hard/typical synths = local):")
famel = {"local": [], "global": [], "other": []}
for v, es in det_elbow.items():
    if fam(v) in famel: famel[fam(v)] += es
for f in ["local", "global", "other"]:
    a = np.array(famel[f]); print(f"   {f:7s} median elbow-beta = {np.median(a):+.2f}   mean = {np.mean(a):+.2f}   (n={len(a)})")
# does elbow-beta separate local vs global detector-observations?
from sklearn.metrics import roc_auc_score as auc2
yy = np.r_[np.ones(len(famel["local"])), np.zeros(len(famel["global"]))]; xx = np.r_[famel["local"], famel["global"]]
print(f"   AUC(elbow-beta separates local vs global detector-obs) = {auc2(yy, xx):.3f}")
print(f"\n2) per-detector median elbow-beta (sorted, local L / global G / other o):")
for v in sorted(det_elbow, key=lambda k: -np.median(det_elbow[k])):
    if len(det_elbow[v]) >= 6: print(f"   {v[:16]:16s} {fam(v)[0].upper()}  elbow {np.median(det_elbow[v]):+.2f}  (n={len(det_elbow[v])})")
dd = pd.DataFrame(ds_rows)
print(f"\n3) dataset hard->easy SWITCH beta: median {dd.switch_beta.median():+.2f}   "
      f"vs max-spread beta median {dd.maxspread_beta.median():+.2f}   corr={spearmanr(dd.switch_beta, dd.maxspread_beta).statistic:+.2f}")
pd.DataFrame([(v, fam(v), np.median(e), len(e)) for v, e in det_elbow.items()], columns=["det", "fam", "elbow", "n"]).to_csv(os.path.join(S, "ELBOW.csv"), index=False)
print("\nsaved ELBOW.csv")
