# -*- coding: utf-8 -*-
"""PILOT: can a detector's beta-response curve (separation AUC vs generation beta) rank models?

Density-reweighted resampling at exponent beta parameterises the synthetic from TYPICAL
(beta>=1, sample ~ density -> multivariate/shuffle-like) to EXTREME (beta<0, sample ~ 1/density
-> single-feature marginal). For a few datasets per family, sweep beta and per detector compute
synth-separation AUC. Questions:
  Q1  Does the BEST beta (highest within-dataset Spearman of synth-AUC vs true ap_norm) differ by
      family? (local -> typical/high beta; global -> extreme/low beta?)
  Q2  Does each DETECTOR have a characteristic best-separation beta that clusters by family?
  Q3  Oracle-beta: if we knew the best beta per dataset, how much does selection improve?
"""
import os, sys, io, contextlib, zipfile, re, warnings
import numpy as np
import pandas as pd
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
BETAS = [2.0, 1.0, 0.5, 0.0, -1.0, -2.0, -4.0]


def find_npz(corpus, name):
    for sub in {"oddbench": ["oddbench"], "ovrbench": ["ovrbench"]}.get(corpus, [corpus]):
        p = os.path.join(ROOT, "data", sub, name + ".npz")
        if os.path.exists(p):
            return p
    return None


def tab_tv(corpus, name):
    d = np.load(find_npz(corpus, name), allow_pickle=True)
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
    st = np.arange(0, len(x) - W + 1, STRIDE); Xw, _ = _window_features(x, w=W, stride=STRIDE); yw = _window_labels(lab, st, w=W, min_count=1); Xw = np.nan_to_num(Xw)
    pos = np.arange(len(Xw)); tr, va, te = block_split3(yw, pos, 0); return Xw[tr], Xw[va]


def gen_beta(Xn, ns, beta, frac=0.4, cap=98, seed=0):
    rng = np.random.default_rng(seed); n, d = Xn.shape
    H = []
    for j in range(d):
        col = Xn[:, j]; cnt, edges = np.histogram(col, bins=30); dens = cnt / max(cnt.sum(), 1)
        lo, hi = np.percentile(col, 100 - cap), np.percentile(col, cap); ctr = (edges[:-1] + edges[1:]) / 2
        allowed = (cnt > 0) & (ctr >= lo) & (ctr <= hi); w = np.where(allowed, np.maximum(dens, 1e-6) ** beta, 0.0)
        H.append((edges, w / w.sum() if w.sum() > 0 else None))
    out = np.empty((ns, d))
    for r in range(ns):
        base = Xn[rng.integers(n)].copy(); cols = rng.choice(d, max(1, int(frac * d)), replace=False)
        for j in cols:
            edges, w = H[j]
            if w is None:
                base[j] = Xn[rng.integers(n), j]; continue
            b = rng.choice(len(w), p=w); base[j] = rng.uniform(edges[b], edges[b + 1])
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


# curated sample: a few per (modality x family)
M = pd.read_csv(os.path.join(S, "HADB_MANIFEST.csv")); inc = M[M.include].copy(); inc["f"] = inc.gt_best.map(fam)
picks = []
for (corp, f), n in [(("oddbench", "local"), 3), (("ovrbench", "local"), 2), (("oddbench", "global"), 3),
                     (("ovrbench", "global"), 2), (("ucr", "local"), 3), (("tsbad_u", "local"), 2)]:
    sub = inc[(inc.corpus == corp) & (inc.f == f)].sort_values("spread_ap_norm", ascending=False).head(n)
    picks.append(sub)
SAMP = pd.concat(picks)

perbeta_rho = {b: [] for b in BETAS}; oracle = []; fixed1 = []; det_bestbeta = {}
for _, r in SAMP.iterrows():
    corpus = r.corpus
    try:
        if corpus in ("oddbench", "ovrbench"): Xtr, Xval = tab_tv(corpus, r.dataset); pool = TAB_POOL
        elif corpus == "ucr": Xtr, Xval = uni_tv(r.dataset, "ucr/UCR_TimeSeriesAnomalyDatasets2021.zip"); pool = TS_POOL
        elif corpus == "tsbad_u": Xtr, Xval = uni_tv(r.dataset, "tsbad/TSB-AD-U.zip"); pool = TS_POOL
        else: continue
    except Exception: continue
    if len(Xtr) < 40 or len(Xval) < 30: continue
    try: ap, _ = true_apnorm(CSVMAP[corpus], r.dataset)
    except Exception: continue
    # beta x detector AUC matrix
    mat = {}
    for b in BETAS:
        mat[b] = sauc(Xtr, Xval, gen_beta(Xval, 150, b), pool)
    common = [v for v, _ in pool if v in ap.index and all(v in mat[b] for b in BETAS)]
    if len(common) < 6: continue
    av = pd.Series({v: ap[v] for v in common}); best = av.max()
    rho_b = {}
    for b in BETAS:
        sv = np.array([mat[b][v] for v in common])
        if np.std(sv) > 0 and av.std() > 0:
            rho_b[b] = spearmanr(sv, av[common]).statistic; perbeta_rho[b].append(rho_b[b])
    # oracle-beta: pick beta with best correlation, then its argmax detector
    if rho_b:
        b_star = max(rho_b, key=rho_b.get); sv = pd.Series({v: mat[b_star][v] for v in common}); oracle.append(best - av[sv.idxmax()])
        sv1 = pd.Series({v: mat[1.0][v] for v in common}); fixed1.append(best - av[sv1.idxmax()])
    # per-detector best-separation beta (which beta each detector separates best at)
    for v in common:
        bb = max(BETAS, key=lambda b: mat[b][v]); det_bestbeta.setdefault(fam(v), []).append(bb)
    print(f"  {corpus:9s} {r.dataset[:26]:28s} best-rho beta={max(rho_b,key=rho_b.get):+.1f} (rho {max(rho_b.values()):+.2f})  true-fam {r.f}")

print(f"\n=== Q1: within-dataset rho by generation beta (mean over {sum(len(v) for v in perbeta_rho.values())//len(BETAS)} datasets) ===")
for b in BETAS:
    a = np.array(perbeta_rho[b]); print(f"   beta={b:+.1f}  mean rho {np.nanmean(a):+.3f}")
print(f"\n=== Q2: per-detector best-separation beta by family ===")
for f in ["local", "global", "other"]:
    a = np.array(det_bestbeta.get(f, [])); print(f"   {f:7s} median best-beta {np.median(a):+.1f}  (n={len(a)})" if len(a) else f"   {f}: none")
print(f"\n=== Q3: oracle-beta vs fixed beta=1 selection ===")
print(f"   oracle-beta regret {np.nanmean(oracle):.4f}   fixed beta=1 regret {np.nanmean(fixed1):.4f}   ({len(oracle)} datasets)")
