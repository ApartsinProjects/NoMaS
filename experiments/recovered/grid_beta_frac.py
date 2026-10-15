# -*- coding: utf-8 -*-
"""(beta x n_features) 2D synthetic-anomaly grid. beta = per-feature extremeness (typical..tail);
n_features = how many features are perturbed (single-feature marginal .. full multivariate).
  global anomaly  ~ few features + extreme (beta<0, k small)
  local anomaly   ~ many features + typical (beta=1, k large)
Q1 does the plane map the anomaly type (which family separates each cell)?
Q2 does n_features DISCRIMINATE the global family where beta alone could not (within-global rho)?
Q3 best cell for ranking detectors vs the fixed (beta=1, frac=0.4) probe.
"""
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
BETAS = [1.0, 0.0, -4.0]
KMODES = [("k1", 1), ("k2", 2), ("q", 0.25), ("h", 0.5), ("all", 1.0)]   # feature-count modes


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


def gen_bf(Xn, ns, beta, nfeat, seed=0):
    rng = np.random.default_rng(seed); n, d = Xn.shape; H = _H(Xn); out = np.empty((ns, d))
    k = max(1, min(d, int(round(nfeat if nfeat >= 1 else nfeat * d))))
    for r in range(ns):
        base = Xn[rng.integers(n)].copy()
        for j in rng.choice(d, k, replace=False):
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


M = pd.read_csv(os.path.join(S, "HADB_MANIFEST.csv")); inc = M[M.include & M.corpus.isin(CSVMAP)].copy()
samp = pd.concat([g.sort_values("spread_ap_norm").iloc[np.linspace(0, len(g) - 1, min(10, len(g))).astype(int)]
                  for c, g in inc.groupby("corpus")])
cells = [(b, kn, kf) for b in BETAS for (kn, kf) in KMODES]
fam_auc = {(b, kn): {"local": [], "global": []} for b in BETAS for kn, _ in KMODES}   # per-cell family AUC
rho_all = {(b, kn): [] for b in BETAS for kn, _ in KMODES}          # within-dataset rho vs ap (all det)
rho_gl = {(b, kn): [] for b in BETAS for kn, _ in KMODES}           # within-GLOBAL rho on global datasets
for _, r in samp.iterrows():
    try:
        if r.corpus in ("oddbench", "ovrbench"): Xtr, Xval = tab_tv(r.corpus, r.dataset); pool = TAB_POOL
        elif r.corpus == "ucr": Xtr, Xval = uni_tv(r.dataset, "ucr/UCR_TimeSeriesAnomalyDatasets2021.zip"); pool = TS_POOL
        else: Xtr, Xval = uni_tv(r.dataset, "tsbad/TSB-AD-U.zip"); pool = TS_POOL
    except Exception: continue
    if len(Xtr) < 40 or len(Xval) < 30: continue
    try: ap, _ = true_apnorm(CSVMAP[r.corpus], r.dataset)
    except Exception: continue
    mat = {}
    for b in BETAS:
        for kn, kf in KMODES:
            mat[(b, kn)] = sauc(Xtr, Xval, gen_bf(Xval, 150, b, kf), pool)
    common = [v for v, _ in pool if v in ap.index and ap[v] == ap[v] and all(v in mat[c] for c in mat)]
    if len(common) < 6: continue
    gcommon = [v for v in common if fam(v) == "global"]
    av = np.array([ap[v] for v in common]); is_global_ds = None
    loc = [ap[v] for v in common if fam(v) == "local"]; glo = [ap[v] for v in common if fam(v) == "global"]
    is_global_ds = (loc and glo and max(glo) > max(loc))
    for c in mat:
        for v in common:
            f = fam(v)
            if f in fam_auc[c]: fam_auc[c][f].append(mat[c][v])
        s = [mat[c][v] for v in common]
        if np.std(s) > 0: rho_all[c].append(spearmanr(s, av).statistic)
        if is_global_ds and len(gcommon) >= 4:
            sg = [mat[c][v] for v in gcommon]; ag = [ap[v] for v in gcommon]
            if np.std(sg) > 0: rho_gl[c].append(spearmanr(sg, ag).statistic)

print("=== (beta x n_features) map: mean LOCAL-family / GLOBAL-family separation AUC ===")
hdr = "  beta\\k  " + "".join(f"{kn:>12s}" for kn, _ in KMODES); print(hdr)
for b in BETAS:
    line = f"  {b:+.0f}    "
    for kn, _ in KMODES:
        L = np.mean(fam_auc[(b, kn)]["local"]); G = np.mean(fam_auc[(b, kn)]["global"]); line += f"  L{L:.2f}/G{G:.2f}"
    print(line)
print("\n=== within-dataset rho(cell AUC, true ap) - ranking quality (all detectors) ===")
best = max(rho_all, key=lambda c: np.mean(rho_all[c]))
for b in BETAS:
    line = f"  {b:+.0f}    "
    for kn, _ in KMODES:
        m = np.mean(rho_all[(b, kn)]); line += f"  {kn}:{m:+.2f}"
    print(line)
print(f"   BEST cell {best} rho={np.mean(rho_all[best]):+.3f}   vs fixed (beta=1, half) rho={np.mean(rho_all[(1.0,'h')]):+.3f}")
print("\n=== within-GLOBAL-family rho on GLOBAL datasets (does n_features discriminate global?) ===")
bg = max(rho_gl, key=lambda c: (np.mean(rho_gl[c]) if rho_gl[c] else -9))
for b in BETAS:
    line = f"  {b:+.0f}    "
    for kn, _ in KMODES:
        m = np.mean(rho_gl[(b, kn)]) if rho_gl[(b, kn)] else float("nan"); line += f"  {kn}:{m:+.2f}"
    print(line)
print(f"   BEST global-discriminating cell {bg} within-global rho={np.mean(rho_gl[bg]):+.3f}  (n_datasets={len(rho_gl[bg])})")
