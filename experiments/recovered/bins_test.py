# -*- coding: utf-8 -*-
"""Does finer histogram BIN resolution make marginal synthetics that DISCRIMINATE the global family?
Fine bins let beta<0 sample INTERIOR low-density gaps (rare by histogram, not by tail-CDF) that HBOS
(histogram) and COPOD/ECOD (CDF/tail) score differently. Coarse bins only see the tails.
Test on GLOBAL datasets: within-global-family ranking rho vs true ap, per (beta, n_bins)."""
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
BETAS = [0.0, -4.0]; NBINS = [15, 30, 60, 120]


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


def gen_bins(Xn, ns, beta, nbins, frac=0.4, cap=98, seed=0):
    rng = np.random.default_rng(seed); n, d = Xn.shape; H = []
    for j in range(d):
        col = Xn[:, j]; cnt, edges = np.histogram(col, bins=nbins); dens = cnt / max(cnt.sum(), 1)
        lo, hi = np.percentile(col, 100 - cap), np.percentile(col, cap); ctr = (edges[:-1] + edges[1:]) / 2
        H.append((edges, dens, (cnt > 0) & (ctr >= lo) & (ctr <= hi)))
    out = np.empty((ns, d)); interior = 0; total = 0
    mu = Xn.mean(0); sd = Xn.std(0) + 1e-9
    for r in range(ns):
        base = Xn[rng.integers(n)].copy()
        for j in rng.choice(d, max(1, int(frac * d)), replace=False):
            edges, dens, allowed = H[j]; w = np.where(allowed, np.maximum(dens, 1e-6) ** beta, 0.0)
            if w.sum() == 0: base[j] = Xn[rng.integers(n), j]
            else:
                b = rng.choice(len(w), p=w / w.sum()); base[j] = rng.uniform(edges[b], edges[b + 1])
            total += 1; interior += int(abs((base[j] - mu[j]) / sd[j]) < 2.0)   # in interior, not tail
        out[r] = base
    return out, (interior / max(total, 1))


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


mp = pd.read_csv(os.path.join(S, "MODALITY_PERF.csv")); glob = mp[mp.best_global > mp.best_local]
samp = pd.concat([g.iloc[np.linspace(0, len(g) - 1, min(8, len(g))).astype(int)] for c, g in glob.groupby("corpus")])
rho_gl = {(b, nb): [] for b in BETAS for nb in NBINS}
rho_all = {(b, nb): [] for b in BETAS for nb in NBINS}
interior = {(b, nb): [] for b in BETAS for nb in NBINS}
for _, r in samp.iterrows():
    try:
        if r.corpus in ("oddbench", "ovrbench"): Xtr, Xval = tab_tv(r.corpus, r.dataset); pool = TAB_POOL
        elif r.corpus == "ucr": Xtr, Xval = uni_tv(r.dataset, "ucr/UCR_TimeSeriesAnomalyDatasets2021.zip"); pool = TS_POOL
        else: Xtr, Xval = uni_tv(r.dataset, "tsbad/TSB-AD-U.zip"); pool = TS_POOL
    except Exception: continue
    if len(Xtr) < 40 or len(Xval) < 30: continue
    try: ap, _ = true_apnorm(CSVMAP[r.corpus], r.dataset)
    except Exception: continue
    for b in BETAS:
        for nb in NBINS:
            syn, intfrac = gen_bins(Xval, 150, b, nb); a = sauc(Xtr, Xval, syn, pool)
            gc = [v for v in a if v in ap.index and ap[v] == ap[v] and fam(v) == "global"]
            allc = [v for v in a if v in ap.index and ap[v] == ap[v]]
            if len(gc) >= 4 and np.std([a[v] for v in gc]) > 0:
                rho_gl[(b, nb)].append(spearmanr([a[v] for v in gc], [ap[v] for v in gc]).statistic)
            if len(allc) >= 6 and np.std([a[v] for v in allc]) > 0:
                rho_all[(b, nb)].append(spearmanr([a[v] for v in allc], [ap[v] for v in allc]).statistic)
            interior[(b, nb)].append(intfrac)
print(f"=== bins sweep on GLOBAL datasets (n={len(samp)}) ===")
print("within-GLOBAL-family rho (does it rank global detectors?) [baseline ~0.04-0.12]:")
print("  beta\\bins " + "".join(f"{nb:>10d}" for nb in NBINS))
for b in BETAS:
    print(f"  {b:+.0f}      " + "".join(f"  {np.mean(rho_gl[(b,nb)]):+.3f}   " for nb in NBINS))
print("\nwithin-dataset rho over ALL detectors:")
for b in BETAS:
    print(f"  {b:+.0f}      " + "".join(f"  {np.mean(rho_all[(b,nb)]):+.3f}   " for nb in NBINS))
print("\ninterior-gap fraction of synth values (|z|<2, i.e. NOT tail - higher = more interior-gap targeting):")
for b in BETAS:
    print(f"  {b:+.0f}      " + "".join(f"  {np.mean(interior[(b,nb)]):.2f}    " for nb in NBINS))
