# -*- coding: utf-8 -*-
"""Per-dataset table: independent modality metrics + detector-family perf + ranking-method regret.
Feeds the two-panel figure (detectors vs modality; ranking methods vs modality)."""
import os, sys, io, contextlib, zipfile, re, warnings
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import skew, kurtosis
warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"; S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
sys.path.insert(0, os.path.join(S, "scratchpad"))
from dev_common import TAB_POOL, TS_POOL, true_apnorm
from hadb_ts_final import W, STRIDE, MAX_LEN, block_split3
from adrank.ts import _window_features, _window_labels
LOCAL = ("LOF", "KNN", "CBLOF"); GLOBAL = ("HBOS", "COPOD", "ECOD", "PCA")
fam = lambda v: "local" if str(v).startswith(LOCAL) else ("global" if str(v).startswith(GLOBAL) else "other")
CSVMAP = {"oddbench": "hadb_oddbench.csv", "ovrbench": "hadb_ovrbench.csv", "ucr": "hadb_ts_ucr.csv", "tsbad_u": "hadb_ts_tsbad.csv"}
TAU = -0.140


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


def modality(Xn):
    if len(Xn) < 20: return np.nan, np.nan, np.nan
    Z = StandardScaler().fit_transform(Xn)
    if Z.shape[1] > 6: Z = PCA(6, random_state=0).fit_transform(Z)
    n = len(Z); Kmax = min(4, n // 8 + 1)
    bics = {}
    for k in range(1, max(2, Kmax) + 1):
        try: bics[k] = GaussianMixture(k, reg_covar=1e-3, random_state=0).fit(Z).bic(Z)
        except Exception: pass
    m_gmm = (bics.get(1, 0) - min(bics.values())) / (abs(bics.get(1, 1)) + 1e-9) if bics else 0.0
    sil = 0.0
    for k in range(2, max(3, Kmax) + 1):
        try:
            lab = KMeans(k, n_init=5, random_state=0).fit_predict(Z)
            if len(set(lab)) > 1: sil = max(sil, silhouette_score(Z, lab))
        except Exception: pass
    corr = 3 * (n - 1) ** 2 / max((n - 2) * (n - 3), 1); bcs = []
    for j in range(Z.shape[1]):
        c = Z[:, j]
        if np.std(c) > 1e-9: bcs.append((skew(c) ** 2 + 1) / (kurtosis(c, fisher=True) + corr))
    m_bc = float(np.mean(bcs)) if bcs else np.nan
    return m_gmm, float(sil), m_bc


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


M = pd.read_csv(os.path.join(S, "HADB_MANIFEST.csv")); inc = M[M.include & M.corpus.isin(CSVMAP)].copy()
rows = []
for i, (_, r) in enumerate(inc.iterrows()):
    corpus = r.corpus
    try:
        if corpus in ("oddbench", "ovrbench"): Xtr, Xval = tab_tv(corpus, r.dataset); pool = TAB_POOL
        elif corpus == "ucr": Xtr, Xval = uni_tv(r.dataset, "ucr/UCR_TimeSeriesAnomalyDatasets2021.zip"); pool = TS_POOL
        else: Xtr, Xval = uni_tv(r.dataset, "tsbad/TSB-AD-U.zip"); pool = TS_POOL
    except Exception: continue
    if len(Xtr) < 40 or len(Xval) < 30: continue
    try: ap, emv = true_apnorm(CSVMAP[corpus], r.dataset)
    except Exception: continue
    m_gmm, m_sil, m_bc = modality(Xval)
    a1 = sauc(Xtr, Xval, gen_beta(Xval, 200, 1.0), pool); a4 = sauc(Xtr, Xval, gen_beta(Xval, 200, -4.0), pool)
    common = [v for v, _ in pool if v in a1 and v in a4 and v in ap.index and ap[v] == ap[v] and v in emv.index and emv[v] == emv[v]]
    if len(common) < 6: continue
    av = pd.Series({v: ap[v] for v in common}); best = av.max()
    loc = [av[v] for v in common if fam(v) == "local"]; glo = [av[v] for v in common if fam(v) == "global"]
    if not loc or not glo: continue
    p1 = max(common, key=lambda v: a1[v]); pe = max(common, key=lambda v: a4[v]); pem = max(common, key=lambda v: emv[v])
    local_ev = max(a1.values()) - max(a4.values())
    reg_matched = (best - av[p1]) if local_ev > TAU else (best - av[pe])
    rows.append({"corpus": corpus, "dataset": r.dataset, "d": Xtr.shape[1], "m_gmm": m_gmm, "m_sil": m_sil, "m_bc": m_bc,
                 "local_ev": local_ev, "best_local": max(loc), "best_global": max(glo),
                 "mean_local": float(np.mean(loc)), "mean_global": float(np.mean(glo)), "best_overall": best,
                 "reg_em": best - av[pem], "reg_beta1": best - av[p1], "reg_betaext": best - av[pe],
                 "reg_matched": reg_matched, "reg_random": best - av.mean(),
                 "truefam": "local" if max(loc) > max(glo) else "global"})
    if i % 20 == 0: print(f"  ..{i}/{len(inc)}  {len(rows)} kept", flush=True)
df = pd.DataFrame(rows); df.to_csv(os.path.join(S, "MODALITY_PERF.csv"), index=False)
print(f"kept {len(df)} datasets  (local {int((df.truefam=='local').sum())} / global {int((df.truefam=='global').sum())})")
from scipy.stats import spearmanr
for m in ["m_sil", "m_gmm", "m_bc", "local_ev"]:
    ok = df[m].notna()
    print(f"  {m:8s} Spearman with (best_local-best_global) = {spearmanr(df[m][ok], (df.best_local-df.best_global)[ok]).statistic:+.3f}")
print("saved MODALITY_PERF.csv")
