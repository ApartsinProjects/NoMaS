# -*- coding: utf-8 -*-
"""Test the locality x multimodality hypothesis.

Claim: best detector is LOCAL when normal data is MULTIMODAL, GLOBAL when UNIMODAL; a continuous
scale. If true, a label-free multimodality score on validation-normals predicts the family gap
(best-local ap_norm minus best-global ap_norm) and can route local-vs-global without synthetics.

Multimodality scores (label-free, from val-normals, standardized, PCA<=6):
  m_gmm  BIC-optimal k in 1..4 (higher = more modes)
  m_sil  best silhouette over k=2..4 (clearer clusters = multimodal)
  m_dip  mean Hartigan dip over features (if diptest installed)
Ground-truth family gap from TEST ap_norm CSV.
"""
import os, sys, io, contextlib, zipfile, re, warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import skew, kurtosis
warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"; S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
sys.path.insert(0, os.path.join(S, "scratchpad"))
from dev_common import true_apnorm
from hadb_ts_final import W, STRIDE, MAX_LEN, block_split3
from adrank.ts import _window_features, _window_labels
try:
    from diptest import diptest as _dip; HAVE_DIP = True
except Exception:
    HAVE_DIP = False
LOCAL = ("LOF", "KNN", "CBLOF"); GLOBAL = ("HBOS", "COPOD", "ECOD", "PCA")
fam = lambda v: "local" if str(v).startswith(LOCAL) else ("global" if str(v).startswith(GLOBAL) else "other")
CSVMAP = {"oddbench": "hadb_oddbench.csv", "ovrbench": "hadb_ovrbench.csv", "ucr": "hadb_ts_ucr.csv", "tsbad_u": "hadb_ts_tsbad.csv"}


def find_npz(corpus, name):
    p = os.path.join(ROOT, "data", corpus, name + ".npz")
    return p if os.path.exists(p) else None


def tab_tv(corpus, name):
    d = np.load(find_npz(corpus, name), allow_pickle=True)
    X = np.vstack([np.asarray(d["train"], float), np.asarray(d["test"], float)]); y = np.concatenate([np.asarray(d["train_labels"]).ravel(), np.asarray(d["test_labels"]).ravel()]).astype(int); X = np.nan_to_num(X)
    if len(X) > 5000:
        r = np.random.RandomState(0); k = r.choice(len(X), 5000, replace=False); X, y = X[k], y[k]
    ni = np.where(y == 0)[0]; g = np.random.default_rng(0); idx = np.arange(len(ni)); g.shuffle(idx)
    return X[ni[idx[int(0.6 * len(idx)):int(0.8 * len(idx))]]]


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
    pos = np.arange(len(Xw)); tr, va, te = block_split3(yw, pos, 0); return Xw[va]


def multimodality(Xn):
    if len(Xn) < 20:
        return None
    Z = StandardScaler().fit_transform(Xn)
    if Z.shape[1] > 6:
        Z = PCA(n_components=6, random_state=0).fit_transform(Z)
    n = len(Z); Kmax = min(4, n // 8 + 1)
    # GMM BIC-gain of best-k over k=1 (relative; positive = evidence for >1 mode)
    bics = {}
    for k in range(1, max(2, Kmax) + 1):
        try:
            g = GaussianMixture(k, covariance_type="full", reg_covar=1e-3, random_state=0).fit(Z); bics[k] = g.bic(Z)
        except Exception: pass
    m_gmm = (bics.get(1, 0) - min(bics.values())) / (abs(bics.get(1, 1)) + 1e-9) if bics else 0.0
    # best silhouette over k=2..Kmax
    sil = 0.0
    for k in range(2, max(3, Kmax) + 1):
        try:
            lab = KMeans(k, n_init=5, random_state=0).fit_predict(Z)
            if len(set(lab)) > 1: sil = max(sil, silhouette_score(Z, lab))
        except Exception: pass
    # bimodality coefficient per feature (SAS): (skew^2+1)/(exkurt + 3(n-1)^2/((n-2)(n-3)))
    # unimodal-normal ~0.33, uniform ~0.55, bimodal -> higher. Report mean + frac>0.555.
    corr = 3 * (n - 1) ** 2 / max((n - 2) * (n - 3), 1)
    bcs = []
    for j in range(Z.shape[1]):
        col = Z[:, j]
        if np.std(col) < 1e-9: continue
        g = skew(col); ku = kurtosis(col, fisher=True)
        bcs.append((g * g + 1.0) / (ku + corr))
    bcs = np.array(bcs) if bcs else np.array([np.nan])
    m_bc = float(np.nanmean(bcs)); m_bcfrac = float(np.mean(bcs > 0.555))
    return m_gmm, float(sil), m_bc, m_bcfrac


M = pd.read_csv(os.path.join(S, "HADB_MANIFEST.csv")); inc = M[M.include].copy()
samp = pd.concat([g.sort_values("spread_ap_norm").iloc[np.linspace(0, len(g) - 1, min(16, len(g))).astype(int)]
                  for c, g in inc.groupby("corpus") if c in CSVMAP])
rows = []
for _, r in samp.iterrows():
    corpus = r.dataset and r.corpus
    try:
        if corpus in ("oddbench", "ovrbench"): Xn = tab_tv(corpus, r.dataset)
        elif corpus == "ucr": Xn = uni_tv(r.dataset, "ucr/UCR_TimeSeriesAnomalyDatasets2021.zip")
        elif corpus == "tsbad_u": Xn = uni_tv(r.dataset, "tsbad/TSB-AD-U.zip")
        else: continue
    except Exception: continue
    mm = multimodality(Xn)
    if mm is None: continue
    try: ap, _ = true_apnorm(CSVMAP[corpus], r.dataset)
    except Exception: continue
    loc = [ap[v] for v in ap.index if fam(v) == "local" and ap[v] == ap[v]]
    glo = [ap[v] for v in ap.index if fam(v) == "global" and ap[v] == ap[v]]
    if not loc or not glo: continue
    bl, bg = max(loc), max(glo)
    rows.append({"corpus": corpus, "dataset": r.dataset, "m_gmm": mm[0], "m_sil": mm[1], "m_bc": mm[2], "m_bcfrac": mm[3],
                 "best_local": bl, "best_global": bg, "gap": bl - bg, "local_wins": int(bl > bg), "d": Xn.shape[1], "n": len(Xn)})
df = pd.DataFrame(rows)
print(f"=== locality x multimodality ({len(df)} datasets) ===")
print(f"   local_wins rate: {df.local_wins.mean():.2f}  ({df.local_wins.sum()}/{len(df)})")
for m in ["m_gmm", "m_sil", "m_bc", "m_bcfrac"]:
    v = df[m]
    if v.isna().all(): print(f"   {m:8s}  (unavailable)"); continue
    ok = v.notna()
    rho = spearmanr(v[ok], df.gap[ok]).statistic
    try: auc = roc_auc_score(df.local_wins[ok], v[ok])
    except Exception: auc = np.nan
    print(f"   {m:8s}  Spearman(score, local-global gap) = {rho:+.3f}   AUC(predict local-wins) = {auc:.3f}")
print("   means by winning family:")
for w, g in df.groupby("local_wins"):
    fnm = "local-wins" if w else "global-wins"
    print(f"     {fnm:11s} n={len(g):2d}  m_gmm {g.m_gmm.mean():+.3f}  m_sil {g.m_sil.mean():.3f}  m_bc {g.m_bc.mean():.3f}  m_bcfrac {g.m_bcfrac.mean():.3f}")
df.to_csv(os.path.join(S, "MULTIMODALITY_TEST.csv"), index=False)
print("   saved MULTIMODALITY_TEST.csv  (diptest available:", HAVE_DIP, ")")
