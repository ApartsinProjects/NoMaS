# -*- coding: utf-8 -*-
"""Does NORMAL SHAPE (train, label-free) predict ANOMALY LOCATION (edge/marginal vs interstitial/local),
and does anomaly location predict the winning detector family? The causal chain behind the thesis.

Per dataset:
  train modality (TRAIN normals): m_sil_tr, m_gmm_tr
  anomaly geometry (TEST anomalies vs TRAIN normals):
    marg  = median over anomalies of max|z|            (edge / marginal extremeness -> global)
    iso   = median [kNN10 dist to train] / [within-train kNN10]   (local isolation -> in a void)
    inter = frac anomalies with max|z|<2 (not marginal) = must be caught by local structure
  best family (TEST ap_norm CSV): best_local, best_global
Chain: m_sil_tr -> (marg, inter) -> (best_global-best_local).
"""
import os, sys, io, zipfile, re, warnings
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"; S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
sys.path.insert(0, os.path.join(S, "scratchpad"))
from dev_common import true_apnorm
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
    idx = np.arange(len(ni)); r.shuffle(idx); Xtr = X[ni[idx[:int(0.6 * len(idx))]]]
    if len(ai) > 400: ai = r.choice(ai, 400, replace=False)
    return Xtr, X[ai]


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
    return Xw[tr], Xw[(yw == 1)]


def modality(Xn):
    Z = StandardScaler().fit_transform(Xn)
    if Z.shape[1] > 6: Z = PCA(6, random_state=0).fit_transform(Z)
    n = len(Z); Kmax = min(4, n // 8 + 1); sil = 0.0
    for k in range(2, max(3, Kmax) + 1):
        try:
            lab = KMeans(k, n_init=5, random_state=0).fit_predict(Z)
            if len(set(lab)) > 1: sil = max(sil, silhouette_score(Z, lab))
        except Exception: pass
    bics = {}
    for k in range(1, max(2, Kmax) + 1):
        try: bics[k] = GaussianMixture(k, reg_covar=1e-3, random_state=0).fit(Z).bic(Z)
        except Exception: pass
    gmm = (bics.get(1, 0) - min(bics.values())) / (abs(bics.get(1, 1)) + 1e-9) if bics else 0.0
    return sil, gmm


mp = pd.read_csv(os.path.join(S, "MODALITY_PERF.csv")); rows = []
for _, r in mp.iterrows():
    try:
        if r.corpus in ("oddbench", "ovrbench"): Xtr, Xan = tab_load(r.corpus, r.dataset)
        elif r.corpus == "ucr": Xtr, Xan = uni_load(r.dataset, "ucr/UCR_TimeSeriesAnomalyDatasets2021.zip")
        else: Xtr, Xan = uni_load(r.dataset, "tsbad/TSB-AD-U.zip")
    except Exception: continue
    if len(Xtr) < 40 or len(Xan) < 5: continue
    Xtr = np.unique(Xtr, axis=0)                      # dedup: near-duplicate rows corrupt the kNN baseline
    if len(Xtr) < 40: continue
    sil_tr, gmm_tr = modality(Xtr)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9; Za = np.abs((Xan - mu) / sd); maxz_a = Za.max(1)
    maxz_tr = np.abs((Xtr - mu) / sd).max(1)          # train normals' own max|z| (same dimensionality)
    marg_rel = float(np.median(maxz_a) / (np.median(maxz_tr) + 1e-9))   # dimensionality-cancelling
    # local isolation: anomaly kNN dist vs within-train baseline, both on deduped train (standardized)
    Zt = (Xtr - mu) / sd; k = min(10, len(Xtr) - 1)
    nn = NearestNeighbors(n_neighbors=k).fit(Zt)
    base = max(np.median(nn.kneighbors(Zt)[0][:, -1]), np.percentile(nn.kneighbors(Zt)[0][:, -1], 25), 1e-6)
    iso = float(min(np.median(nn.kneighbors((Xan - mu) / sd)[0][:, -1]) / base, 50.0))   # winsorize
    rows.append({"corpus": r.corpus, "dataset": r.dataset, "m_sil_tr": sil_tr, "m_gmm_tr": gmm_tr, "m_sil_val": r.m_sil,
                 "marg": float(np.median(maxz_a)), "marg_rel": marg_rel, "iso": iso, "inter": float(np.mean(maxz_a < 2.0)),
                 "best_local": r.best_local, "best_global": r.best_global, "glob_adv": r.best_global - r.best_local})
df = pd.DataFrame(rows); df.to_csv(os.path.join(S, "SHAPE_ANOMALY.csv"), index=False)
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score
def sp(a, b): return spearmanr(df[a], df[b]).statistic
df["local_wins"] = (df.glob_adv < 0).astype(int)
print(f"=== normal shape (TRAIN) -> anomaly location -> family  ({len(df)} datasets) ===")
print(f"\n0) does measuring modality on TRAIN sharpen the family signal vs VAL?")
print(f"   Spearman(m_sil_val, local_adv) = {-sp('m_sil_val','glob_adv'):+.3f}   (val, was +0.13)")
print(f"   Spearman(m_sil_tr,  local_adv) = {-sp('m_sil_tr','glob_adv'):+.3f}   (train)")
print(f"\n1) normal shape -> anomaly location:")
print(f"   Spearman(m_sil_tr, marg_rel [edge, dim-corrected]) = {sp('m_sil_tr','marg_rel'):+.3f}   (expect -)")
print(f"   Spearman(m_sil_tr, inter    [interstitial])        = {sp('m_sil_tr','inter'):+.3f}   (expect +)")
print(f"   Spearman(m_sil_tr, iso      [local void])          = {sp('m_sil_tr','iso'):+.3f}   (expect +)")
print(f"\n2) anomaly location -> winning family:")
for m, exp in [("marg_rel", "+"), ("iso", "-"), ("inter", "-")]:
    auc = roc_auc_score(df.local_wins, -df[m] if exp == "+" else df[m])  # AUC for predicting local-wins
    print(f"   Spearman({m:8s}, glob_adv) = {sp(m,'glob_adv'):+.3f}   AUC(predict local-wins) = {auc:.3f}   (expect glob_adv sign {exp})")
print(f"\n   means split by family (winsorized iso): ")
for f, g in df.groupby("local_wins"):
    nm = "local-wins" if f else "global-wins"
    mw = mannwhitneyu(df[df.local_wins == 1].marg_rel, df[df.local_wins == 0].marg_rel).pvalue
    print(f"     {nm:11s} n={len(g):3d}  m_sil_tr={g.m_sil_tr.mean():.3f}  marg_rel={g.marg_rel.mean():.2f}  inter={g.inter.mean():.2f}  iso={g.iso.mean():.2f}")
print(f"   Mann-Whitney marg_rel (local vs global wins) p={mannwhitneyu(df[df.local_wins==1].marg_rel, df[df.local_wins==0].marg_rel).pvalue:.3f}")
print("\nsaved SHAPE_ANOMALY.csv")
