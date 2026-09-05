# -*- coding: utf-8 -*-
"""Mine label-free NORMAL-data statistics that separate LOCAL- vs GLOBAL-family datasets. Panel:
variance/spectrum (eff-dim, top-PC var, low-PC var, intrinsic dim, eigengap), locality/density (kNN
scale, density heterogeneity, two-NN intrinsic dim, hubness), fine structure (GMM modes, void_mass,
silhouette, marginal multimodality), latent geometry (correlation strength, intrinsic/ambient ratio,
kurtosis, skew). Target = local-global gap (maxL-maxG) from the co-computed pool. Rank by AUC & Spearman.
Incremental per-dataset CSV + resume."""
import os, sys, warnings
import numpy as np, pandas as pd
from scipy.stats import spearmanr, kurtosis, skew
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score, silhouette_score
warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"; S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
D = os.path.join(S, "scratchpad", "streamline"); sys.path.insert(0, os.path.join(S, "scratchpad")); sys.path.insert(0, os.path.join(ROOT, "src"))
import adrank.pipeline as PP
from hadb_ts_final import W, STRIDE, block_split3
from hadb_ts_mts import mts_window_features, window_labels as mts_wlabels, load_mts
LOCAL = ("LOF", "KNN", "CBLOF"); GLOBAL = ("HBOS", "COPOD", "ECOD", "PCA"); OUT = os.path.join(D, "STREAM_MINE.csv")
OBJ = {}
for sub in ("adbench", "dami"):
    dd = os.path.join(ROOT, "data", sub)
    if os.path.isdir(dd):
        for ds in PP.load_npz_dir(dd): OBJ[(sub, str(ds.name)[:40])] = (np.nan_to_num(np.asarray(ds.X, float)), np.asarray(ds.y, int).ravel())
MTS = {}
for name, src, Xc, lab in load_mts(200): MTS[str(name)[:40]] = (Xc, lab)
def get_norm(corp, name):
    if corp in ("oddbench", "ovrbench"):
        d = np.load(os.path.join(ROOT, "data", corp, name + ".npz"), allow_pickle=True)
        X = np.nan_to_num(np.vstack([np.asarray(d["train"], float), np.asarray(d["test"], float)])); y = np.concatenate([np.asarray(d["train_labels"]).ravel(), np.asarray(d["test_labels"]).ravel()]).astype(int); Xn = X[y == 0]
    elif corp in ("adbench", "dami"):
        X, y = OBJ[(corp, name)]; Xn = X[y == 0]
    else:
        Xc, lab = MTS[name]; Xw, st = mts_window_features(Xc); yw = mts_wlabels(lab, st); Xw = np.nan_to_num(Xw)
        pos = np.arange(len(Xw)); tr, va, te = block_split3(yw, pos, 0); Xn = Xw[tr][yw[tr] == 0]
    return Xn
def feats(Xn):
    r = np.random.default_rng(0); sub = Xn if len(Xn) <= 2000 else Xn[r.choice(len(Xn), 2000, replace=False)]
    Z = StandardScaler().fit_transform(sub); d = Z.shape[1]; F = {"d": d, "logn": float(np.log(len(Xn)))}
    # --- variance / spectrum ---
    pca = PCA().fit(Z); ev = pca.explained_variance_ratio_
    F["eff_dim"] = float(1.0 / np.sum(ev ** 2)); F["eff_dim_ratio"] = F["eff_dim"] / d
    F["top1_var"] = float(ev[0]); F["low_pc_var"] = float(ev[len(ev) // 2:].sum())
    F["intrinsic95"] = int(np.argmax(np.cumsum(ev) >= 0.95) + 1); F["intr_ratio"] = F["intrinsic95"] / d
    F["eigengap"] = float(ev[0] - ev[1]) if len(ev) > 1 else 0.0
    F["kurt"] = float(np.mean(kurtosis(sub, axis=0))); F["skew_abs"] = float(np.mean(np.abs(skew(sub, axis=0))))
    C = np.corrcoef(Z.T); F["corr_str"] = float(np.mean(np.abs(C[np.triu_indices(d, 1)]))) if d > 1 else 0.0
    # --- locality / density ---
    nn = NearestNeighbors(n_neighbors=min(6, len(Z) - 1)).fit(Z); dist, _ = nn.kneighbors(Z)
    d1 = dist[:, 1]; F["nn_scale"] = float(np.median(d1))
    F["dens_het"] = float(np.std(d1) / (np.mean(d1) + 1e-9))                 # density heterogeneity (local var)
    F["nn_ratio"] = float(np.percentile(d1, 90) / (np.percentile(d1, 10) + 1e-9))
    r2 = dist[:, 2] if dist.shape[1] > 2 else dist[:, 1]; mu = r2 / (d1 + 1e-12); mu = mu[mu > 1]
    F["twoNN_id"] = float((len(mu)) / (np.sum(np.log(mu)) + 1e-9)) if len(mu) > 5 else float(d)   # Facco intrinsic dim
    F["twoNN_ratio"] = F["twoNN_id"] / d
    kocc = np.bincount(nn.kneighbors(Z, return_distance=False)[:, 1:].ravel(), minlength=len(Z))
    F["hubness"] = float(skew(kocc))                                        # k-occurrence skew (hubness)
    # --- fine structure / multimodality ---
    try:
        models = [GaussianMixture(k, covariance_type="diag", random_state=0, reg_covar=1e-4).fit(Z) for k in range(1, 9)]
        F["n_modes"] = int(np.argmin([m.bic(Z) for m in models]) + 1)
        g1, gk = models[0], models[F["n_modes"] - 1]; samp, _ = g1.sample(1500); thr = np.quantile(gk.score_samples(Z), 0.05)
        F["void_mass"] = float(np.mean(gk.score_samples(samp) < thr)); F["ll_gap"] = float(gk.score(Z) - g1.score(Z))
    except Exception: F["n_modes"], F["void_mass"], F["ll_gap"] = 1, 0.0, 0.0
    try: F["silh"] = float(max(silhouette_score(Z, KMeans(k, n_init=3, random_state=0).fit_predict(Z)) for k in (2, 3, 4)))
    except Exception: F["silh"] = 0.0
    return F
dm = pd.read_csv(os.path.join(D, "STREAM_DEEP_MODAL.csv")); idx = pd.read_csv(os.path.join(D, "bundle", "_index.csv"))[["key", "corpus", "dataset"]]
m = dm.merge(idx, on="key", how="inner")
lc = [c for c in dm.columns if str(c).startswith(LOCAL)]; gc = [c for c in dm.columns if str(c).startswith(GLOBAL)]
m["maxL"] = m[lc].max(1); m["maxG"] = m[gc].max(1); m["gap"] = m.maxL - m.maxG
# (all 173 datasets - the ranking analysis still weights by family; corr_str computed for every dataset)
done = set(pd.read_csv(OUT).dataset) if os.path.exists(OUT) else set(); fo = None
for _, r in m.iterrows():
    if r.dataset in done: continue
    try: Xn = get_norm(r.corpus, r.dataset)
    except Exception: continue
    if len(Xn) < 60: continue
    try: F = feats(Xn)
    except Exception: continue
    F.update({"corpus": r.corpus, "dataset": r.dataset, "gap": float(r.gap), "fam": "local" if r.gap > 0 else "global"})
    if fo is None:
        newfile = not os.path.exists(OUT); fo = open(OUT, "a")
        if newfile: fo.write(",".join(F.keys()) + "\n")
    fo.write(",".join(str(F[k]) for k in F) + "\n"); fo.flush()
    print(f"  {r.dataset[:22]} fam={F['fam']} eff_dim={F['eff_dim']:.1f} nn_het={F['dens_het']:.2f} modes={F['n_modes']}", flush=True)
if fo: fo.close()
df = pd.read_csv(OUT); y = (df.fam == "local").astype(int)
FEATS = ["eff_dim", "eff_dim_ratio", "top1_var", "low_pc_var", "intr_ratio", "eigengap", "kurt", "skew_abs", "corr_str",
         "nn_scale", "dens_het", "nn_ratio", "twoNN_id", "twoNN_ratio", "hubness", "n_modes", "void_mass", "ll_gap", "silh", "d"]
print(f"\n=== features separating LOCAL vs GLOBAL ({len(df)}: {int(y.sum())} local / {int((1-y).sum())} global) ===")
print("  ranked by |AUC-0.5| (AUC>0.5 => higher value predicts LOCAL):\n")
res = []
for c in FEATS:
    v = df[c].values.astype(float); ok = np.isfinite(v)
    if ok.sum() < 20 or np.std(v[ok]) == 0: continue
    auc = roc_auc_score(y[ok], v[ok]); rho = spearmanr(v[ok], df.gap.values[ok]).statistic
    res.append((c, auc, rho, np.mean(v[y == 1]), np.mean(v[y == 0])))
for c, auc, rho, ml, mg in sorted(res, key=lambda t: -abs(t[1] - 0.5)):
    print(f"  {c:14s} AUC={auc:.2f}  rho_gap={rho:+.2f}   local {ml:8.2f}  global {mg:8.2f}")
print("saved streamline/STREAM_MINE.csv")
