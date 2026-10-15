# -*- coding: utf-8 -*-
"""TRUE intersection test on tabular global MV-wins datasets: is the SAME point in both classes?
Measure, for each HARD anomaly, its nearest-normal distance (standardized) vs the normal-normal NN
scale. Exact duplicates = literal collision; NN distance <= normal NN scale = 'sitting on a normal'
(label not a function of features). Compares across the tabular global MV-wins datasets."""
import os, sys, warnings
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"; S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
D = os.path.join(S, "scratchpad", "streamline"); sys.path.insert(0, os.path.join(S, "scratchpad")); sys.path.insert(0, os.path.join(ROOT, "src"))
import adrank.pipeline as PP
Q = 0.05
OBJ = {}
for sub in ("adbench", "dami"):
    dd = os.path.join(ROOT, "data", sub)
    if os.path.isdir(dd):
        for ds in PP.load_npz_dir(dd): OBJ[(sub, str(ds.name)[:40])] = (np.nan_to_num(np.asarray(ds.X, float)), np.asarray(ds.y, int).ravel())
def severity(Xtr, X, bins=30):
    n, d = X.shape; sev = np.zeros(n)
    for j in range(d):
        tr = np.sort(Xtr[:, j])
        if tr[-1] - tr[0] < 1e-12: continue
        m = len(tr); fb = np.searchsorted(tr, X[:, j], side="right") / m
        cnt, edges = np.histogram(Xtr[:, j], bins=bins); dens = cnt / max(cnt.sum(), 1); b = np.clip(np.digitize(X[:, j], edges[1:-1]), 0, bins - 1)
        sev = np.maximum(sev, np.maximum(-np.log(np.maximum(2 * np.minimum(fb, 1 - fb), 1.0 / (2 * m))), -np.log(dens[b] + 1e-9)))
    return sev
def harden_mask(Xtr, Xhold, Xa):
    tho = np.quantile(severity(Xtr, Xhold), 1 - Q); so_a = severity(Xtr, Xa)
    try:
        sc = StandardScaler().fit(Xtr); pca = PCA(n_components=0.95, whiten=True, random_state=0).fit(sc.transform(Xtr))
        Ptr, Ph, Pa = pca.transform(sc.transform(Xtr)), pca.transform(sc.transform(Xhold)), pca.transform(sc.transform(Xa))
        thp = np.quantile(severity(Ptr, Ph), 1 - Q); sp_a = severity(Ptr, Pa)
    except Exception:
        sp_a = np.zeros(len(Xa)); thp = np.inf
    return ~((so_a > tho) | (sp_a > thp))
def loadXY(corp, name):
    if corp in ("oddbench", "ovrbench"):
        d = np.load(os.path.join(ROOT, "data", corp, name + ".npz"), allow_pickle=True)
        X = np.nan_to_num(np.vstack([np.asarray(d["train"], float), np.asarray(d["test"], float)])); y = np.concatenate([np.asarray(d["train_labels"]).ravel(), np.asarray(d["test_labels"]).ravel()]).astype(int)
    else:
        X, y = OBJ[(corp, name)]
    return X[y == 0], X[y == 1]
rk = pd.read_csv(os.path.join(D, "STREAM_RANK.csv")); g = rk[rk.truefam == "global"].copy(); g["gap"] = g.reg_matched - g.reg_mv
TAB = g[g.corpus.isin(["oddbench", "ovrbench", "adbench", "dami"])].sort_values("gap", ascending=False).head(12)
rows = []
for _, r in TAB.iterrows():
    try: Xn, Xa = loadXY(r.corpus, r.dataset)
    except Exception: continue
    rr = np.random.default_rng(0); idx = np.arange(len(Xn)); rr.shuffle(idx); k = int(0.6 * len(idx)); Xtr, Xhold = Xn[idx[:k]], Xn[idx[k:]]
    if len(Xtr) < 40 or len(Xhold) < 20 or len(Xa) < 5: continue
    Xh = Xa[harden_mask(Xtr, Xhold, Xa)]
    if len(Xh) < 10: continue
    sc = StandardScaler().fit(Xtr); Zh_n = sc.transform(Xhold); Za = sc.transform(Xh)          # standardized by train
    nn_norm = NearestNeighbors(n_neighbors=2).fit(Zh_n); dnn = nn_norm.kneighbors(Zh_n)[0][:, 1]  # normal-normal 1-NN
    scale = np.median(dnn) + 1e-12
    da = NearestNeighbors(n_neighbors=1).fit(Zh_n).kneighbors(Za)[0][:, 0]                        # anomaly -> nearest normal
    # exact duplicate in RAW space (same point in both classes)
    setn = set(map(lambda v: v.tobytes(), np.round(Xhold, 6))); exact = np.mean([np.round(a, 6).tobytes() in setn for a in Xh])
    rows.append({"dataset": r.dataset[:22], "corpus": r.corpus, "n_hard": len(Xh),
                 "med_anom_to_normal/normal_scale": round(float(np.median(da) / scale), 2),
                 "frac_within_normal_scale": round(float((da <= scale).mean()), 2),
                 "frac_within_0.1scale": round(float((da <= 0.1 * scale).mean()), 2),
                 "exact_dup_frac": round(float(exact), 3), "reg_mv": round(r.reg_mv, 2)})
df = pd.DataFrame(rows); df.to_csv(os.path.join(D, "STREAM_COLLISION.csv"), index=False)
print("=== TRUE-intersection test on tabular global MV-wins datasets (hard anomalies) ===")
print("  med_anom_to_normal/normal_scale: anomaly's nearest-normal distance in units of the normal-normal NN scale")
print("    ~1 => an anomaly sits as close to a normal as normals sit to each other (embedded);  >>1 => separated cloud")
print("  frac_within_normal_scale: fraction of anomalies sitting within one normal-NN-radius of a normal")
print("  exact_dup_frac: fraction of anomalies that are an EXACT duplicate of a normal row (literal same point)\n")
print(df.to_string(index=False))
print(f"\n  MEANS: med ratio {df['med_anom_to_normal/normal_scale'].mean():.2f}  |  within normal scale {df.frac_within_normal_scale.mean():.2f}  |  exact dup {df.exact_dup_frac.mean():.3f}")
print("saved streamline/STREAM_COLLISION.csv")
