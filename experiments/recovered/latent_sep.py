# -*- coding: utf-8 -*-
"""Does better CLUSTER SEPARATION in the latent space predict better detector selection?

This is the load-bearing link behind "engineer a latent with clear cluster structure".
It is model-free: instead of tuning VaDE, we generate several DIFFERENT latents per
dataset (linear and nonlinear), measure separation label-free, and run the pseudo-task
in each. If separation does not track regret WITHIN a dataset, the whole line is closed
regardless of which model builds the latent.

KILL CRITERION (fixed in advance): mean within-dataset Spearman(separation, regret) must
be NEGATIVE (better separation -> lower regret) and sign-consistent on >= 2/3 of datasets.

Labels are used ONLY for the true-AUC evaluation column.
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (roc_auc_score, silhouette_score,
                             calinski_harabasz_score, davies_bouldin_score)

warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"
sys.path.insert(0, os.path.join(ROOT, "src"))
from adrank.pipeline import fit_and_score  # noqa: E402

DETS = ["IForest", "LOF", "KNN", "ECOD", "COPOD", "HBOS", "PCA", "CBLOF", "LODA"]
SEED = 0
K = 20
MAX_CLUSTERS = 12          # cap groups per latent for runtime
MAX_N = 5000
OUT = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d\latent_sep.csv"

# balanced by true-best family, measured earlier
SETS = [("dami", "Annthyroid"), ("dami", "Cardiotocography"), ("dami", "PageBlocks"),
        ("dami", "SpamBase"), ("dami", "Stamps"), ("dami", "WDBC"),
        ("dami", "Pima"), ("dami", "Waveform"), ("dami", "Wilt"),
        ("adbench", "14_glass"), ("adbench", "20_letter"), ("adbench", "22_magic.gamma")]


def latents(Xs, rng):
    """Several latent spaces of the SAME data, spanning linear -> nonlinear."""
    d = Xs.shape[1]
    out = {"raw": Xs}
    for k in (4, 8, 16, 32):
        if k < d:
            out[f"pca{k}"] = PCA(n_components=k, random_state=SEED).fit_transform(Xs)
    if d > 8:
        out["rp16"] = GaussianRandomProjection(n_components=min(16, d - 1),
                                               random_state=SEED).fit_transform(Xs)
    try:                                   # nonlinear latent, subsample the fit
        sub = Xs[rng.choice(len(Xs), min(1500, len(Xs)), replace=False)]
        kp = KernelPCA(n_components=min(16, d), kernel="rbf",
                       gamma=1.0 / max(d, 1), random_state=SEED).fit(sub)
        out["kpca16"] = kp.transform(Xs)
    except Exception:
        pass
    return out


def separation(Z, lab, rng):
    """Label-free cluster-separation quality of this latent."""
    idx = rng.choice(len(Z), min(2000, len(Z)), replace=False)
    Zs, ls = Z[idx], lab[idx]
    if len(np.unique(ls)) < 2:
        return None
    try:
        return dict(silhouette=float(silhouette_score(Zs, ls)),
                    calinski=float(calinski_harabasz_score(Zs, ls)),
                    davies=float(-davies_bouldin_score(Zs, ls)))   # negated: higher=better
    except Exception:
        return None


rows, t0 = [], time.time()
for sub, name in SETS:
    p = os.path.join(ROOT, "data", sub, name + ".npz")
    if not os.path.exists(p):
        print(f"missing {p}", flush=True); continue
    a = np.load(p)
    X, y = np.asarray(a["X"], float), np.asarray(a["y"], int).ravel()
    rng = np.random.RandomState(SEED)
    if len(X) > MAX_N:
        keep = rng.choice(len(X), MAX_N, replace=False)
        X, y = X[keep], y[keep]
    if y.sum() < 5:
        print(f"skip {name} (anom={int(y.sum())})", flush=True); continue

    true = {}
    for d in DETS:
        s = fit_and_score(d, X, X)
        if s is not None:
            true[d] = float(roc_auc_score(y, s))
    if len(true) < 6:
        print(f"skip {name} (detectors failed)", flush=True); continue
    best = max(true.values())

    Xn = X[y == 0]
    Xs = StandardScaler().fit_transform(Xn)
    for lname, Z in latents(Xs, rng).items():
        lab = MiniBatchKMeans(n_clusters=min(K, max(2, len(Z) // 20)),
                              random_state=SEED, n_init=5, batch_size=256).fit_predict(Z)
        sep = separation(Z, lab, rng)
        if sep is None:
            continue
        # pseudo-task over held-out clusters in THIS latent
        acc = {d: [] for d in DETS}
        used = 0
        for j in np.unique(lab):
            if used >= MAX_CLUSTERS:
                break
            m = lab == j
            nj = int(m.sum())
            if nj < 15 or nj > 0.25 * len(Xn):
                continue
            rest = np.where(~m)[0]; rng.shuffle(rest)
            nh = max(20, int(0.2 * len(rest)))
            hold, tr = rest[:nh], rest[nh:]
            Xsc = np.vstack([Xn[m], Xn[hold]])
            yt = np.r_[np.ones(nj), np.zeros(len(hold))]
            for d in DETS:
                s = fit_and_score(d, Xn[tr], Xsc)
                if s is not None and d in true:
                    acc[d].append(roc_auc_score(yt, s))
            used += 1
        score = {d: np.mean(v) for d, v in acc.items() if v}
        if len(score) < 5 or used < 4:
            continue
        pick = max(score, key=score.get)
        rows.append(dict(dataset=name, latent=lname, dim=Z.shape[1], n_groups=used,
                         regret=best - true[pick], pick=pick, **sep))
    print(f"{name}: latents={sum(1 for r in rows if r['dataset']==name)} "
          f"best_det={max(true,key=true.get)} | {time.time()-t0:.0f}s", flush=True)

pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"\nwrote {OUT} rows={len(rows)} elapsed={time.time()-t0:.0f}s")
