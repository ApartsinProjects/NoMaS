# -*- coding: utf-8 -*-
"""DECODER-GENERATED synthetic anomalies (cheap proxy for VaDE-as-generator).

User's proposal: train a generative model, sample in LATENT space near-but-outside the
clusters, DECODE to produce synthetic anomalies. Clustering stays PCA+k-means; the
generative model is used ONLY to manufacture anomalies.

This tests the load-bearing question without GPU: do ON-MANIFOLD decoder-generated
anomalies rank detectors better than the current real held-out cluster?

Generators (all sample at a swept latent radius r from a cluster centre, then decode):
  gmm{r}  linear decoder  : latent = PCA-16, sample at r sigma outside a component, PCA.inverse
  ae{r}   NONLINEAR decoder: latent = small torch autoencoder, sample at r sigma, decode
  real    baseline        : the actual held-out cluster (current NoMaS pseudo-task)

VALIDITY GATE (the decoder-pullback risk): a decoder maps latents back onto the data
manifold, so "anomalies" may decode into plausible NORMAL points. We therefore first check
DETECTABILITY: if mean pseudo-AUC over detectors is ~0.5, generation failed and the run is
reported as such rather than as a ranking result.

Labels are used only for the true-AUC evaluation column.
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"
sys.path.insert(0, os.path.join(ROOT, "src"))
from adrank.pipeline import embed, cluster, fit_and_score  # noqa: E402

try:
    import torch
    import torch.nn as nn
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False

DETS = ["IForest", "LOF", "KNN", "ECOD", "COPOD", "HBOS", "PCA", "CBLOF", "LODA"]
LOCAL = {"LOF", "KNN", "CBLOF"}
GLOBAL = {"HBOS", "COPOD", "ECOD", "PCA"}
fam = lambda d: "local" if d in LOCAL else ("global" if d in GLOBAL else "other")

SEED, K, MAX_CLUSTERS, MAX_N = 0, 20, 8, 4000
RADII = [1.0, 2.0, 4.0]
OUT = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d\gen_anom.csv"
SETS = [("dami", "Annthyroid"), ("dami", "Cardiotocography"), ("dami", "PageBlocks"),
        ("dami", "SpamBase"), ("dami", "Stamps"), ("dami", "WDBC"),
        ("dami", "Waveform"), ("dami", "Wilt"),
        ("adbench", "20_letter"), ("adbench", "22_magic.gamma")]


def train_ae(X, dim=8, epochs=150, seed=SEED):
    """Small autoencoder -> a genuine NONLINEAR decoder (the VaDE-faithful part)."""
    torch.manual_seed(seed)
    n, d = X.shape
    h = max(dim * 2, min(64, d * 2))
    enc = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, dim))
    dec = nn.Sequential(nn.Linear(dim, h), nn.ReLU(), nn.Linear(h, d))
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3)
    T = torch.tensor(X, dtype=torch.float32)
    for _ in range(epochs):
        opt.zero_grad()
        loss = ((dec(enc(T)) - T) ** 2).mean()
        loss.backward(); opt.step()
    with torch.no_grad():
        Z = enc(T).numpy()
    return enc, dec, Z


def sample_outside(Zc, r, n, rng):
    """Sample n points at radius r (in local sigma units) OUTSIDE cluster Zc, in latent space."""
    mu, sd = Zc.mean(0), Zc.std(0) + 1e-9
    v = rng.normal(size=(n, Zc.shape[1]))
    v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
    return mu + v * sd * r


rows, t0 = [], time.time()
print(f"torch available: {HAVE_TORCH}", flush=True)
for sub, name in SETS:
    p = os.path.join(ROOT, "data", sub, name + ".npz")
    if not os.path.exists(p):
        continue
    a = np.load(p)
    X, y = np.asarray(a["X"], float), np.asarray(a["y"], int).ravel()
    rng = np.random.RandomState(SEED)
    if len(X) > MAX_N:
        keep = rng.choice(len(X), MAX_N, replace=False)
        X, y = X[keep], y[keep]
    if y.sum() < 5:
        continue
    true = {}
    for d in DETS:
        s = fit_and_score(d, X, X)
        if s is not None:
            true[d] = float(roc_auc_score(y, s))
    if len(true) < 6:
        continue

    Xn = X[y == 0]
    Zp = embed(Xn, dim=16)                      # clustering space (UNCHANGED, as specified)
    lab = cluster(Zp, K, seed=SEED)

    # linear decoder (PCA) fitted on raw normals
    mu_raw, sd_raw = Xn.mean(0), Xn.std(0) + 1e-9
    Xs = (Xn - mu_raw) / sd_raw
    npc = min(16, Xs.shape[1], len(Xs) - 1)
    pca = PCA(n_components=npc, random_state=SEED).fit(Xs)
    Zl = pca.transform(Xs)

    ae = None
    if HAVE_TORCH:
        try:
            ae = train_ae(Xs, dim=min(8, Xs.shape[1]), seed=SEED)
        except Exception as e:
            print(f"  AE failed on {name}: {e}", flush=True)

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
        Xtr = Xn[tr]                              # training set NEVER touched
        yt = np.r_[np.ones(nj), np.zeros(len(hold))]

        variants = {"real": Xn[m]}
        for r in RADII:
            Zs = sample_outside(Zl[m], r, nj, rng)
            variants[f"gmm{r:g}"] = pca.inverse_transform(Zs) * sd_raw + mu_raw
            if ae is not None:
                enc, dec, Za = ae
                Zs2 = sample_outside(Za[m], r, nj, rng)
                with torch.no_grad():
                    Xd = dec(torch.tensor(Zs2, dtype=torch.float32)).numpy()
                variants[f"ae{r:g}"] = Xd * sd_raw + mu_raw

        for vname, Vpts in variants.items():
            Xsc = np.vstack([Vpts, Xn[hold]])
            for d in DETS:
                s = fit_and_score(d, Xtr, Xsc)
                if s is None or d not in true:
                    continue
                rows.append(dict(dataset=name, cluster=int(j), gen=vname, detector=d,
                                 fam=fam(d), pseudo_auc=float(roc_auc_score(yt, s)),
                                 true_auc=true[d]))
        used += 1
    print(f"{name}: clusters={used} best={max(true,key=true.get)} | {time.time()-t0:.0f}s", flush=True)

pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"\nwrote {OUT} rows={len(rows)} elapsed={time.time()-t0:.0f}s")
