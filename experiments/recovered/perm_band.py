# -*- coding: utf-8 -*-
"""PERMUTATION with a difficulty DIAL and a latent-distance BAND.

Refined design:
  1. Generate MANY candidates by permuting a FRACTION f of the features of a real record.
     Every value stays a genuinely observed value, so the record is valid in the original
     space by construction. f is the difficulty dial: f small = subtle, near-manifold
     anomaly; f = 1 = maximally implausible combination.
  2. Encode candidates with an autoencoder ENCODER (never the decoder) and score them by
     distance to the nearest latent cluster centre  <- the VaDE-filter logic.
  3. Keep only candidates whose latent distance falls in a BAND [lo, hi] percentile, NOT
     the top-q tail. A band gives CONTROLLED difficulty; the tail gives saturation, where
     every detector succeeds and discrimination dies (measured: at displacement delta=4,
     69% of pseudo-AUCs exceeded 0.95 and the family gap REOPENED).

Scope: tables and time series only. Permuting pixels across images yields noise, not a
plausible image; and the local/global problem this addresses is absent on deep embeddings
(measured global-best share: images 0%, text 1.5%, tabular 15%, DAMI 42%).

Labels are used only for the true-AUC evaluation column.
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"
sys.path.insert(0, os.path.join(ROOT, "src"))
from adrank.pipeline import embed, cluster, fit_and_score  # noqa: E402
import torch, torch.nn as nn

DETS = ["IForest", "LOF", "KNN", "ECOD", "COPOD", "HBOS", "PCA", "CBLOF", "LODA"]
LOCAL = {"LOF", "KNN", "CBLOF"}
GLOBAL = {"HBOS", "COPOD", "ECOD", "PCA"}
fam = lambda d: "local" if d in LOCAL else ("global" if d in GLOBAL else "other")
SEED, K, MAX_CLUSTERS, MAX_N = 0, 20, 6, 4000
FRACS = [0.25, 0.5, 1.0]                        # difficulty dial
BANDS = [(50, 70), (70, 90), (90, 100)]         # latent-distance percentile bands
OVERSAMPLE = 12
OUT = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d\perm_band.csv"
SETS = [("dami", "Annthyroid"), ("dami", "Cardiotocography"), ("dami", "PageBlocks"),
        ("dami", "SpamBase"), ("dami", "Stamps"), ("dami", "WDBC"),
        ("dami", "Waveform"), ("dami", "Wilt"),
        ("adbench", "20_letter"), ("adbench", "22_magic.gamma")]


def ae_encoder(X, dim=8, epochs=150, seed=SEED):
    torch.manual_seed(seed)
    n, d = X.shape
    h = max(dim * 2, min(64, d * 2))
    enc = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, dim))
    dec = nn.Sequential(nn.Linear(dim, h), nn.ReLU(), nn.Linear(h, d))
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3)
    T = torch.tensor(X, dtype=torch.float32)
    for _ in range(epochs):
        opt.zero_grad(); ((dec(enc(T)) - T) ** 2).mean().backward(); opt.step()
    return enc


rows, t0 = [], time.time()
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
    sd = Xn.std(0) + 1e-9
    Zp = embed(Xn, dim=16)
    lab = cluster(Zp, K, seed=SEED)
    dcol = Xn.shape[1]

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
        Xtr = Xn[tr]
        yt = np.r_[np.ones(nj), np.zeros(len(hold))]
        Ts = (Xtr - Xtr.mean(0)) / sd

        # one encoder per (dataset, cluster); reused across fracs and bands
        try:
            enc = ae_encoder(Ts, dim=min(8, dcol), seed=SEED)
            with torch.no_grad():
                Lt = enc(torch.tensor(Ts, dtype=torch.float32)).numpy()
            lmu, lsd = Lt.mean(0), Lt.std(0) + 1e-6
            Ltn = (Lt - lmu) / lsd
            gm = GaussianMixture(n_components=int(min(6, max(2, len(Ltn) // 200))),
                                 covariance_type="diag", reg_covar=1e-3, n_init=3,
                                 random_state=SEED).fit(Ltn)
        except Exception as e:
            print(f"  encoder failed {name} c{j}: {e}", flush=True)
            continue

        for f in FRACS:
            ncand = OVERSAMPLE * nj
            nperm = max(1, int(round(f * dcol)))
            base = Xtr[rng.randint(0, len(Xtr), ncand)].copy()
            for i in range(ncand):                       # permute a FRACTION of features
                ks = rng.choice(dcol, nperm, replace=False)
                base[i, ks] = Xtr[rng.randint(0, len(Xtr), nperm), ks]
            Cs = (base - Xtr.mean(0)) / sd
            with torch.no_grad():
                Lc = enc(torch.tensor(Cs, dtype=torch.float32)).numpy()
            dist = np.linalg.norm(((Lc - lmu) / lsd)[:, None, :] - gm.means_[None, :, :],
                                  axis=2).min(1)
            for lo, hi in BANDS:
                a_lo, a_hi = np.percentile(dist, lo), np.percentile(dist, hi)
                sel = np.where((dist >= a_lo) & (dist <= a_hi))[0]
                if len(sel) < nj:
                    continue
                sel = sel[rng.choice(len(sel), nj, replace=False)]
                V = base[sel]
                Xsc = np.vstack([V, Xn[hold]])
                for d in DETS:
                    s = fit_and_score(d, Xtr, Xsc)
                    if s is None or d not in true:
                        continue
                    rows.append(dict(dataset=name, cluster=int(j), frac=f,
                                     band=f"{lo}-{hi}", detector=d, fam=fam(d),
                                     lat_dist=float(dist[sel].mean()),
                                     pseudo_auc=float(roc_auc_score(yt, s)), true_auc=true[d]))
        # baseline: the real held-out cluster
        Xsc = np.vstack([Xn[m], Xn[hold]])
        for d in DETS:
            s = fit_and_score(d, Xtr, Xsc)
            if s is not None and d in true:
                rows.append(dict(dataset=name, cluster=int(j), frac=0.0, band="real",
                                 detector=d, fam=fam(d), lat_dist=np.nan,
                                 pseudo_auc=float(roc_auc_score(yt, s)), true_auc=true[d]))
        used += 1
    print(f"{name}: clusters={used} best={max(true,key=true.get)} | {time.time()-t0:.0f}s", flush=True)

pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"\nwrote {OUT} rows={len(rows)} elapsed={time.time()-t0:.0f}s")
