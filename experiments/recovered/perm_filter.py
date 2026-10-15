# -*- coding: utf-8 -*-
"""PERMUTATION + FILTER: generate candidates by column permutation, keep only the ones a
density model judges anomalous.

User's design: permutation guarantees ORIGINAL-SPACE VALIDITY (every value is a genuinely
observed value); a learned model then FILTERS out the permutations that landed back in a
dense region. The model is used as a JUDGE only, never as a generator, so the
decoder-pullback failure (AE-decoded points scored 0.548, barely anomalous) cannot occur.

Filters compared (all trained on the TRAINING normals only, labels never seen):
  none      keep all permutations (the unfiltered baseline)
  knn       distance to k-th nearest training point        [panel-adjacent to KNN/LOF]
  gmm       negative log-likelihood under a PCA-space GMM  [panel-adjacent to PCA/ECOD]
  aelat     distance to the nearest LATENT cluster of a small autoencoder
            <- the faithful cheap proxy for the VaDE filter: ENCODER ONLY, and it is
               NOT a member of the detector panel, so it is an independent judge

Filter strength q = fraction of candidates kept (most-anomalous first).

Reported per (filter, q): acceptance rate, detectability (mean pseudo-AUC), the
global-minus-local family gap (does the filter BIAS a family?), and regret.

CIRCULARITY CHECK: a filter that is panel-adjacent hands its relatives an advantage.
That is why 'knn' and 'gmm' are included as the biased controls and 'aelat' as the
independent one. If aelat's gap differs from knn's, the filter choice matters.
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"
sys.path.insert(0, os.path.join(ROOT, "src"))
from adrank.pipeline import embed, cluster, fit_and_score  # noqa: E402

try:
    import torch, torch.nn as nn
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False

DETS = ["IForest", "LOF", "KNN", "ECOD", "COPOD", "HBOS", "PCA", "CBLOF", "LODA"]
LOCAL = {"LOF", "KNN", "CBLOF"}
GLOBAL = {"HBOS", "COPOD", "ECOD", "PCA"}
fam = lambda d: "local" if d in LOCAL else ("global" if d in GLOBAL else "other")
SEED, K, MAX_CLUSTERS, MAX_N = 0, 20, 8, 4000
OVERSAMPLE = 6                       # generate 6x candidates, then filter down
QS = [1.0, 0.5, 0.2]                 # fraction kept
OUT = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d\perm_filter.csv"
SETS = [("dami", "Annthyroid"), ("dami", "Cardiotocography"), ("dami", "PageBlocks"),
        ("dami", "SpamBase"), ("dami", "Stamps"), ("dami", "WDBC"),
        ("dami", "Waveform"), ("dami", "Wilt"),
        ("adbench", "20_letter"), ("adbench", "22_magic.gamma")]


def ae_latent(X, dim=8, epochs=150, seed=SEED):
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
print(f"torch={HAVE_TORCH}", flush=True)
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

        # --- generate OVERSAMPLE x nj permutation candidates (valid by construction) ---
        ncand = OVERSAMPLE * nj
        cand = np.column_stack([Xtr[rng.randint(0, len(Xtr), ncand), k] for k in range(dcol)])

        # --- filters, all fitted on Xtr only ---
        Ts = (Xtr - Xtr.mean(0)) / sd
        Cs = (cand - Xtr.mean(0)) / sd
        scores = {"none": rng.rand(ncand)}
        try:
            nnb = NearestNeighbors(n_neighbors=min(10, len(Ts) - 1)).fit(Ts)
            scores["knn"] = nnb.kneighbors(Cs)[0][:, -1]
        except Exception:
            pass
        try:
            npc = min(16, Ts.shape[1], len(Ts) - 1)
            pc = PCA(n_components=npc, random_state=SEED).fit(Ts)
            gm = GaussianMixture(n_components=min(8, max(2, len(Ts) // 100)),
                                 random_state=SEED).fit(pc.transform(Ts))
            scores["gmm"] = -gm.score_samples(pc.transform(Cs))
        except Exception:
            pass
        if HAVE_TORCH:
            try:
                enc = ae_latent(Ts, dim=min(8, Ts.shape[1]), seed=SEED)
                with torch.no_grad():
                    Lt = enc(torch.tensor(Ts, dtype=torch.float32)).numpy()
                    Lc = enc(torch.tensor(Cs, dtype=torch.float32)).numpy()
                # Standardize the latent, then fit a DIAGONAL, regularized mixture. The
                # full-covariance version collapses on these small tabular latents with
                # "ill-defined empirical covariance" - the same degenerate-component
                # failure the paper documents for VaDE (effK 2-3).
                lmu, lsd = Lt.mean(0), Lt.std(0) + 1e-6
                Ltn, Lcn = (Lt - lmu) / lsd, (Lc - lmu) / lsd
                ncomp = int(min(6, max(2, len(Ltn) // 200)))
                gm2 = GaussianMixture(n_components=ncomp, covariance_type="diag",
                                      reg_covar=1e-3, n_init=3,
                                      random_state=SEED).fit(Ltn)
                # distance to NEAREST latent cluster centre (VaDE-filter logic)
                dd = np.linalg.norm(Lcn[:, None, :] - gm2.means_[None, :, :], axis=2).min(1)
                scores["aelat"] = dd
            except Exception as e:
                print(f"  aelat failed {name}: {e}", flush=True)

        for fname, sc in scores.items():
            order = np.argsort(-sc)                     # most anomalous first
            for q in QS:
                if fname == "none" and q != 1.0:
                    continue
                keepn = max(nj, int(q * ncand))
                sel = order[:keepn]
                sel = sel[rng.choice(len(sel), nj, replace=False)] if len(sel) > nj else sel
                V = cand[sel]
                Xsc = np.vstack([V, Xn[hold]])
                for d in DETS:
                    s = fit_and_score(d, Xtr, Xsc)
                    if s is None or d not in true:
                        continue
                    rows.append(dict(dataset=name, cluster=int(j), filt=fname, q=q,
                                     detector=d, fam=fam(d),
                                     pseudo_auc=float(roc_auc_score(yt, s)),
                                     true_auc=true[d]))
        # real held-out cluster baseline
        Xsc = np.vstack([Xn[m], Xn[hold]])
        for d in DETS:
            s = fit_and_score(d, Xtr, Xsc)
            if s is not None and d in true:
                rows.append(dict(dataset=name, cluster=int(j), filt="real", q=1.0,
                                 detector=d, fam=fam(d),
                                 pseudo_auc=float(roc_auc_score(yt, s)), true_auc=true[d]))
        used += 1
    print(f"{name}: clusters={used} best={max(true,key=true.get)} | {time.time()-t0:.0f}s", flush=True)

pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"\nwrote {OUT} rows={len(rows)} elapsed={time.time()-t0:.0f}s")
