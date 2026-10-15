# -*- coding: utf-8 -*-
"""Geometry-only sweep (no model fits): across ALL bundle datasets, for a grid of eps/alpha/k, how many
density valleys does kNN density-valley probing find, and how CLOSE are they (dnn to nearest normal in
normal-median units)? Decides whether close deep-fooling voids are common or rare benchmark-wide."""
import glob, os, warnings, numpy as np
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
S = r"E:/tmp/claude/E--Projects-Submitted-ADRank/236d6247-42ad-4e62-9828-db625dfa055d/scratchpad"
BUN = os.path.join(S, "streamline", "bundle_v2")

def valley_yield(Z, k, eps, alpha, n_try=300, n_dir=6, seed=0):
    rng = np.random.default_rng(seed); n, d = Z.shape
    if n < k + 2: return 0, np.nan
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z)
    dist, idx = nn.kneighbors(Z); r = dist[:, -1]; Ni = [set(row[1:]) for row in idx]
    rho_orig = 1.0 / (dist[:, 1:].mean(1) + 1e-9)
    dnn_med = np.median(dist[:, 1]) + 1e-9
    def rho(pts): return 1.0 / (nn.kneighbors(pts, n_neighbors=k, return_distance=True)[0].mean(1) + 1e-9)
    acc, dnns = 0, []
    for i in rng.choice(n, min(n, n_try), replace=False):
        for _ in range(n_dir):
            u = rng.normal(size=d); u /= np.linalg.norm(u) + 1e-12
            z = Z[i] + eps * r[i] * u
            nbrs = nn.kneighbors(z[None], n_neighbors=k, return_distance=False)[0]
            newn = list(set(nbrs) - Ni[i] - {i})
            if not newn: continue
            j = min(newn, key=lambda jj: np.linalg.norm(Z[jj] - z))
            ts = np.linspace(0.15, 0.85, 8); seg = (1 - ts)[:, None] * Z[i] + ts[:, None] * Z[j]
            rr = rho(seg); m = int(rr.argmin())
            if rr[m] < alpha * min(rho_orig[i], rho_orig[j]):
                acc += 1
                dnns.append(nn.kneighbors(seg[m][None], n_neighbors=1, return_distance=True)[0][0, 0] / dnn_med)
    return acc, (np.median(dnns) if dnns else np.nan)

files = sorted(glob.glob(os.path.join(BUN, "*.npz")))
GRID = [(10, 1.6, 0.6), (10, 1.0, 0.6), (8, 0.7, 0.75), (8, 0.5, 0.8), (12, 1.0, 0.7)]
print(f"datasets: {len(files)}  (n_try=300 anchors, 6 dirs each = up to 1800 probes/dataset)\n")
print(f"{'k,eps,alpha':>16s} {'datasets w/>=20 valleys':>24s} {'mean valleys/ds':>16s} {'median dnn (close<3)':>20s}")
for (k, eps, alpha) in GRID:
    yields, dnns = [], []
    for f in files:
        try:
            z = np.load(f); Xtr = z["Xtr"].astype("float64")
            if len(Xtr) < 40 or Xtr.shape[1] < 2: continue
            Z = StandardScaler().fit_transform(Xtr)
            a, dnn = valley_yield(Z, k, eps, alpha)
            yields.append(a);
            if not np.isnan(dnn): dnns.append(dnn)
        except Exception: continue
    y = np.array(yields)
    print(f"{f'{k},{eps},{alpha}':>16s} {int((y>=20).sum()):>10d} / {len(y):<11d} {y.mean():>16.1f} {np.nanmedian(dnns):>20.1f}")
