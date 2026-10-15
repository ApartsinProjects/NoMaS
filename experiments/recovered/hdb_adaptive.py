# -*- coding: utf-8 -*-
"""Can a LABEL-FREE adaptive min_cluster_size make HDBSCAN produce enough regimes?

Rule (fixed in advance, uses no labels): scan min_cluster_size downward over a fixed
grid and take the SMALLEST mcs whose cluster count reaches TARGET_K, reporting the
cluster count and noise fraction. No labels are read anywhere in this script.
"""
import os, sys, warnings
import numpy as np
from sklearn.cluster import HDBSCAN

warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"
sys.path.insert(0, os.path.join(ROOT, "src"))
from adrank.pipeline import embed  # noqa: E402

TARGET_K = 15
DATA = os.path.join(ROOT, "data", "dami")
print(f"{'dataset':18s} {'n':>6s} {'d':>5s} | default_mcs -> k  | adaptive: mcs -> k (noise%)")
for fn in sorted(f for f in os.listdir(DATA) if f.endswith(".npz")):
    a = np.load(os.path.join(DATA, fn))
    X, y = np.asarray(a["X"], float), np.asarray(a["y"], int).ravel()
    if X.shape[0] < 200 or X.shape[0] > 20000:
        continue
    Xn = X[y == 0]                       # normal pool only, as the method assumes
    Z = embed(Xn, dim=16)
    n = len(Z)
    dflt = max(15, n // 100)
    k_dflt = len(set(l for l in HDBSCAN(min_cluster_size=dflt).fit_predict(Z) if l >= 0))
    best = None
    for mcs in [dflt, 50, 30, 20, 15, 10, 8, 6, 5]:
        if mcs < 5 or mcs > n // 3:
            continue
        lab = HDBSCAN(min_cluster_size=int(mcs)).fit_predict(Z)
        k = len(set(l for l in lab if l >= 0))
        noise = float((lab == -1).mean())
        if best is None or k > best[1]:
            best = (int(mcs), k, noise)
        if k >= TARGET_K:
            best = (int(mcs), k, noise)
            break
    print(f"{fn[:-4]:18s} {n:6d} {Z.shape[1]:5d} | {dflt:11d} -> {k_dflt:2d} | "
          f"{best[0]:11d} -> {best[1]:2d} ({100*best[2]:.0f}%)")
