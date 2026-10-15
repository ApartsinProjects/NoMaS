# -*- coding: utf-8 -*-
"""Thoughtful single-dataset inspection of a TABULAR global MV-wins dataset (thyroid-ann): are the
anomalies genuinely an INTERSECTING set with normals (embedded in the normal cloud), and what does the
MV-picked detector exploit that our synthetic misses? Uses the HARDENED anomalies (post severity filter)."""
import os, sys, io, contextlib, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"; S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
D = os.path.join(S, "scratchpad", "streamline"); sys.path.insert(0, os.path.join(S, "scratchpad")); sys.path.insert(0, os.path.join(ROOT, "src"))
from dev_common import TAB_POOL
CORP, NAME = "ovrbench", "thyroid-ann"; Q = 0.05
LOCAL = ("LOF", "KNN", "CBLOF"); GLOBAL = ("HBOS", "COPOD", "ECOD", "PCA")
fam = lambda v: "local" if str(v).startswith(LOCAL) else ("global" if str(v).startswith(GLOBAL) else "other")
def severity(Xtr, X, bins=30):
    n, d = X.shape; sev = np.zeros(n)
    for j in range(d):
        tr = np.sort(Xtr[:, j])
        if tr[-1] - tr[0] < 1e-12: continue
        m = len(tr); fb = np.searchsorted(tr, X[:, j], side="right") / m
        cnt, edges = np.histogram(Xtr[:, j], bins=bins); dens = cnt / max(cnt.sum(), 1); b = np.clip(np.digitize(X[:, j], edges[1:-1]), 0, bins - 1)
        sev = np.maximum(sev, np.maximum(-np.log(np.maximum(2 * np.minimum(fb, 1 - fb), 1.0 / (2 * m))), -np.log(dens[b] + 1e-9)))
    return sev
def harden(Xtr, Xhold, Xa):
    tho = np.quantile(severity(Xtr, Xhold), 1 - Q); so_a = severity(Xtr, Xa)
    sc = StandardScaler().fit(Xtr); pca = PCA(n_components=0.95, whiten=True, random_state=0).fit(sc.transform(Xtr))
    Ptr, Ph, Pa = pca.transform(sc.transform(Xtr)), pca.transform(sc.transform(Xhold)), pca.transform(sc.transform(Xa))
    thp = np.quantile(severity(Ptr, Ph), 1 - Q); sp_a = severity(Ptr, Pa)
    return ~((so_a > tho) | (sp_a > thp))
d = np.load(os.path.join(ROOT, "data", CORP, NAME + ".npz"), allow_pickle=True)
X = np.nan_to_num(np.vstack([np.asarray(d["train"], float), np.asarray(d["test"], float)]))
y = np.concatenate([np.asarray(d["train_labels"]).ravel(), np.asarray(d["test_labels"]).ravel()]).astype(int)
Xn, Xa = X[y == 0], X[y == 1]
r = np.random.default_rng(0); idx = np.arange(len(Xn)); r.shuffle(idx); k = int(0.6 * len(idx))
Xtr, Xhold = Xn[idx[:k]], Xn[idx[k:]]
hmask = harden(Xtr, Xhold, Xa); Xh = Xa[hmask]
print(f"=== thyroid-ann: {X.shape[1]} features, {len(Xn)} normals, {len(Xa)} anomalies -> {len(Xh)} HARD (survive severity filter) ===\n")
# 1. Marginal intersection: fraction of hard anomalies INSIDE the normal [Q1,Q99] box on EACH feature
qlo, qhi = np.percentile(Xtr, 1, 0), np.percentile(Xtr, 99, 0)
inside_each = ((Xh >= qlo) & (Xh <= qhi))                        # per (anom,feat) within normal central 98%
all_inside = inside_each.all(1)                                  # anomaly inside the normal box on ALL features
print(f"1. MARGINAL INTERSECTION (normal central 98% box per feature):")
print(f"   hard anomalies inside the normal box on ALL {X.shape[1]} features: {all_inside.mean():.0%}  (high => embedded, not marginally extreme)")
print(f"   mean fraction of features on which an anomaly is inside the normal box: {inside_each.mean():.0%}")
# 2. best single feature separation + where the anomalies sit in that feature's normal distribution
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9; Zn = (Xhold - mu) / sd; Za = (Xh - mu) / sd
yb = np.r_[np.zeros(len(Zn)), np.ones(len(Za))]
aucs = [roc_auc_score(yb, np.r_[np.abs(Zn[:, j]), np.abs(Za[:, j])]) if np.std(np.r_[Zn[:, j], Za[:, j]]) > 0 else 0.5 for j in range(X.shape[1])]
bj = int(np.argmax(np.abs(np.array(aucs) - 0.5)) )
pct_anom = np.mean(Xtr[:, bj][None, :] < Xh[:, bj][:, None], 1)   # percentile of each anomaly within TRAIN normals on best feat
print(f"\n2. BEST SINGLE FEATURE (feat {bj}, |AUC|={aucs[bj]:.2f}): anomalies' percentile within normal distribution:")
print(f"   median {np.median(pct_anom):.0%},  IQR [{np.percentile(pct_anom,25):.0%}, {np.percentile(pct_anom,75):.0%}]   (near 50% => overlapping; near 100% => separated)")
# 3. detector pool: oracle / MV-pick / matched-pick, and score-overlap of the MV detector
Xte = np.vstack([Xhold, Xh]); yte = np.r_[np.zeros(len(Xhold)), np.ones(len(Xh))]; base = yte.mean()
t = np.linspace(0, 100, 800); alpha = np.linspace(0.9, 0.999, 800)
from hadb_round2_common import _em_auc, _mv_auc
lo, hi = Xhold.min(0), Xhold.max(0); U = lo + r.random((800, X.shape[1])) * np.where(hi > lo, hi - lo, 1.0)
ap, mvv, sc_store = {}, {}, {}
for vn, ct in TAB_POOL:
    try:
        m = ct()
        with contextlib.redirect_stdout(io.StringIO()): m.fit(Xtr)
        st = np.asarray(m.decision_function(Xte), float); sh = np.asarray(m.decision_function(Xhold), float); su = np.asarray(m.decision_function(U), float)
        if not (np.all(np.isfinite(st)) and np.nanstd(st) > 1e-12): continue
        ap[vn] = (average_precision_score(yte, st) - base) / (1 - base + 1e-12)
        su = np.nan_to_num(su, nan=float(np.nanmedian(su[np.isfinite(su)])) if np.isfinite(su).any() else 0.0)
        mvv[vn] = _mv_auc(alpha, 1.0, -su, -sh, 800); sc_store[vn] = st
    except Exception: pass
av = pd.Series(ap); oracle = av.idxmax(); mvpick = min(mvv, key=lambda v: mvv[v])
print(f"\n3. DETECTORS (ap_norm on hardened test):")
print(f"   oracle best   : {oracle:20s} ap_norm={av[oracle]:.2f}  [{fam(oracle)}]")
print(f"   MV picks      : {mvpick:20s} ap_norm={av[mvpick]:.2f}  [{fam(mvpick)}]   (regret {av.max()-av[mvpick]:.2f})")
print(f"   best local ap : {max([av[v] for v in av.index if fam(v)=='local'], default=float('nan')):.2f}   best global ap : {max([av[v] for v in av.index if fam(v)=='global'], default=float('nan')):.2f}")
# 4. figure: PCA scatter + best-feature hist + MV-detector score hist
sc2 = StandardScaler().fit(Xtr); P = PCA(2, random_state=0).fit(sc2.transform(Xtr))
Pn, Pa = P.transform(sc2.transform(Xhold)), P.transform(sc2.transform(Xh))
fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
ax[0].scatter(Pn[:, 0], Pn[:, 1], s=6, c="#bbb", alpha=.4, label="normal"); ax[0].scatter(Pa[:, 0], Pa[:, 1], s=14, c="#d1495b", alpha=.8, label="hard anomaly")
ax[0].set_title(f"PCA(2): anomalies embedded in normal cloud?"); ax[0].legend(); ax[0].set_xlabel("PC1"); ax[0].set_ylabel("PC2")
ax[1].hist(Xhold[:, bj], bins=50, color="#7aa", alpha=.6, density=True, label="normal"); ax[1].hist(Xh[:, bj], bins=30, color="#d1495b", alpha=.6, density=True, label="anomaly")
ax[1].set_title(f"best feature {bj}: distributions overlap"); ax[1].legend()
sh_mv = sc_store[mvpick]; ax[2].hist(sh_mv[yte == 0], bins=50, color="#7aa", alpha=.6, density=True, label="normal"); ax[2].hist(sh_mv[yte == 1], bins=30, color="#d1495b", alpha=.6, density=True, label="anomaly")
ax[2].set_title(f"MV-picked {mvpick} scores"); ax[2].legend()
plt.tight_layout(); fig.savefig(os.path.join(D, "STREAM_INSPECT_THYROID.png"), dpi=110); print("\nsaved streamline/STREAM_INSPECT_THYROID.png")
