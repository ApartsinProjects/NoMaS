# -*- coding: utf-8 -*-
"""Fable-designed honest dataset-type figure.
PANEL A: Lorenz curves of the NORMAL-data covariance spectrum. Per dataset: cumulative explained-variance
vs fraction of components k/d, one thin line colored by winning family, family-median curves + IQR bands,
y=x diagonal = 'perfectly isotropic (flat spectrum)'. Anisotropic bows to the corner; isotropic hugs the
diagonal. Plots spectrum SHAPE vs a meaningful anchor (not two collinear scalars); overlap shown honestly.
PANEL B: dose-response. Local-win fraction per quintile of a label-free anisotropy index (z(top1_var)+
z(corr_str)+z(eigengap)), Wilson 95% CIs, base-rate line. Renders the modest AUC~0.66 as a felt probability."""
import os, sys, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"; S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
D = os.path.join(S, "scratchpad", "streamline"); sys.path.insert(0, os.path.join(S, "scratchpad")); sys.path.insert(0, os.path.join(ROOT, "src"))
import adrank.pipeline as PP
from hadb_ts_final import block_split3
from hadb_ts_mts import mts_window_features, window_labels as mts_wlabels, load_mts
OBJ = {}
for sub in ("adbench", "dami"):
    dd = os.path.join(ROOT, "data", sub)
    if os.path.isdir(dd):
        for ds in PP.load_npz_dir(dd): OBJ[(sub, str(ds.name)[:40])] = (np.nan_to_num(np.asarray(ds.X, float)), np.asarray(ds.y, int).ravel())
MTS = {}
for name, src, Xc, lab in load_mts(200): MTS[str(name)[:40]] = (Xc, lab)
def normals(corp, name):
    if corp in ("oddbench", "ovrbench"):
        dd = np.load(os.path.join(ROOT, "data", corp, name + ".npz"), allow_pickle=True)
        X = np.nan_to_num(np.vstack([np.asarray(dd["train"], float), np.asarray(dd["test"], float)]))
        y = np.concatenate([np.asarray(dd["train_labels"]).ravel(), np.asarray(dd["test_labels"]).ravel()]).astype(int)
        Xn = X[y == 0]
    elif corp in ("adbench", "dami"):
        X, y = OBJ[(corp, name)]; Xn = X[y == 0]
    else:
        Xc, lab = MTS[name]; Xw, st = mts_window_features(Xc); yw = mts_wlabels(lab, st); Xw = np.nan_to_num(Xw)
        pos = np.arange(len(Xw)); tr, va, te = block_split3(yw, pos, 0); return Xw[tr][yw[tr] == 0]
    r = np.random.default_rng(0)
    if len(Xn) > 4000: Xn = Xn[r.choice(len(Xn), 4000, replace=False)]
    return Xn
def wilson(k, n, z=1.96):
    if n == 0: return (0, 0, 0)
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0, c - half), min(1, c + half)
mine = pd.read_csv(os.path.join(D, "STREAM_MINE.csv"))
d = mine[mine.fam.isin(["local", "global"])].copy()
for c in ["top1_var", "corr_str", "eigengap", "eff_dim"]: d[c] = pd.to_numeric(d[c], errors="coerce")
TEAL, ORNG = "#2a9d8f", "#e76f51"   # anisotropic/local = teal, isotropic/global = orange (consistent w/ other figs)
GRID = np.linspace(0, 1, 61)
# ---- Panel A data: cumulative EV curve per dataset on common k/d grid ----
curves = {"local": [], "global": []}
for _, r in d.iterrows():
    try:
        Xn = normals(r.corpus, r.dataset)
        if Xn.shape[1] < 3 or len(Xn) < 40: continue
        ev = PCA().fit(StandardScaler().fit_transform(Xn)).explained_variance_ratio_
        dd = len(ev); xk = np.arange(0, dd + 1) / dd; cum = np.concatenate([[0], np.cumsum(ev)])
        curves[r.fam].append(np.interp(GRID, xk, cum))
    except Exception:
        continue
nL, nG = len(curves["local"]), len(curves["global"])
CL = np.array(curves["local"]); CG = np.array(curves["global"])
# ---- Panel B data: anisotropy index quintiles -> local-win fraction ----
d3 = d.dropna(subset=["top1_var", "corr_str", "eigengap"]).copy()
z = lambda s: (s - s.mean()) / (s.std() + 1e-9)
d3["idx"] = z(d3.top1_var) + z(d3.corr_str) + z(d3.eigengap)
AUC = max(roc_auc_score((d3.fam == "local").astype(int), d3.idx), 1 - roc_auc_score((d3.fam == "local").astype(int), d3.idx))
NQ = 5
d3["q"] = pd.qcut(d3.idx, NQ, labels=False)
base = (d.fam == "local").mean()
qs = []
for q in range(NQ):
    g = d3[d3.q == q]; k = int((g.fam == "local").sum()); n = len(g); p, lo, hi = wilson(k, n)
    qs.append((q, p, lo, hi, n))
mono = all(qs[i][1] <= qs[i + 1][1] for i in range(NQ - 1))
# ---- draw ----
fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.7), gridspec_kw={"width_ratios": [1.15, 1]})
# Panel A
for arr, col in [(CG, ORNG), (CL, TEAL)]:
    for row in arr: axA.plot(GRID, row, color=col, lw=0.7, alpha=0.11, zorder=1)
for arr, col, lab, n in [(CG, ORNG, "isotropic (global)", nG), (CL, TEAL, "anisotropic (local)", nL)]:
    med = np.median(arr, 0); q1 = np.percentile(arr, 25, 0); q3 = np.percentile(arr, 75, 0)
    axA.fill_between(GRID, q1, q3, color=col, alpha=0.22, zorder=2)
    axA.plot(GRID, med, color=col, lw=3.0, zorder=4, solid_capstyle="round")
axA.plot([0, 1], [0, 1], "--", color="#888", lw=1.4, zorder=3)
axA.annotate("perfectly isotropic\n(flat spectrum)", (0.62, 0.62), rotation=39, color="#666", fontsize=8.5, ha="center", va="center", rotation_mode="anchor")
axA.annotate("variance concentrated\nin few components", (0.30, 0.90), color=TEAL, fontsize=9, ha="center", fontweight="bold")
axA.set_xlim(0, 1); axA.set_ylim(0, 1); axA.set_aspect("equal")
axA.set_xlabel("fraction of principal components  (k / d)", fontsize=10); axA.set_ylabel("cumulative variance fraction", fontsize=10)
axA.set_title("Normal-data variance concentration (spectrum shape)", fontsize=10.5, fontweight="bold")
axA.legend(handles=[Line2D([], [], color=TEAL, lw=3, label=f"anisotropic (local), n={nL}"),
                    Line2D([], [], color=ORNG, lw=3, label=f"isotropic (global), n={nG}"),
                    Line2D([], [], color="#888", lw=1.4, ls="--", label="flat-spectrum reference")], fontsize=8.5, loc="lower right", frameon=True)
# Panel B
xq = np.arange(NQ)
axB.axhline(base, ls="--", color="#c00", lw=1.2, alpha=.7); axB.annotate(f"base rate {base:.2f}", (NQ - 0.65, base), color="#c00", fontsize=8, va="center", ha="right")
for q, p, lo, hi, n in qs:
    axB.errorbar(q, p, yerr=[[p - lo], [hi - p]], fmt="o", ms=13, color=TEAL, ecolor="#555", elinewidth=1.5, capsize=6, zorder=5)
    axB.annotate(f"n={n}", (q, hi + 0.02), ha="center", fontsize=8, color="#555")
axB.plot(xq, [p for _, p, _, _, _ in qs], color=TEAL, lw=1.6, alpha=.5, zorder=3)
labs = [""] * NQ; labs[0] = "Q1\nleast\nanisotropic"; labs[-1] = f"Q{NQ}\nmost\nanisotropic"
for m in range(1, NQ - 1): labs[m] = f"Q{m+1}"
axB.set_xticks(xq); axB.set_xticklabels(labs, fontsize=8.5)
axB.set_ylim(0, 1); axB.set_xlim(-0.4, NQ - 0.6); axB.set_ylabel("fraction of datasets won by LOCAL detectors", fontsize=10)
axB.set_xlabel("quintile of label-free anisotropy index", fontsize=10)
pmin = min(p for _, p, _, _, _ in qs); pmax = max(p for _, p, _, _, _ in qs)
axB.set_title(f"Label-free anisotropy predicts the winning family\n(index AUC = {AUC:.2f}; overlap is real, local-win rate spans {pmin:.0%}–{pmax:.0%})", fontsize=10)
plt.suptitle("Dataset type is a label-free property of the NORMAL data: anisotropic datasets concentrate variance and favor LOCAL detectors", fontsize=11, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(os.path.join(D, "FIG_lorenz_dose.png"), dpi=125)
print(f"A: n_local={nL} n_global={nG}  B: AUC={AUC:.3f} base={base:.3f}")
print("quintile local-win:", [(q, round(p, 2), n) for q, p, _, _, n in qs]); print("saved FIG_lorenz_dose.png")
