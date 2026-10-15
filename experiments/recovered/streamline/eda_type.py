# -*- coding: utf-8 -*-
"""Honest dataset-type EDA (label-free). LEFT: PCA eigenvalue spectra of the most illustrative exemplars,
steep (anisotropic->local family) vs flat (isotropic->global family). RIGHT: population scatter of two
label-free structure stats (top-1 variance fraction vs effective dimension), every dataset colored by the
winning detector family, exemplars starred, single-stat family AUC annotated. No anomaly-separation claim:
the type signal lives in the NORMAL-data structure, not in an anomaly-geometry projection."""
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
    if corp in ("adbench", "dami"):
        X, y = OBJ[(corp, name)]; Xn = X[y == 0]
        r = np.random.default_rng(0)
        if len(Xn) > 4000: Xn = Xn[r.choice(len(Xn), 4000, replace=False)]
        return Xn
    Xc, lab = MTS[name]; Xw, st = mts_window_features(Xc); yw = mts_wlabels(lab, st); Xw = np.nan_to_num(Xw)
    pos = np.arange(len(Xw)); tr, va, te = block_split3(yw, pos, 0); return Xw[tr][yw[tr] == 0]
mine = pd.read_csv(os.path.join(D, "STREAM_MINE.csv"))
d = mine[mine.fam.isin(["local", "global"])].copy()
for c in ["top1_var", "eff_dim", "corr_str", "eigengap"]: d[c] = pd.to_numeric(d[c], errors="coerce")
d = d.dropna(subset=["top1_var", "eff_dim"])
y = (d.fam == "local").astype(int).values
AUC = max(roc_auc_score(y, d.top1_var.values), 1 - roc_auc_score(y, d.top1_var.values))
TEAL, ORNG = "#2a9d8f", "#e76f51"
NPC = 5
# ---- pick most illustrative exemplars by structural extremity (dim>=NPC+1 so a 5-PC spectrum is meaningful) ----
def dim_ok(r):
    try: return normals(r.corpus, r.dataset).shape[1] >= NPC + 1
    except Exception: return False
def first(cand):
    for _, r in cand.iterrows():
        if dim_ok(r): return r
    return None
TS = ~d.corpus.isin(["adbench", "dami"])  # time-series vs tabular, for domain diversity
# one tabular + one time-series exemplar per family (guarantees no corpus-duplicate like SpamBase x2)
aniso = d[d.fam == "local"]; iso = d[d.fam == "global"]
picks = [(first(aniso[~TS.loc[aniso.index]].sort_values("top1_var", ascending=False)), TEAL, "anisotropic"),
         (first(aniso[TS.loc[aniso.index]].sort_values("top1_var", ascending=False)), TEAL, "anisotropic"),
         (first(iso[TS.loc[iso.index]].sort_values("top1_var", ascending=True)), ORNG, "isotropic"),
         (first(iso[~TS.loc[iso.index]].sort_values("top1_var", ascending=True)), ORNG, "isotropic")]
seen = set(); exemplars = []
for r, col, tag in picks:
    if r is None: continue
    kk = str(r.dataset)
    if kk in seen: continue
    seen.add(kk); exemplars.append((r, col, tag))
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.6), gridspec_kw={"width_ratios": [1, 1.25]})
# ---- LEFT: eigenvalue-decay curves ----
for r, col, tag in exemplars:
    Xn = normals(r.corpus, r.dataset); ev = PCA().fit(StandardScaler().fit_transform(Xn)).explained_variance_ratio_[:NPC]
    axL.plot(range(1, NPC + 1), ev, "-o", color=col, lw=2.2, ms=6, alpha=.9)
axL.set_xlabel("principal component", fontsize=10); axL.set_ylabel("variance fraction", fontsize=10); axL.set_xticks(range(1, NPC + 1))
axL.set_title("Normal-data spectra: STEEP (anisotropic) vs FLAT (isotropic)", fontsize=10.5, fontweight="bold")
axL.legend(handles=[Line2D([], [], color=TEAL, marker="o", lw=2.2, label="anisotropic \u2192 local family"),
                    Line2D([], [], color=ORNG, marker="o", lw=2.2, label="isotropic \u2192 global family")], fontsize=8.5, frameon=False)
# ---- RIGHT: population scatter ----
loc = d[d.fam == "local"]; glo = d[d.fam == "global"]
axR.scatter(glo.top1_var, glo.eff_dim, s=34, c=ORNG, marker="o", edgecolors="#5a2a1a", linewidths=.5, alpha=.75, label=f"isotropic (global), n={len(glo)}")
axR.scatter(loc.top1_var, loc.eff_dim, s=34, c=TEAL, marker="^", edgecolors="#12463f", linewidths=.5, alpha=.75, label=f"anisotropic (local), n={len(loc)}")
for ei, (r, col, tag) in enumerate(exemplars):
    axR.scatter([r.top1_var], [r.eff_dim], s=230, marker="*", c=col, edgecolors="black", linewidths=1.1, zorder=6)
    dy = 8 if ei % 2 == 0 else -14
    axR.annotate(str(r.dataset)[:16], (r.top1_var, r.eff_dim), fontsize=7.8, fontweight="bold", xytext=(8, dy), textcoords="offset points", zorder=7)
axR.set_yscale("log"); axR.set_xlabel("top-1 PCA variance fraction  (anisotropy \u2192)", fontsize=10); axR.set_ylabel("effective dimension (participation ratio, log)", fontsize=10)
axR.set_title(f"Winning detector family by label-free structure\n(single-stat family AUC = {AUC:.2f}; families overlap, signal is real but modest)", fontsize=10)
axR.legend(fontsize=8.5, loc="upper right", frameon=True)
plt.suptitle("Dataset type is a label-free property of the NORMAL data: anisotropic (steep spectrum) datasets favor local detectors; isotropic (flat) favor global", fontsize=11, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(os.path.join(D, "FIG_type_separation.png"), dpi=125)
print("exemplars:", [(str(r.dataset)[:18], round(float(r.top1_var), 2), round(float(r.eff_dim), 1)) for r, _, _ in exemplars])
print(f"top1_var family AUC={AUC:.3f}  n_local={len(loc)} n_global={len(glo)}"); print("saved FIG_type_separation.png")
