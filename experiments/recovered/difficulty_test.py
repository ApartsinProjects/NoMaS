# -*- coding: utf-8 -*-
"""Are GLOBAL-wins datasets just EASY (separable by a single-feature threshold)?
Per dataset compare best single-feature |z| AUC (a 1D threshold) to the best detector's AUC.
If on global-wins datasets a single feature ~ matches the best detector, 'global wins' = 'marginal-easy'.
"""
import os, sys, io, zipfile, re, warnings
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"; S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
sys.path.insert(0, os.path.join(S, "scratchpad"))
from hadb_ts_final import W, STRIDE, MAX_LEN, block_split3
from adrank.ts import _window_features, _window_labels
CSVMAP = {"oddbench": "hadb_oddbench.csv", "ovrbench": "hadb_ovrbench.csv", "ucr": "hadb_ts_ucr.csv", "tsbad_u": "hadb_ts_tsbad.csv"}


def tab_load(corpus, name):
    d = np.load(os.path.join(ROOT, "data", corpus, name + ".npz"), allow_pickle=True)
    X = np.vstack([np.asarray(d["train"], float), np.asarray(d["test"], float)]); y = np.concatenate([np.asarray(d["train_labels"]).ravel(), np.asarray(d["test_labels"]).ravel()]).astype(int); X = np.nan_to_num(X)
    r = np.random.default_rng(0); ni = np.where(y == 0)[0]; ai = np.where(y == 1)[0]
    if len(ni) > 5000: ni = r.choice(ni, 5000, replace=False)
    if len(ai) > 400: ai = r.choice(ai, 400, replace=False)
    idx = np.arange(len(ni)); r.shuffle(idx); ni = ni[idx]
    return X[ni[:int(0.6 * len(ni))]], X[ni[int(0.8 * len(ni)):]], X[ai]   # train, held-out normals, anomalies


def uni_load(name, zipname):
    z = zipfile.ZipFile(os.path.join(ROOT, "data", zipname))
    cand = [n for n in z.namelist() if os.path.basename(n) == name] or [n for n in z.namelist() if n.lower().endswith((".txt", ".csv")) and name.split("_")[0] + "_" in os.path.basename(n)]
    fn = cand[0]
    if fn.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(z.read(fn))); x = np.nan_to_num(df.iloc[:, 0].to_numpy(float)); lab = df.iloc[:, 1].to_numpy(int)
    else:
        m = re.search(r"_(\d+)_(\d+)_(\d+)\.txt$", fn); x = np.array([float(v) for v in z.read(fn).decode("utf-8", "replace").split() if v.strip()]); a0, a1 = int(m.group(2)), int(m.group(3)); lab = np.zeros(len(x), int); lab[a0:min(a1 + 1, len(x))] = 1
    if len(x) > MAX_LEN:
        a = np.where(lab == 1)[0]; c = int((a[0] + a[-1]) // 2) if len(a) else MAX_LEN // 2; lo = max(0, min(c - MAX_LEN // 2, len(x) - MAX_LEN)); x, lab = x[lo:lo + MAX_LEN], lab[lo:lo + MAX_LEN]
    Xw, _ = _window_features(x, w=W, stride=STRIDE); st = np.arange(0, len(x) - W + 1, STRIDE); yw = _window_labels(lab, st, w=W, min_count=1); Xw = np.nan_to_num(Xw)
    pos = np.arange(len(Xw)); tr, va, te = block_split3(yw, pos, 0)
    return Xw[tr][yw[tr] == 0], Xw[te][yw[te] == 0], Xw[(yw == 1)]


mp = pd.read_csv(os.path.join(S, "MODALITY_PERF.csv"))
mp["truefam"] = np.where(mp.best_local > mp.best_global, "local", "global")
# oracle detector AUC per dataset from CSVs
bestauc = {}
for c, f in CSVMAP.items():
    d = pd.read_csv(os.path.join(S, f)); ba = d.groupby(["dataset", "variant"]).auc.mean().groupby("dataset").max()
    for ds, v in ba.items(): bestauc[ds] = v
rows = []
for _, r in mp.iterrows():
    try:
        if r.corpus in ("oddbench", "ovrbench"): Xtr, Xn, Xa = tab_load(r.corpus, r.dataset)
        elif r.corpus == "ucr": Xtr, Xn, Xa = uni_load(r.dataset, "ucr/UCR_TimeSeriesAnomalyDatasets2021.zip")
        else: Xtr, Xn, Xa = uni_load(r.dataset, "tsbad/TSB-AD-U.zip")
    except Exception: continue
    if len(Xtr) < 40 or len(Xn) < 10 or len(Xa) < 5: continue
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Zn = np.abs((Xn - mu) / sd); Za = np.abs((Xa - mu) / sd)
    y = np.r_[np.zeros(len(Zn)), np.ones(len(Za))]
    best1 = 0.0
    for j in range(Zn.shape[1]):
        sc = np.r_[Zn[:, j], Za[:, j]]
        if np.std(sc) > 1e-9: best1 = max(best1, roc_auc_score(y, sc))
    rows.append({"dataset": r.dataset, "truefam": r.truefam, "best_1feat_auc": best1,
                 "best_det_auc": bestauc.get(r.dataset, np.nan), "best_apnorm": r.best_overall})
df = pd.DataFrame(rows).dropna(subset=["best_det_auc"])
df["gap"] = df.best_det_auc - df.best_1feat_auc
print(f"=== are GLOBAL-wins datasets single-feature-EASY? ({len(df)} datasets) ===")
print(f"    {'family':12s} {'n':>4s} {'1feat_AUC':>10s} {'bestDet_AUC':>12s} {'oracle_apnorm':>14s} {'det-1feat gap':>14s}")
for f in ["local", "global"]:
    g = df[df.truefam == f]
    print(f"    {f+'-wins':12s} {len(g):4d} {g.best_1feat_auc.mean():10.3f} {g.best_det_auc.mean():12.3f} {g.best_apnorm.mean():14.3f} {g.gap.mean():14.3f}")
lo, gl = df[df.truefam == "local"], df[df.truefam == "global"]
print(f"\n    Mann-Whitney 1feat_AUC (global > local?): p={mannwhitneyu(gl.best_1feat_auc, lo.best_1feat_auc, alternative='greater').pvalue:.4f}")
print(f"    Mann-Whitney det-1feat gap (local > global?): p={mannwhitneyu(lo.gap, gl.gap, alternative='greater').pvalue:.4f}")
print(f"    frac datasets where 1 feature is within 0.05 AUC of best detector:  local {np.mean(lo.gap<0.05):.2f}  global {np.mean(gl.gap<0.05):.2f}")
df.to_csv(os.path.join(S, "DIFFICULTY.csv"), index=False); print("\nsaved DIFFICULTY.csv")
