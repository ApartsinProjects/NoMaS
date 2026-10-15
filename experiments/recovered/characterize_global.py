# -*- coding: utf-8 -*-
"""What is special about GLOBAL-wins datasets? Test 4 distinctive properties, find the sharpest
(and which are label-free = filter-worthy). Split by family (from ap_norm).
  axis_drop   single-feature AUC minus single-feature AUC after a random rotation (axis-aligned?)
  hist_gain   HBOS-lite single-feature AUC minus |z| single-feature AUC (histogram-rare vs tail?)
  norm_corr   mean |feature correlation| of the NORMALS (label-free)
  kfrac       fraction of features needed for |z|-sum to reach 90% of full AUC (low-dim signal?)
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
    return X[ni[:int(0.6 * len(ni))]], X[ni[int(0.8 * len(ni)):]], X[ai]


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


def best1(Zn, Za):
    y = np.r_[np.zeros(len(Zn)), np.ones(len(Za))]; best = 0.0
    for j in range(Zn.shape[1]):
        sc = np.r_[Zn[:, j], Za[:, j]]
        if np.std(sc) > 1e-9: best = max(best, roc_auc_score(y, sc))
    return best


mp = pd.read_csv(os.path.join(S, "MODALITY_PERF.csv")); mp["truefam"] = np.where(mp.best_local > mp.best_global, "local", "global")
rows = []
for _, r in mp.iterrows():
    try:
        if r.corpus in ("oddbench", "ovrbench"): Xtr, Xn, Xa = tab_load(r.corpus, r.dataset)
        elif r.corpus == "ucr": Xtr, Xn, Xa = uni_load(r.dataset, "ucr/UCR_TimeSeriesAnomalyDatasets2021.zip")
        else: Xtr, Xn, Xa = uni_load(r.dataset, "tsbad/TSB-AD-U.zip")
    except Exception: continue
    if len(Xtr) < 40 or len(Xn) < 10 or len(Xa) < 5: continue
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9; d = Xtr.shape[1]
    Zn = (Xn - mu) / sd; Za = (Xa - mu) / sd
    z1 = best1(np.abs(Zn), np.abs(Za))
    # rotation (axis-alignment)
    rng = np.random.default_rng(0); Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    r1 = best1(np.abs(Zn @ Q), np.abs(Za @ Q)); axis_drop = z1 - r1
    # histogram single-feature (HBOS-lite): -log density of the bin
    y = np.r_[np.zeros(len(Zn)), np.ones(len(Za))]; hbest = 0.0
    Ztr = (Xtr - mu) / sd
    for j in range(d):
        cnt, edges = np.histogram(Ztr[:, j], bins=30); dens = cnt / max(cnt.sum(), 1) + 1e-9
        def sc(col): b = np.clip(np.digitize(col, edges[1:-1]), 0, 29); return -np.log(dens[b])
        s = np.r_[sc(Zn[:, j]), sc(Za[:, j])]
        if np.std(s) > 1e-9: hbest = max(hbest, roc_auc_score(y, s))
    hist_gain = hbest - z1
    # normal feature correlation (label-free)
    C = np.corrcoef(Xtr.T) if d > 1 else np.array([[1.0]]); norm_corr = np.nanmean(np.abs(C[np.triu_indices(d, 1)])) if d > 1 else 0.0
    # k features to 90% of full |z|-sum AUC
    order = sorted(range(d), key=lambda j: -roc_auc_score(y, np.r_[np.abs(Zn[:, j]), np.abs(Za[:, j])]) if np.std(np.r_[Zn[:, j], Za[:, j]]) > 1e-9 else 0)
    full = roc_auc_score(y, np.r_[np.abs(Zn).sum(1), np.abs(Za).sum(1)]); kfrac = 1.0
    for k in range(1, d + 1):
        cols = order[:k]; a = roc_auc_score(y, np.r_[np.abs(Zn[:, cols]).sum(1), np.abs(Za[:, cols]).sum(1)])
        if a >= 0.9 * full: kfrac = k / d; break
    rows.append({"dataset": r.dataset, "truefam": r.truefam, "axis_drop": axis_drop, "hist_gain": hist_gain,
                 "norm_corr": norm_corr, "kfrac": kfrac, "d": d})
df = pd.DataFrame(rows); df.to_csv(os.path.join(S, "CHARACTERIZE_GLOBAL.csv"), index=False)
lo, gl = df[df.truefam == "local"], df[df.truefam == "global"]
print(f"=== what is special about GLOBAL-wins datasets? ({len(df)}: {len(lo)} local / {len(gl)} global) ===")
print(f"    {'metric':12s} {'local':>8s} {'global':>8s} {'MW p':>8s} {'AUC(sep global)':>16s}  label-free?")
for m, direction, lf in [("axis_drop", "greater", "no"), ("hist_gain", "greater", "no"), ("norm_corr", "less", "YES"), ("kfrac", "less", "no")]:
    # direction = expected relation of GLOBAL vs local
    p = mannwhitneyu(gl[m], lo[m], alternative=direction).pvalue
    yv = np.r_[np.ones(len(gl)), np.zeros(len(lo))]; xv = np.r_[gl[m], lo[m]]
    auc = roc_auc_score(yv, xv if direction == "greater" else -xv)
    print(f"    {m:12s} {lo[m].mean():8.3f} {gl[m].mean():8.3f} {p:8.4f} {auc:16.3f}  {lf}")
print("\nsaved CHARACTERIZE_GLOBAL.csv")
