# -*- coding: utf-8 -*-
"""Corrected TRIVIAL detector: per-feature CALIBRATED OR. Each feature scored by its own rarity
(empirical two-sided ECDF tail, and histogram density), flag on the single most-rare feature
(max_j = 'at least one per-feature threshold met'). Compare its ap_norm to the oracle detector,
base-rate matched, and count how many datasets a triviality filter would drop, by family."""
import os, sys, io, zipfile, re, warnings
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
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
    if len(ni) > 6000: ni = r.choice(ni, 6000, replace=False)
    idx = np.arange(len(ni)); r.shuffle(idx); ni = ni[idx]
    return X[ni[:int(0.6 * len(ni))]], X[ni[int(0.6 * len(ni)):]], X[ai]


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


def apn(y, s):
    b = y.mean(); return (average_precision_score(y, s) - b) / (1 - b + 1e-12)


def tail_rarity(Xtr_col, q):
    """-log two-sided empirical tail prob of q under train column (per-feature calibrated)."""
    xs = np.sort(Xtr_col); n = len(xs); fb = np.searchsorted(xs, q, side="right") / n
    tail = 2 * np.minimum(fb, 1 - fb); return -np.log(np.maximum(tail, 1.0 / (2 * n)))


def hist_rarity(Xtr_col, q, bins=50):
    cnt, edges = np.histogram(Xtr_col, bins=bins); dens = cnt / max(cnt.sum(), 1) + 1e-9
    b = np.clip(np.digitize(q, edges[1:-1]), 0, bins - 1); return -np.log(dens[b])


meta = {}
for c, f in CSVMAP.items():
    d = pd.read_csv(os.path.join(S, f)); d["ap_norm"] = (d.ap - d.base_rate) / (1 - d.base_rate)
    br = d.groupby("dataset").base_rate.mean(); ba = d.groupby(["dataset", "variant"]).ap_norm.mean().groupby("dataset").max()
    for ds in br.index: meta[ds] = (br[ds], ba.get(ds, np.nan))
mp = pd.read_csv(os.path.join(S, "MODALITY_PERF.csv")); mp["truefam"] = np.where(mp.best_local > mp.best_global, "local", "global")
rows = []
for _, r in mp.iterrows():
    if r.dataset not in meta: continue
    br, best_apn = meta[r.dataset]
    try:
        if r.corpus in ("oddbench", "ovrbench"): Xtr, Xn, Xa = tab_load(r.corpus, r.dataset)
        elif r.corpus == "ucr": Xtr, Xn, Xa = uni_load(r.dataset, "ucr/UCR_TimeSeriesAnomalyDatasets2021.zip")
        else: Xtr, Xn, Xa = uni_load(r.dataset, "tsbad/TSB-AD-U.zip")
    except Exception: continue
    if len(Xtr) < 40 or len(Xn) < 20 or len(Xa) < 5 or br != br: continue
    rng = np.random.default_rng(1); na = max(3, int(round(br / (1 - br) * len(Xn))))
    Xa2 = Xa[rng.choice(len(Xa), na, replace=False)] if len(Xa) > na else Xa
    d = Xtr.shape[1]; Xe = np.vstack([Xn, Xa2]); y = np.r_[np.zeros(len(Xn)), np.ones(len(Xa2))]
    tail_max = np.full(len(Xe), -np.inf); hist_max = np.full(len(Xe), -np.inf)
    for j in range(d):
        tr = Xtr[:, j]
        if np.std(tr) < 1e-9: continue
        tail_max = np.maximum(tail_max, tail_rarity(tr, Xe[:, j]))
        hist_max = np.maximum(hist_max, hist_rarity(tr, Xe[:, j]))
    or_tail = apn(y, tail_max); or_hist = apn(y, hist_max)
    rows.append({"dataset": r.dataset, "truefam": r.truefam, "det_apnorm": best_apn, "best_global": r.best_global,
                 "or_tail": or_tail, "or_hist": or_hist, "or_best": max(or_tail, or_hist)})
df = pd.DataFrame(rows).dropna(); df["mv_gain"] = df.det_apnorm - df.or_best; df.to_csv(os.path.join(S, "CALIBRATED_OR.csv"), index=False)
lo, gl = df[df.truefam == "local"], df[df.truefam == "global"]
print(f"=== per-feature CALIBRATED OR trivial detector ({len(df)}: {len(lo)} local / {len(gl)} global) ===")
print(f"    {'family':12s} {'n':>4s} {'OR_tail':>8s} {'OR_hist':>8s} {'OR_best':>8s} {'best_det':>9s} {'best_global':>12s} {'mv_gain':>8s}")
for f in ["local", "global"]:
    g = df[df.truefam == f]
    print(f"    {f+'-wins':12s} {len(g):4d} {g.or_tail.mean():8.3f} {g.or_hist.mean():8.3f} {g.or_best.mean():8.3f} {g.det_apnorm.mean():9.3f} {g.best_global.mean():12.3f} {g.mv_gain.mean():8.3f}")
print(f"\n    Does the calibrated OR match the fitted global detector? corr(OR_best, best_global)={df.or_best.corr(df.best_global):.2f}")
print(f"\n    FILTER: drop where trivial OR reaches within eps of the ORACLE (mv_gain < eps):")
for eps in [0.03, 0.05, 0.08]:
    drop = df.mv_gain < eps
    dl = ((df.truefam == "local") & drop).sum(); dg = ((df.truefam == "global") & drop).sum()
    print(f"      eps={eps}:  drop {drop.sum():3d}/{len(df)}  (local {dl}/{len(lo)}, global {dg}/{len(gl)})  "
          f"-> remaining {len(df)-drop.sum()} = {len(lo)-dl} local + {len(gl)-dg} global")
print("\nsaved CALIBRATED_OR.csv")
