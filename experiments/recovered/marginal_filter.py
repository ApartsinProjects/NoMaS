# -*- coding: utf-8 -*-
"""Marginal-triviality filter: does an AXIS-ALIGNED marginal rule (OR of per-feature thresholds, or
HBOS-sum) already achieve near-oracle ap_norm? If so the dataset needs no multivariate modeling ->
drop it. Verify it removes global-wins (marginal-solvable) while keeping local-wins (marginal fails).
All ap_norm base-rate-matched to the benchmark."""
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
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9; d = Xtr.shape[1]
    Zn = np.abs((Xn - mu) / sd); Za = np.abs((Xa2 - mu) / sd); y = np.r_[np.zeros(len(Zn)), np.ones(len(Za))]
    # OR of thresholds = max per-feature |z|;  HBOS-sum = sum of -log hist density
    m_or = apn(y, np.r_[Zn.max(1), Za.max(1)])
    Ztr = np.abs((Xtr - mu) / sd); hb_n = np.zeros(len(Zn)); hb_a = np.zeros(len(Za))
    for j in range(d):
        cnt, edges = np.histogram((Xtr[:, j] - mu[j]) / sd[j], bins=30); dens = cnt / max(cnt.sum(), 1) + 1e-9
        def sc(v): b = np.clip(np.digitize((v), edges[1:-1]), 0, 29); return -np.log(dens[b])
        hb_n += sc((Xn[:, j] - mu[j]) / sd[j]); hb_a += sc((Xa2[:, j] - mu[j]) / sd[j])
    m_hbos = apn(y, np.r_[hb_n, hb_a])
    rows.append({"dataset": r.dataset, "truefam": r.truefam, "det_apnorm": best_apn,
                 "marg_or_apnorm": m_or, "marg_hbos_apnorm": m_hbos,
                 "marg_best": max(m_or, m_hbos)})
df = pd.DataFrame(rows).dropna(); df["mv_gain"] = df.det_apnorm - df.marg_best; df.to_csv(os.path.join(S, "MARGINAL_FILTER.csv"), index=False)
lo, gl = df[df.truefam == "local"], df[df.truefam == "global"]
print(f"=== marginal (axis-aligned) rule ap_norm vs oracle ({len(df)}: {len(lo)} local / {len(gl)} global) ===")
print(f"    {'family':12s} {'n':>4s} {'OR_ap':>7s} {'HBOS_ap':>8s} {'best_det':>9s} {'multivar_gain':>14s}")
for f in ["local", "global"]:
    g = df[df.truefam == f]
    print(f"    {f+'-wins':12s} {len(g):4d} {g.marg_or_apnorm.mean():7.3f} {g.marg_hbos_apnorm.mean():8.3f} {g.det_apnorm.mean():9.3f} {g.mv_gain.mean():14.3f}")
print(f"\n    FILTER: drop datasets where marginal rule reaches within eps of the oracle (multivar_gain < eps)")
for eps in [0.03, 0.05, 0.08]:
    drop = df.mv_gain < eps
    dl = ((df.truefam == "local") & drop).sum(); dg = ((df.truefam == "global") & drop).sum()
    print(f"      eps={eps}:  drop {drop.sum():3d}/{len(df)}  (local {dl}/{len(lo)}, global {dg}/{len(gl)})  "
          f"-> remaining {len(df)-drop.sum()} = {len(lo)-dl} local + {len(gl)-dg} global")
print("\nsaved MARGINAL_FILTER.csv")
