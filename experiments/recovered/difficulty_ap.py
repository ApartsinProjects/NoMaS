# -*- coding: utf-8 -*-
"""Re-check 'global = single-feature-easy' under the RANKING metric (ap_norm), not AUC.
Also report recall@10%FPR. Base-rate matched to the benchmark so single-feature ap_norm is
comparable to the CSV best-detector ap_norm."""
import os, sys, io, zipfile, re, warnings
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score, average_precision_score
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
    return X[ni[:int(0.6 * len(ni))]], X[ni[int(0.6 * len(ni)):]], X[ai]   # train, held-out normals(40%), anomalies


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
def recall_at_fpr(y, s, fpr=0.10):
    ns = s[y == 0]; thr = np.quantile(ns, 1 - fpr); return float((s[y == 1] > thr).mean())


# benchmark base rate + best-detector ap_norm/auc per dataset
meta = {}
for c, f in CSVMAP.items():
    d = pd.read_csv(os.path.join(S, f)); d["ap_norm"] = (d.ap - d.base_rate) / (1 - d.base_rate)
    br = d.groupby("dataset").base_rate.mean()
    ba = d.groupby(["dataset", "variant"]).ap_norm.mean().groupby("dataset").max()
    au = d.groupby(["dataset", "variant"]).auc.mean().groupby("dataset").max()
    for ds in br.index: meta[ds] = (br[ds], ba.get(ds, np.nan), au.get(ds, np.nan))
mp = pd.read_csv(os.path.join(S, "MODALITY_PERF.csv")); mp["truefam"] = np.where(mp.best_local > mp.best_global, "local", "global")
rows = []
for _, r in mp.iterrows():
    if r.dataset not in meta: continue
    br, best_apn, best_auc = meta[r.dataset]
    try:
        if r.corpus in ("oddbench", "ovrbench"): Xtr, Xn, Xa = tab_load(r.corpus, r.dataset)
        elif r.corpus == "ucr": Xtr, Xn, Xa = uni_load(r.dataset, "ucr/UCR_TimeSeriesAnomalyDatasets2021.zip")
        else: Xtr, Xn, Xa = uni_load(r.dataset, "tsbad/TSB-AD-U.zip")
    except Exception: continue
    if len(Xtr) < 40 or len(Xn) < 20 or len(Xa) < 5 or br != br: continue
    # base-rate match: n_anom = br/(1-br)*n_norm
    rng = np.random.default_rng(1); na_target = max(3, int(round(br / (1 - br) * len(Xn))))
    if len(Xa) > na_target: Xa2 = Xa[rng.choice(len(Xa), na_target, replace=False)]
    else:
        Xa2 = Xa; nn = min(len(Xn), max(10, int(round((1 - br) / br * len(Xa)))))
        Xn = Xn[rng.choice(len(Xn), nn, replace=False)] if len(Xn) > nn else Xn
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9; Zn = np.abs((Xn - mu) / sd); Za = np.abs((Xa2 - mu) / sd)
    y = np.r_[np.zeros(len(Zn)), np.ones(len(Za))]
    b_ap = b_au = b_rc = 0.0
    for j in range(Zn.shape[1]):
        sc = np.r_[Zn[:, j], Za[:, j]]
        if np.std(sc) > 1e-9:
            b_ap = max(b_ap, apn(y, sc)); b_au = max(b_au, roc_auc_score(y, sc)); b_rc = max(b_rc, recall_at_fpr(y, sc))
    rows.append({"dataset": r.dataset, "truefam": r.truefam, "base_rate": br,
                 "f1_apnorm": b_ap, "det_apnorm": best_apn, "gap_ap": best_apn - b_ap,
                 "f1_auc": b_au, "det_auc": best_auc, "gap_auc": best_auc - b_au, "f1_recall10": b_rc})
df = pd.DataFrame(rows).dropna(); df.to_csv(os.path.join(S, "DIFFICULTY_AP.csv"), index=False)
lo, gl = df[df.truefam == "local"], df[df.truefam == "global"]
print(f"=== single-feature difficulty under AP vs AUC ({len(df)}: {len(lo)} local / {len(gl)} global) ===")
print(f"    {'family':12s} {'n':>4s} | {'1feat_AP':>9s} {'detAP':>7s} {'gapAP':>7s} | {'1feat_AUC':>10s} {'detAUC':>7s} {'gapAUC':>7s} | {'1f_rec@10':>9s}")
for f in ["local", "global"]:
    g = df[df.truefam == f]
    print(f"    {f+'-wins':12s} {len(g):4d} | {g.f1_apnorm.mean():9.3f} {g.det_apnorm.mean():7.3f} {g.gap_ap.mean():7.3f} | "
          f"{g.f1_auc.mean():10.3f} {g.det_auc.mean():7.3f} {g.gap_auc.mean():7.3f} | {g.f1_recall10.mean():9.3f}")
print(f"\n    Mann-Whitney gap_AP  (local > global?): p={mannwhitneyu(lo.gap_ap, gl.gap_ap, alternative='greater').pvalue:.4f}")
print(f"    Mann-Whitney gap_AUC (local > global?): p={mannwhitneyu(lo.gap_auc, gl.gap_auc, alternative='greater').pvalue:.4f}")
print(f"    frac where 1-feature within 0.05 of best detector:  AP-> local {np.mean(lo.gap_ap<0.05):.2f} global {np.mean(gl.gap_ap<0.05):.2f}"
      f"   |  AUC-> local {np.mean(lo.gap_auc<0.05):.2f} global {np.mean(gl.gap_auc<0.05):.2f}")
print("\nsaved DIFFICULTY_AP.csv")
