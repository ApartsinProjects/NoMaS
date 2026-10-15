# -*- coding: utf-8 -*-
"""Definitive construct-matched leaderboard on the CURRENT 162-dataset benchmark.
UOMS criteria (em, mv, consensus, model_centrality, hits) read from the arm CSVs (validation-side,
3-way protocol); our synthetic picks (beta=1, beta=-4, matched) recomputed in the same pass; every
method picks from the SAME common pool per dataset and is scored by regret on test ap_norm."""
import os, sys, io, contextlib, zipfile, re, warnings
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"; S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
sys.path.insert(0, os.path.join(S, "scratchpad"))
from dev_common import TAB_POOL, TS_POOL
from hadb_ts_final import W, STRIDE, MAX_LEN, block_split3
from adrank.ts import _window_features, _window_labels
LOCAL = ("LOF", "KNN", "CBLOF"); GLOBAL = ("HBOS", "COPOD", "ECOD", "PCA")
fam = lambda v: "local" if str(v).startswith(LOCAL) else ("global" if str(v).startswith(GLOBAL) else "other")
CSVMAP = {"oddbench": "hadb_oddbench.csv", "ovrbench": "hadb_ovrbench.csv", "ucr": "hadb_ts_ucr.csv", "tsbad_u": "hadb_ts_tsbad.csv"}
CRIT = ["em", "mv", "consensus", "model_centrality", "hits"]; TAU = -0.140


def tab_tv(corpus, name):
    d = np.load(os.path.join(ROOT, "data", corpus, name + ".npz"), allow_pickle=True)
    X = np.vstack([np.asarray(d["train"], float), np.asarray(d["test"], float)]); y = np.concatenate([np.asarray(d["train_labels"]).ravel(), np.asarray(d["test_labels"]).ravel()]).astype(int); X = np.nan_to_num(X)
    if len(X) > 5000:
        r = np.random.RandomState(0); k = r.choice(len(X), 5000, replace=False); X, y = X[k], y[k]
    ni = np.where(y == 0)[0]; g = np.random.default_rng(0); idx = np.arange(len(ni)); g.shuffle(idx)
    return X[ni[idx[:int(0.6 * len(idx))]]], X[ni[idx[int(0.6 * len(idx)):int(0.8 * len(idx))]]]


def uni_tv(name, zipname):
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
    pos = np.arange(len(Xw)); tr, va, te = block_split3(yw, pos, 0); return Xw[tr], Xw[va]


def _H(Xn, cap=98):
    H = []
    for j in range(Xn.shape[1]):
        col = Xn[:, j]; cnt, edges = np.histogram(col, bins=30); dens = cnt / max(cnt.sum(), 1)
        lo, hi = np.percentile(col, 100 - cap), np.percentile(col, cap); ctr = (edges[:-1] + edges[1:]) / 2
        H.append((edges, dens, (cnt > 0) & (ctr >= lo) & (ctr <= hi)))
    return H


def gen_beta(Xn, ns, beta, frac=0.4, seed=0):
    rng = np.random.default_rng(seed); n, d = Xn.shape; H = _H(Xn); out = np.empty((ns, d))
    for r in range(ns):
        base = Xn[rng.integers(n)].copy()
        for j in rng.choice(d, max(1, int(frac * d)), replace=False):
            edges, dens, allowed = H[j]; w = np.where(allowed, np.maximum(dens, 1e-6) ** beta, 0.0)
            base[j] = Xn[rng.integers(n), j] if w.sum() == 0 else rng.uniform(*edges[[(b := rng.choice(len(w), p=w / w.sum())), b + 1]])
        out[r] = base
    return out


def sauc(Xtr, Xval, syn, pool):
    Xe = np.vstack([Xval, syn]); ye = np.r_[np.zeros(len(Xval)), np.ones(len(syn))]; o = {}
    for vn, ct in pool:
        try:
            m = ct()
            with contextlib.redirect_stdout(io.StringIO()): m.fit(Xtr)
            s = np.asarray(m.decision_function(Xe), float)
            if np.all(np.isfinite(s)) and np.nanstd(s) > 1e-12: o[vn] = float(roc_auc_score(ye, s))
        except Exception: pass
    return o


# per-dataset per-variant table from CSVs: ap_norm + criteria
csv = {c: pd.read_csv(os.path.join(S, f)) for c, f in CSVMAP.items()}
for c in csv:
    csv[c]["ap_norm"] = (csv[c].ap - csv[c].base_rate) / (1 - csv[c].base_rate)
mp = pd.read_csv(os.path.join(S, "MODALITY_PERF.csv"))[["corpus", "dataset"]]
def dstab(corpus, name):
    d = csv[corpus]; d = d[d.dataset == name]
    return d.groupby("variant").agg({"ap_norm": "mean", **{k: "mean" for k in CRIT}})
# global best-fixed detector = best mean ap_norm across the 162
acc = {}
for _, r in mp.iterrows():
    t = dstab(r.corpus, r.dataset)
    for v, a in t.ap_norm.items(): acc.setdefault(v, []).append(a)
fixed_det = max(acc, key=lambda v: np.mean(acc[v])); print("best-fixed detector:", fixed_det, flush=True)

rows = []
for i, (_, r) in enumerate(mp.iterrows()):
    try:
        if r.corpus in ("oddbench", "ovrbench"): Xtr, Xval = tab_tv(r.corpus, r.dataset); pool = TAB_POOL
        elif r.corpus == "ucr": Xtr, Xval = uni_tv(r.dataset, "ucr/UCR_TimeSeriesAnomalyDatasets2021.zip"); pool = TS_POOL
        else: Xtr, Xval = uni_tv(r.dataset, "tsbad/TSB-AD-U.zip"); pool = TS_POOL
    except Exception: continue
    if len(Xtr) < 40 or len(Xval) < 30: continue
    t = dstab(r.corpus, r.dataset)
    a1 = sauc(Xtr, Xval, gen_beta(Xval, 200, 1.0), pool); a4 = sauc(Xtr, Xval, gen_beta(Xval, 200, -4.0), pool)
    common = [v for v in t.index if v in a1 and v in a4 and t.loc[v, "ap_norm"] == t.loc[v, "ap_norm"]
              and all(t.loc[v, c] == t.loc[v, c] for c in CRIT)]
    if len(common) < 6: continue
    ap = t.loc[common, "ap_norm"]; best = ap.max()
    loc = [ap[v] for v in common if fam(v) == "local"]; glo = [ap[v] for v in common if fam(v) == "global"]
    if not loc or not glo: continue
    local_ev = max(a1[v] for v in common) - max(a4[v] for v in common)
    p_beta1 = max(common, key=lambda v: a1[v]); p_ext = max(common, key=lambda v: a4[v])
    p_matched = p_beta1 if local_ev > TAU else p_ext
    pick = {"em": t.loc[common, "em"].idxmax(), "mv": t.loc[common, "mv"].idxmin(),
            "consensus": t.loc[common, "consensus"].idxmax(), "model_centrality": t.loc[common, "model_centrality"].idxmax(),
            "hits": t.loc[common, "hits"].idxmax(), "beta1": p_beta1, "betaext": p_ext, "matched": p_matched}
    ifs = [v for v in common if str(v).startswith(("IForest", "IF_"))]
    rec = {"dataset": r.dataset, "truefam": "local" if max(loc) > max(glo) else "global",
           "reg_random": best - ap.mean(), "reg_best_fixed": best - (ap[fixed_det] if fixed_det in ap.index else ap.mean()),
           "reg_iforest_random": best - np.mean([ap[v] for v in ifs]) if ifs else np.nan}
    for k, v in pick.items(): rec["reg_" + k] = best - ap[v]
    rows.append(rec)
    if i % 20 == 0: print(f"  ..{i}/{len(mp)}  {len(rows)} kept", flush=True)
df = pd.DataFrame(rows); df.to_csv(os.path.join(S, "FULL_LEADERBOARD.csv"), index=False)

def macro(col): return np.mean([df[df.truefam == f][col].mean() for f in ["local", "global"] if (df.truefam == f).any()])
NAMES = [("oracle", None), ("NoMaS matched (ours)", "reg_matched"), ("NoMaS beta=1 (ours)", "reg_beta1"),
         ("best_fixed %s (cheats)" % fixed_det, "reg_best_fixed"), ("EM", "reg_em"), ("MV", "reg_mv"),
         ("beta=-4 extreme", "reg_betaext"), ("consensus/UDR", "reg_consensus"),
         ("ModelCentrality", "reg_model_centrality"), ("HITS", "reg_hits"),
         ("iforest_random", "reg_iforest_random"), ("random", "reg_random")]
e = df.reg_em.values
print(f"\n=== FULL construct-matched leaderboard ({len(df)} datasets: "
      f"{int((df.truefam=='local').sum())} local / {int((df.truefam=='global').sum())} global) ===")
print(f"    {'method':34s} {'micro':>7s} {'macro':>7s} {'vs EM p':>9s} {'vs rand p':>10s}")
res = []
for name, col in NAMES:
    if col is None: print(f"    {name:34s} {0.0:7.3f} {0.0:7.3f}"); continue
    a = df[col].values; ok = ~np.isnan(a)
    pe = wilcoxon(a[ok], e[ok]).pvalue if col != "reg_em" and (a[ok] != e[ok]).any() else np.nan
    pr = wilcoxon(a[ok], df.reg_random.values[ok]).pvalue if col != "reg_random" and (a[ok] != df.reg_random.values[ok]).any() else np.nan
    print(f"    {name:34s} {np.nanmean(a):7.3f} {macro(col):7.3f} {pe:9.3f} {pr:10.3f}")
    res.append((name, round(np.nanmean(a), 4), round(macro(col), 4)))
pd.DataFrame(res, columns=["method", "micro", "macro"]).to_csv(os.path.join(S, "FULL_LEADERBOARD_SUMMARY.csv"), index=False)
print("\nsaved FULL_LEADERBOARD.csv / FULL_LEADERBOARD_SUMMARY.csv")
