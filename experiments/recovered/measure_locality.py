# -*- coding: utf-8 -*-
"""Measure each detector's EFFECTIVE locality per dataset (black-box), instead of a fixed
name-based family. A detector's locality on a dataset = how much better it separates beta=+1
synthetics (typical/multivariate = local structure) than beta=-4 synthetics (marginal tails).

Q1  Data-dependence: does a detector's measured locality vary across datasets / flip sign?
    If yes, the static name label is lossy (user's premise).
Q2  A MEASURED dataset descriptor local_evidence = max_d AUC_d(b=+1) - max_d AUC_d(b=-4):
    does it predict which family truly wins better than the static multimodality scalar (0.62)?
"""
import os, sys, io, contextlib, zipfile, re, warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"; S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
sys.path.insert(0, os.path.join(S, "scratchpad"))
from dev_common import TAB_POOL, TS_POOL, true_apnorm
from hadb_ts_final import W, STRIDE, MAX_LEN, block_split3
from adrank.ts import _window_features, _window_labels
LOCAL = ("LOF", "KNN", "CBLOF"); GLOBAL = ("HBOS", "COPOD", "ECOD", "PCA")
fam = lambda v: "local" if str(v).startswith(LOCAL) else ("global" if str(v).startswith(GLOBAL) else "other")
CSVMAP = {"oddbench": "hadb_oddbench.csv", "ovrbench": "hadb_ovrbench.csv", "ucr": "hadb_ts_ucr.csv", "tsbad_u": "hadb_ts_tsbad.csv"}
BETAS = [1.0, -4.0]


def find_npz(corpus, name):
    p = os.path.join(ROOT, "data", corpus, name + ".npz"); return p if os.path.exists(p) else None


def tab_tv(corpus, name):
    d = np.load(find_npz(corpus, name), allow_pickle=True)
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
        allowed = (cnt > 0) & (ctr >= lo) & (ctr <= hi); H.append((edges, dens, allowed))
    return H


def gen_beta(Xn, ns, beta, frac=0.4, seed=0):
    rng = np.random.default_rng(seed); n, d = Xn.shape; H = _H(Xn); out = np.empty((ns, d))
    for r in range(ns):
        base = Xn[rng.integers(n)].copy()
        for j in rng.choice(d, max(1, int(frac * d)), replace=False):
            edges, dens, allowed = H[j]; w = np.where(allowed, np.maximum(dens, 1e-6) ** beta, 0.0)
            if w.sum() == 0: base[j] = Xn[rng.integers(n), j]
            else: b = rng.choice(len(w), p=w / w.sum()); base[j] = rng.uniform(edges[b], edges[b + 1])
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


M = pd.read_csv(os.path.join(S, "HADB_MANIFEST.csv")); inc = M[M.include].copy()
samp = pd.concat([g.sort_values("spread_ap_norm").iloc[np.linspace(0, len(g) - 1, min(13, len(g))).astype(int)]
                  for c, g in inc.groupby("corpus") if c in CSVMAP])
det_mloc = {}      # detector name -> list of measured localness across datasets
ds_rows = []
for _, r in samp.iterrows():
    corpus = r.corpus
    try:
        if corpus in ("oddbench", "ovrbench"): Xtr, Xval = tab_tv(corpus, r.dataset); pool = TAB_POOL
        elif corpus == "ucr": Xtr, Xval = uni_tv(r.dataset, "ucr/UCR_TimeSeriesAnomalyDatasets2021.zip"); pool = TS_POOL
        elif corpus == "tsbad_u": Xtr, Xval = uni_tv(r.dataset, "tsbad/TSB-AD-U.zip"); pool = TS_POOL
        else: continue
    except Exception: continue
    if len(Xtr) < 40 or len(Xval) < 30: continue
    try: ap, _ = true_apnorm(CSVMAP[corpus], r.dataset)
    except Exception: continue
    a1 = sauc(Xtr, Xval, gen_beta(Xval, 150, 1.0), pool); a4 = sauc(Xtr, Xval, gen_beta(Xval, 150, -4.0), pool)
    common = [v for v, _ in pool if v in a1 and v in a4 and v in ap.index and ap[v] == ap[v]]
    if len(common) < 6: continue
    for v in common:
        det_mloc.setdefault(v, []).append(a1[v] - a4[v])   # >0 local-behaving here, <0 global-behaving
    loc = [ap[v] for v in common if fam(v) == "local"]; glo = [ap[v] for v in common if fam(v) == "global"]
    if not loc or not glo: continue
    # measured dataset descriptor: which probe is more separable overall
    local_ev = max(a1.values()) - max(a4.values())
    ds_rows.append({"corpus": corpus, "dataset": r.dataset, "local_ev": local_ev,
                    "gap": max(loc) - max(glo), "local_wins": int(max(loc) > max(glo))})

print(f"=== Q1: is a detector's MEASURED locality data-dependent? (AUC(b=+1)-AUC(b=-4) per dataset) ===")
print("   detector           name-family   mean_mloc   std    frac-flip-vs-name  n")
for v in sorted(det_mloc, key=lambda k: -len(det_mloc[k])):
    a = np.array(det_mloc[v]); nf = fam(v)
    # a name-'local' detector is 'behaving against its label' when mloc<=0; a 'global' when mloc>0
    flip = np.mean(a <= 0) if nf == "local" else (np.mean(a > 0) if nf == "global" else np.nan)
    if len(a) >= 8:
        print(f"   {v[:18]:18s} {nf:9s}    {a.mean():+.3f}    {a.std():.3f}   {flip:.2f}   {len(a)}")

df = pd.DataFrame(ds_rows)
print(f"\n=== Q2: MEASURED dataset descriptor vs static multimodality (n={len(df)}) ===")
auc = roc_auc_score(df.local_wins, df.local_ev); rho = spearmanr(df.local_ev, df.gap).statistic
print(f"   local_ev = max AUC(b=+1) - max AUC(b=-4)")
print(f"   Spearman(local_ev, local-global gap) = {rho:+.3f}   AUC(predict local-wins) = {auc:.3f}")
print(f"   (compare: best static multimodality scalar silhouette AUC 0.62)")
print("   means by winning family:")
for w, g in df.groupby("local_wins"):
    print(f"     {'local-wins' if w else 'global-wins':11s} n={len(g):2d}  local_ev {g.local_ev.mean():+.3f}")
df.to_csv(os.path.join(S, "MEASURE_LOCALITY.csv"), index=False)
