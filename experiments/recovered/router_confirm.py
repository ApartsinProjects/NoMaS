# -*- coding: utf-8 -*-
"""Representative confirmation: max-spread beta-router vs mixed generator vs fixed beta vs EM,
on a NATURAL-mix sample of the finalized benchmark (evenly spaced by spread per corpus, not the
curated global-heavy pilot sample). Reports regret by family and the head-to-head Wilcoxon tests.
"""
import os, sys, io, contextlib, zipfile, re, warnings
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
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
BETAS = [1.0, 0.0, -2.0, -4.0]


def find_npz(corpus, name):
    for sub in {"oddbench": ["oddbench"], "ovrbench": ["ovrbench"]}.get(corpus, [corpus]):
        p = os.path.join(ROOT, "data", sub, name + ".npz")
        if os.path.exists(p):
            return p


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
    st = np.arange(0, len(x) - W + 1, STRIDE); Xw, _ = _window_features(x, w=W, stride=STRIDE); yw = _window_labels(lab, st, w=W, min_count=1); Xw = np.nan_to_num(Xw)
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


def gen_mixed(Xn, ns, frac=0.4, seed=0):
    rng = np.random.default_rng(seed); n, d = Xn.shape; H = _H(Xn); out = np.empty((ns, d))
    for r in range(ns):
        base = Xn[rng.integers(n)].copy()
        for i, j in enumerate(rng.choice(d, max(1, int(frac * d)), replace=False)):
            if i % 2 == 0: base[j] = Xn[rng.integers(n), j]
            else:
                edges, dens, allowed = H[j]; w = np.where(allowed, np.maximum(dens, 1e-6) ** -2.0, 0.0)
                if w.sum() > 0: b = rng.choice(len(w), p=w / w.sum()); base[j] = rng.uniform(edges[b], edges[b + 1])
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


M = pd.read_csv(os.path.join(S, "HADB_MANIFEST.csv")); inc = M[M.include].copy(); inc["f"] = inc.gt_best.map(fam)
# natural-mix representative sample: evenly spaced by spread per corpus (tabular + uni-TS)
samp = pd.concat([g.sort_values("spread_ap_norm").iloc[np.linspace(0, len(g) - 1, min(14, len(g))).astype(int)]
                  for c, g in inc.groupby("corpus") if c in CSVMAP])
reg = {k: [] for k in ["router", "mixed", "beta1", "em"]}; byf = {k: {"local": [], "global": []} for k in ["router", "mixed", "em"]}
for _, r in samp.iterrows():
    corpus = r.corpus
    try:
        if corpus in ("oddbench", "ovrbench"): Xtr, Xval = tab_tv(corpus, r.dataset); pool = TAB_POOL
        elif corpus == "ucr": Xtr, Xval = uni_tv(r.dataset, "ucr/UCR_TimeSeriesAnomalyDatasets2021.zip"); pool = TS_POOL
        elif corpus == "tsbad_u": Xtr, Xval = uni_tv(r.dataset, "tsbad/TSB-AD-U.zip"); pool = TS_POOL
        else: continue
    except Exception: continue
    if len(Xtr) < 40 or len(Xval) < 30: continue
    try: ap, emv = true_apnorm(CSVMAP[corpus], r.dataset)
    except Exception: continue
    mat = {b: sauc(Xtr, Xval, gen_beta(Xval, 150, b), pool) for b in BETAS}
    mx = sauc(Xtr, Xval, gen_mixed(Xval, 150), pool)
    common = [v for v, _ in pool if v in ap.index and v in emv.index and emv[v] == emv[v] and v in mx and all(v in mat[b] for b in BETAS)]
    if len(common) < 6: continue
    av = pd.Series({v: ap[v] for v in common}); best = av.max(); ev = pd.Series({v: emv[v] for v in common}); tf = r.f
    pick = lambda sc: best - av[max(sc, key=sc.get)]
    stds = {b: np.std([mat[b][v] for v in common]) for b in BETAS}; bs = max(stds, key=stds.get)
    rr = {"router": pick({v: mat[bs][v] for v in common}), "mixed": pick({v: mx[v] for v in common}),
          "beta1": pick({v: mat[1.0][v] for v in common}), "em": best - av[ev.idxmax()]}
    for k in reg: reg[k].append(rr[k])
    for k in byf:
        if tf in byf[k]: byf[k][tf].append(rr[k])
print(f"=== REPRESENTATIVE confirmation ({len(reg['em'])} datasets, natural family mix) ===")
e = np.array(reg["em"]); b1 = np.array(reg["beta1"])
for k in ["router", "mixed", "beta1", "em"]:
    a = np.array(reg[k])
    line = f"   {k:8s} {a.mean():.4f}"
    if k != "em": line += f"  vs EM p={wilcoxon(a,e).pvalue:.3f} wins {int((a<e).sum())}/{len(a)}"
    if k in ("router", "mixed"): line += f"  |  vs beta1 p={wilcoxon(a,b1).pvalue:.3f}"
    print(line)
print("   by family (regret):")
for f in ["local", "global"]:
    print(f"     {f:7s}  router {np.mean(byf['router'][f]):.3f}  mixed {np.mean(byf['mixed'][f]):.3f}  em {np.mean(byf['em'][f]):.3f}  (n={len(byf['em'][f])})")
