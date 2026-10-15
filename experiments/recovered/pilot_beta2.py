# -*- coding: utf-8 -*-
"""Shape fingerprint + beta-router pilot.

SHAPE: does a detector's separation at TYPICAL beta (beta=1, multivariate synth) vs EXTREME
(beta=-4, marginal synth) fingerprint its family? Local detectors should stay good at beta=1
(they separate multivariate synthetics); global should collapse there and only recover at
beta=-4.  localness_d = AUC_d(beta=1);  shape_gap_d = AUC_d(-4) - AUC_d(1).

ROUTER: use the beta-profile to route selection per dataset. Heuristics compared to fixed-beta
and EM:
  max_spread   pick the beta whose detector-AUCs spread most (most discriminative), argmax there
  localness    if the pool contains a strongly-local detector (high AUC(1) and small gap) the
               dataset is local -> select at beta=1 (typical); else -> beta=-2 (extreme)
  profile_top  rank each detector by its MAX AUC over beta (best-case separation) - a shape summary
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


def gen_beta(Xn, ns, beta, frac=0.4, cap=98, seed=0):
    rng = np.random.default_rng(seed); n, d = Xn.shape; H = []
    for j in range(d):
        col = Xn[:, j]; cnt, edges = np.histogram(col, bins=30); dens = cnt / max(cnt.sum(), 1)
        lo, hi = np.percentile(col, 100 - cap), np.percentile(col, cap); ctr = (edges[:-1] + edges[1:]) / 2
        allowed = (cnt > 0) & (ctr >= lo) & (ctr <= hi); w = np.where(allowed, np.maximum(dens, 1e-6) ** beta, 0.0)
        H.append((edges, w / w.sum() if w.sum() > 0 else None))
    out = np.empty((ns, d))
    for r in range(ns):
        base = Xn[rng.integers(n)].copy(); cols = rng.choice(d, max(1, int(frac * d)), replace=False)
        for j in cols:
            edges, w = H[j]
            base[j] = Xn[rng.integers(n), j] if w is None else rng.uniform(*edges[[(bb := rng.choice(len(w), p=w)), bb + 1]])
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
picks = []
for (corp, f), n in [(("oddbench", "local"), 4), (("ovrbench", "local"), 3), (("oddbench", "global"), 4),
                     (("ovrbench", "global"), 3), (("ucr", "local"), 4), (("tsbad_u", "local"), 2)]:
    picks.append(inc[(inc.corpus == corp) & (inc.f == f)].sort_values("spread_ap_norm", ascending=False).head(n))
SAMP = pd.concat(picks)

shape = {"local": [], "global": []}
reg = {"em": [], "beta1": [], "beta_ext": [], "max_spread": [], "localness": [], "profile_top": []}
for _, r in SAMP.iterrows():
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
    common = [v for v, _ in pool if v in ap.index and v in emv.index and emv[v] == emv[v] and all(v in mat[b] for b in BETAS)]
    if len(common) < 6: continue
    av = pd.Series({v: ap[v] for v in common}); best = av.max(); ev = pd.Series({v: emv[v] for v in common})
    # SHAPE fingerprint per detector
    for v in common:
        shape[fam(v)].append((mat[1.0][v], mat[-4.0][v])) if fam(v) in shape else None
    # selectors
    def pick(scored): return best - av[max(scored, key=scored.get)]
    reg["em"].append(best - av[ev.idxmax()])
    reg["beta1"].append(pick({v: mat[1.0][v] for v in common}))
    reg["beta_ext"].append(pick({v: mat[-2.0][v] for v in common}))
    # max_spread: use beta with highest detector-AUC std
    stds = {b: np.std([mat[b][v] for v in common]) for b in BETAS}
    bs = max(stds, key=stds.get); reg["max_spread"].append(pick({v: mat[bs][v] for v in common}))
    # localness gate: strong local detector present (some detector AUC(1) high AND gap small)?
    loc_signal = max((mat[1.0][v] - (mat[-4.0][v] - mat[1.0][v])) for v in common)  # high AUC1, small gap
    bgate = 1.0 if loc_signal > 0.6 else -2.0
    reg["localness"].append(pick({v: mat[bgate][v] for v in common}))
    # profile_top: rank by max AUC over beta (best-case separation)
    reg["profile_top"].append(pick({v: max(mat[b][v] for b in BETAS) for v in common}))

print(f"=== SHAPE fingerprint: detector separation at typical (beta=1) vs extreme (beta=-4) ===")
for f in ["local", "global"]:
    a = np.array(shape[f]); print(f"   {f:7s} AUC(beta=1) {a[:,0].mean():.3f}   AUC(beta=-4) {a[:,1].mean():.3f}   gap {(a[:,1]-a[:,0]).mean():+.3f}  (n={len(a)})")
print("   (local should keep higher AUC at beta=1; global should have a larger extreme-vs-typical gap)")
print(f"\n=== ROUTER: regret vs EM ({len(reg['em'])} datasets) ===")
e = np.array(reg["em"])
for k in ["max_spread", "localness", "profile_top", "beta1", "beta_ext", "em"]:
    a = np.array(reg[k])
    if k == "em": print(f"   {k:12s} {a.mean():.4f}"); continue
    p = wilcoxon(a, e).pvalue if (a != e).any() else np.nan
    print(f"   {k:12s} {a.mean():.4f}  (vs EM p={p:.3f}, wins {int((a<e).sum())}/{len(a)})")
