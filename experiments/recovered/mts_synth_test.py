# -*- coding: utf-8 -*-
"""Mixed generator (shuffle + histogram low-prob injection) vs EM across ALL THREE modalities,
including multivariate time series with a CHANNEL-AWARE generator.

  tabular / univariate TS: feature-level shuffle + per-feature beta<0 low-prob injection.
  multivariate TS: features are 8 statistics per channel. Channel-block shuffle (swap a whole
    channel's 8-feature block from another window -> cross-channel rupture) + per-channel beta<0
    injection (push one channel's statistics into their empirical tails -> single-channel extreme).

Corrected protocol (fit detectors on TRAIN, score VALIDATION normals vs synth). Reports regret
vs EM by modality and by true-best family, on the finalized 199-dataset benchmark.
"""
import os, sys, io, zipfile, re, contextlib, warnings
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
from hadb_ts_mts import mts_window_features, window_labels as mts_wlab, MAX_CH
LOCAL = ("LOF", "KNN", "CBLOF"); GLOBAL = ("HBOS", "COPOD", "ECOD", "PCA")
fam = lambda v: "local" if str(v).startswith(LOCAL) else ("global" if str(v).startswith(GLOBAL) else "other")
CSVMAP = {"oddbench": "hadb_oddbench.csv", "ovrbench": "hadb_ovrbench.csv", "adbench_dami": "hadb_v3.csv",
          "ucr": "hadb_ts_ucr.csv", "tsbad_u": "hadb_ts_tsbad.csv", "tsbad_m": "hadb_ts_mts.csv"}
FPC = 8   # features per channel (MTS)


def find_npz(corpus, name):
    for sub in {"oddbench": ["oddbench"], "ovrbench": ["ovrbench"], "adbench_dami": ["adbench", "dami"]}.get(corpus, [corpus]):
        p = os.path.join(ROOT, "data", sub, name + ".npz")
        if os.path.exists(p):
            return p
    return None


def tab_tv(corpus, name):
    d = np.load(find_npz(corpus, name), allow_pickle=True)
    if "X" in d:
        X = np.asarray(d["X"], float); y = np.asarray(d["y"]).ravel().astype(int)
    else:
        X = np.vstack([np.asarray(d["train"], float), np.asarray(d["test"], float)]); y = np.concatenate([np.asarray(d["train_labels"]).ravel(), np.asarray(d["test_labels"]).ravel()]).astype(int)
    X = np.nan_to_num(X)
    if len(X) > 5000:
        r = np.random.RandomState(0); k = r.choice(len(X), 5000, replace=False); X, y = X[k], y[k]
    ni = np.where(y == 0)[0]; g = np.random.default_rng(0); idx = np.arange(len(ni)); g.shuffle(idx)
    return X[ni[idx[:int(0.6 * len(idx))]]], X[ni[idx[int(0.6 * len(idx)):int(0.8 * len(idx))]]], 0


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
    pos = np.arange(len(Xw)); tr, va, te = block_split3(yw, pos, 0); return Xw[tr], Xw[va], 0


def mts_tv(name):
    z = zipfile.ZipFile(os.path.join(ROOT, "data", "tsbad", "TSB-AD-M.zip"))
    cand = [n for n in z.namelist() if os.path.basename(n) == name] or [n for n in z.namelist() if n.lower().endswith(".csv") and name.split("_")[0] in n]
    df = pd.read_csv(io.BytesIO(z.read(cand[0]))); lc = [q for q in df.columns if q.lower() in ("label", "is_anomaly", "anomaly")]
    lab = df[lc[0]].to_numpy(int); Xc = np.nan_to_num(df.drop(columns=lc).to_numpy(float))
    if len(Xc) > MAX_LEN:
        a = np.where(lab == 1)[0]; c = int((a[0] + a[-1]) // 2) if len(a) else MAX_LEN // 2; lo = max(0, min(c - MAX_LEN // 2, len(Xc) - MAX_LEN)); Xc, lab = Xc[lo:lo + MAX_LEN], lab[lo:lo + MAX_LEN]
    if Xc.shape[1] > MAX_CH:
        Xc = Xc[:, np.argsort(-Xc.std(0))[:MAX_CH]]
    Xw, starts = mts_window_features(Xc); yw = mts_wlab(lab, starts); Xw = np.nan_to_num(Xw)
    nch = Xc.shape[1]
    pos = np.arange(len(Xw)); tr, va, te = block_split3(yw, pos, 0); return Xw[tr], Xw[va], nch


def _hists(Xn, B=30, cap=98):
    H = []
    for j in range(Xn.shape[1]):
        col = Xn[:, j]; cnt, edges = np.histogram(col, bins=B); dens = cnt / max(cnt.sum(), 1)
        lo, hi = np.percentile(col, 100 - cap), np.percentile(col, cap); centers = (edges[:-1] + edges[1:]) / 2
        allowed = (cnt > 0) & (centers >= lo) & (centers <= hi)
        H.append((edges, dens, allowed))
    return H


def inject(base, j, H, rng, beta=-2.0):
    edges, dens, allowed = H[j]
    w = np.where(allowed, np.maximum(dens, 1e-6) ** beta, 0.0)
    if w.sum() == 0:
        return
    b = rng.choice(len(w), p=w / w.sum()); base[j] = rng.uniform(edges[b], edges[b + 1])


def gen_mixed(Xn, ns, nch=0, frac=0.4, seed=0):
    rng = np.random.default_rng(seed); n, d = Xn.shape; H = _hists(Xn); out = np.empty((ns, d))
    if nch >= 2:                                  # MULTIVARIATE: operate on channel blocks
        for r in range(ns):
            base = Xn[rng.integers(n)].copy()
            k = max(1, int(frac * nch)); chans = rng.choice(nch, k, replace=False)
            for i, ch in enumerate(chans):
                sl = slice(ch * FPC, ch * FPC + FPC)
                if i % 2 == 0:                     # channel-block shuffle (cross-channel rupture)
                    base[sl] = Xn[rng.integers(n), sl]
                else:                              # per-channel beta injection (single-channel extreme)
                    for j in range(ch * FPC, ch * FPC + FPC):
                        inject(base, j, H, rng)
            out[r] = base
    else:                                          # tabular / univariate: feature level
        for r in range(ns):
            base = Xn[rng.integers(n)].copy(); k = max(1, int(frac * d)); cols = rng.choice(d, k, replace=False)
            for i, j in enumerate(cols):
                if i % 2 == 0:
                    base[j] = Xn[rng.integers(n), j]
                else:
                    inject(base, j, H, rng)
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


M = pd.read_csv(os.path.join(S, "HADB_MANIFEST.csv")); inc = M[M.include].copy(); inc["tbf"] = inc.gt_best.map(fam)
samp = pd.concat([g.sort_values("spread_ap_norm").iloc[np.linspace(0, len(g) - 1, min(12, len(g))).astype(int)] for _, g in inc.groupby("corpus")])
res = {"mixed": [], "em": []}
bymod = {}
for _, r in samp.iterrows():
    corpus = r.corpus
    try:
        if corpus in ("oddbench", "ovrbench", "adbench_dami"): Xtr, Xval, nch = tab_tv(corpus, r.dataset); pool = TAB_POOL; mod = "tabular"
        elif corpus == "tsbad_u": Xtr, Xval, nch = uni_tv(r.dataset, "tsbad/TSB-AD-U.zip"); pool = TS_POOL; mod = "uni-TS"
        elif corpus == "ucr": Xtr, Xval, nch = uni_tv(r.dataset, "ucr/UCR_TimeSeriesAnomalyDatasets2021.zip"); pool = TS_POOL; mod = "uni-TS"
        elif corpus == "tsbad_m": Xtr, Xval, nch = mts_tv(r.dataset); pool = TS_POOL; mod = "MTS"
        else: continue
    except Exception: continue
    if len(Xtr) < 40 or len(Xval) < 30: continue
    try: ap, emv = true_apnorm(CSVMAP[corpus], r.dataset)
    except Exception: continue
    common0 = [v for v, _ in pool if v in ap.index and v in emv.index and emv[v] == emv[v]]
    if len(common0) < 6: continue
    av = pd.Series({v: ap[v] for v in common0}); best = av.max(); ev = pd.Series({v: emv[v] for v in common0})
    sa = sauc(Xtr, Xval, gen_mixed(Xval, 150, nch=nch), pool); common = [v for v in sa if v in av.index]
    if len(common) < 6: continue
    sv = pd.Series({v: sa[v] for v in common}); sr = best - av[sv.idxmax()]; er = best - av[ev.idxmax()]
    res["mixed"].append(sr); res["em"].append(er)
    bymod.setdefault(mod, []).append((sr, er, r.tbf))
print(f"=== MIXED (shuffle+extreme, channel-aware for MTS) vs EM, all modalities ({len(res['em'])} datasets) ===")
s, e = np.array(res["mixed"]), np.array(res["em"])
print(f"  overall regret:  mixed {s.mean():.4f}   EM {e.mean():.4f}   (p={wilcoxon(s,e).pvalue:.3f}, mixed wins {int((s<e).sum())}/{len(s)})")
print("  by modality:")
for mod, v in bymod.items():
    a = np.array([(x[0], x[1]) for x in v]); print(f"    {mod:8s} n={len(a):2d}  mixed {a[:,0].mean():.3f}  EM {a[:,1].mean():.3f}  wins {int((a[:,0]<a[:,1]).sum())}/{len(a)}")
print("  by family:")
allv = [x for v in bymod.values() for x in v]
for f in ["local", "global", "other"]:
    a = np.array([(x[0], x[1]) for x in allv if x[2] == f])
    if len(a): print(f"    {f:7s} n={len(a):2d}  mixed {a[:,0].mean():.3f}  EM {a[:,1].mean():.3f}  wins {int((a[:,0]<a[:,1]).sum())}/{len(a)}")
# does MTS DRIVE the significance? compute p with and without MTS
print("\n  significance decomposition:")
non_mts = np.array([(x[0], x[1]) for mod, v in bymod.items() if mod != "MTS" for x in v])
mts_only = np.array([(x[0], x[1]) for x in bymod.get("MTS", [])])
if len(non_mts) > 5:
    p = wilcoxon(non_mts[:, 0], non_mts[:, 1]).pvalue
    print(f"    WITHOUT MTS ({len(non_mts)}): mixed {non_mts[:,0].mean():.4f} EM {non_mts[:,1].mean():.4f}  p={p:.4f}")
if len(mts_only) > 5:
    p = wilcoxon(mts_only[:, 0], mts_only[:, 1]).pvalue
    print(f"    MTS only    ({len(mts_only)}): mixed {mts_only[:,0].mean():.4f} EM {mts_only[:,1].mean():.4f}  p={p:.4f}")
print(f"    WITH MTS    ({len(s)}): p={wilcoxon(s,e).pvalue:.4f}")
