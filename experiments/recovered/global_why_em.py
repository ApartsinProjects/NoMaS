# -*- coding: utf-8 -*-
"""Why does EM beat the beta=-4 (extreme) probe on GLOBAL datasets?
Compare beta=-4 synthetic anomalies to real global anomalies, and ask whether beta=-4 separation
ranks the global detectors as well as EM does."""
import os, sys, io, contextlib, zipfile, re, warnings
import numpy as np, pandas as pd
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


def tab_load(corpus, name):
    d = np.load(os.path.join(ROOT, "data", corpus, name + ".npz"), allow_pickle=True)
    X = np.vstack([np.asarray(d["train"], float), np.asarray(d["test"], float)]); y = np.concatenate([np.asarray(d["train_labels"]).ravel(), np.asarray(d["test_labels"]).ravel()]).astype(int); X = np.nan_to_num(X)
    r = np.random.default_rng(0); ni = np.where(y == 0)[0]; ai = np.where(y == 1)[0]
    if len(ni) > 5000: ni = r.choice(ni, 5000, replace=False)
    if len(ai) > 400: ai = r.choice(ai, 400, replace=False)
    idx = np.arange(len(ni)); r.shuffle(idx); ni = ni[idx]
    return X[ni[:int(0.6 * len(ni))]], X[ni[int(0.6 * len(ni)):int(0.8 * len(ni))]], X[ai]


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
    return Xw[tr][yw[tr] == 0], Xw[va][yw[va] == 0], Xw[(yw == 1)]


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


mp = pd.read_csv(os.path.join(S, "MODALITY_PERF.csv"))
glob = mp[mp.best_global > mp.best_local]
rows = []
for _, r in glob.iterrows():
    try:
        if r.corpus in ("oddbench", "ovrbench"): Xtr, Xval, Xan = tab_load(r.corpus, r.dataset); pool = TAB_POOL
        elif r.corpus == "ucr": Xtr, Xval, Xan = uni_load(r.dataset, "ucr/UCR_TimeSeriesAnomalyDatasets2021.zip"); pool = TS_POOL
        else: Xtr, Xval, Xan = uni_load(r.dataset, "tsbad/TSB-AD-U.zip"); pool = TS_POOL
    except Exception: continue
    if len(Xtr) < 40 or len(Xval) < 20 or len(Xan) < 5: continue
    syn = gen_beta(Xval, 200, -4.0)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    real_mz = np.median(np.abs((Xan - mu) / sd).max(1)); syn_mz = np.median(np.abs((syn - mu) / sd).max(1))
    real_nx = np.mean((np.abs((Xan - mu) / sd) > 3).sum(1)); syn_nx = np.mean((np.abs((syn - mu) / sd) > 3).sum(1))
    a4 = sauc(Xtr, Xval, syn, pool)
    try: ap, emv = true_apnorm(CSVMAP[r.corpus], r.dataset)
    except Exception: continue
    common = [v for v, _ in pool if v in a4 and v in ap.index and ap[v] == ap[v] and v in emv.index and emv[v] == emv[v]]
    gcommon = [v for v in common if fam(v) == "global"]
    if len(common) < 6 or len(gcommon) < 3: continue
    av = pd.Series({v: ap[v] for v in common}); best = av.max()
    # within-dataset ranking quality of the beta=-4 probe vs EM (over ALL detectors and over GLOBAL only)
    rho_a4 = spearmanr([a4[v] for v in common], [ap[v] for v in common]).statistic
    rho_em = spearmanr([emv[v] for v in common], [ap[v] for v in common]).statistic
    rho_a4_g = spearmanr([a4[v] for v in gcommon], [ap[v] for v in gcommon]).statistic if len(gcommon) >= 4 else np.nan
    rho_em_g = spearmanr([emv[v] for v in gcommon], [ap[v] for v in gcommon]).statistic if len(gcommon) >= 4 else np.nan
    rows.append({"dataset": r.dataset, "real_mz": real_mz, "syn_mz": syn_mz, "real_nx": real_nx, "syn_nx": syn_nx,
                 "a4_spread": float(np.std([a4[v] for v in common])), "ap_spread": float(np.std([ap[v] for v in common])),
                 "rho_a4": rho_a4, "rho_em": rho_em, "rho_a4_g": rho_a4_g, "rho_em_g": rho_em_g,
                 "betaext_reg": best - av[max(common, key=lambda v: a4[v])], "em_reg": best - av[max(common, key=lambda v: emv[v])]})
df = pd.DataFrame(rows); df.to_csv(os.path.join(S, "GLOBAL_WHY_EM.csv"), index=False)
print(f"=== why EM beats the beta=-4 extreme probe on GLOBAL datasets (n={len(df)}) ===")
print(f"\n1) are beta=-4 synthetics as extreme as real global anomalies? (max|z| vs train)")
print(f"   real anomalies  median max|z| = {df.real_mz.mean():.2f}   mean #features>3sigma = {df.real_nx.mean():.2f}")
print(f"   beta=-4 synth   median max|z| = {df.syn_mz.mean():.2f}   mean #features>3sigma = {df.syn_nx.mean():.2f}")
print(f"   -> synth reaches {df.syn_mz.mean()/df.real_mz.mean()*100:.0f}% of real extremeness")
print(f"\n2) does beta=-4 separation RANK the global detectors as well as EM? (within-dataset Spearman vs true ap)")
print(f"   over ALL detectors:    rho(a4)={df.rho_a4.mean():+.3f}   rho(EM)={df.rho_em.mean():+.3f}")
print(f"   over GLOBAL detectors: rho(a4)={df.rho_a4_g.mean():+.3f}   rho(EM)={df.rho_em_g.mean():+.3f}")
print(f"   a4 spread across detectors = {df.a4_spread.mean():.3f}  (flat = non-discriminative)   ap spread = {df.ap_spread.mean():.3f}")
print(f"\n3) selection regret on global datasets: betaext {df.betaext_reg.mean():.3f}   EM {df.em_reg.mean():.3f}")
print("\nsaved GLOBAL_WHY_EM.csv")
