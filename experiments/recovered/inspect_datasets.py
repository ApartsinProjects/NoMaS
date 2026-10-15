# -*- coding: utf-8 -*-
"""Deep per-dataset EDA: a1/a4 detector rankings vs true ap_norm + real-anomaly geometry."""
import os, sys, io, contextlib, warnings
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"; S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
sys.path.insert(0, os.path.join(S, "scratchpad"))
from dev_common import TAB_POOL, true_apnorm
LOCAL = ("LOF", "KNN", "CBLOF"); GLOBAL = ("HBOS", "COPOD", "ECOD", "PCA")
fam = lambda v: "L" if str(v).startswith(LOCAL) else ("G" if str(v).startswith(GLOBAL) else "o")
CSVMAP = {"oddbench": "hadb_oddbench.csv", "ovrbench": "hadb_ovrbench.csv"}
TARGETS = [("oddbench", "UserEventAnomalies"), ("ovrbench", "UrlAnalysis"), ("ovrbench", "PatientSurvival1"),
           ("oddbench", "DiamondClarity"), ("oddbench", "RacingMotion")]


def tab_full(corpus, name):
    d = np.load(os.path.join(ROOT, "data", corpus, name + ".npz"), allow_pickle=True)
    X = np.vstack([np.asarray(d["train"], float), np.asarray(d["test"], float)]); y = np.concatenate([np.asarray(d["train_labels"]).ravel(), np.asarray(d["test_labels"]).ravel()]).astype(int); X = np.nan_to_num(X)
    ni = np.where(y == 0)[0]; ai = np.where(y == 1)[0]
    if len(ni) > 5000:
        r = np.random.RandomState(0); ni = r.choice(ni, 5000, replace=False)
    g = np.random.default_rng(0); idx = np.arange(len(ni)); g.shuffle(idx); ni = ni[idx]
    Xtr = X[ni[:int(0.6 * len(ni))]]; Xval = X[ni[int(0.6 * len(ni)):int(0.8 * len(ni))]]; Xan = X[ai]
    return Xtr, Xval, Xan


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
            base[j] = Xn[rng.integers(n), j] if w.sum() == 0 else rng.uniform(*edges[[(b := rng.choice(len(w), p=w / w.sum())), b + 1]])
        out[r] = base
    return out


def sauc(Xtr, Xval, syn):
    Xe = np.vstack([Xval, syn]); ye = np.r_[np.zeros(len(Xval)), np.ones(len(syn))]; o = {}
    for vn, ct in TAB_POOL:
        try:
            m = ct()
            with contextlib.redirect_stdout(io.StringIO()): m.fit(Xtr)
            s = np.asarray(m.decision_function(Xe), float)
            if np.all(np.isfinite(s)) and np.nanstd(s) > 1e-12: o[vn] = float(roc_auc_score(ye, s))
        except Exception: pass
    return o


for corpus, name in TARGETS:
    try:
        Xtr, Xval, Xan = tab_full(corpus, name)
        ap, emv = true_apnorm(CSVMAP[corpus], name)
    except Exception as e:
        print(f"[skip {name}: {e}]"); continue
    a1 = sauc(Xtr, Xval, gen_beta(Xval, 200, 1.0)); a4 = sauc(Xtr, Xval, gen_beta(Xval, 200, -4.0))
    common = [v for v, _ in TAB_POOL if v in a1 and v in a4 and v in ap.index and ap[v] == ap[v] and v in emv.index and emv[v] == emv[v]]
    if len(common) < 6: print(f"[skip {name}: few common]"); continue
    av = pd.Series({v: ap[v] for v in common}); best_v = av.idxmax(); best = av.max()
    p1 = max(common, key=lambda v: a1[v]); pe = max(common, key=lambda v: a4[v]); pem = max(common, key=lambda v: emv[v])
    local_ev = max(a1.values()) - max(a4.values())
    # anomaly geometry vs train-normal reference
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9; Z = np.abs((Xan - mu) / sd); maxz = Z.max(1)
    n_ext_feat = (Z > 3).sum(1)  # how many features are marginally extreme per anomaly
    print(f"\n================  {corpus}/{name}   d={Xtr.shape[1]}  n_anom={len(Xan)} ================")
    print(f"  local_ev={local_ev:+.3f}   true-best={best_v}({fam(best_v)}) ap={best:.3f}")
    print(f"  GEOMETRY  median max|z|={np.median(maxz):.2f}  frac marginal(max|z|>3)={np.mean(maxz>3):.2f}  "
          f"frac multivar-only(max|z|<2.5)={np.mean(maxz<2.5):.2f}  mean #extreme-feats={n_ext_feat.mean():.1f}")
    print(f"  PICKS  beta1>{p1}({fam(p1)}) ap={av[p1]:.3f} reg={best-av[p1]:.3f} | betaext]{pe}({fam(pe)}) ap={av[pe]:.3f} reg={best-av[pe]:.3f} | EM:{pem}({fam(pem)}) ap={av[pem]:.3f} reg={best-av[pem]:.3f}")
    tbl = pd.DataFrame({"det": common, "fam": [fam(v) for v in common], "a1": [a1[v] for v in common],
                        "a4": [a4[v] for v in common], "ap_norm": [av[v] for v in common], "em": [emv[v] for v in common]})
    tbl = tbl.sort_values("a1", ascending=False).head(8)
    print("  top-8 by a1 (beta=+1 ranking):")
    for _, r in tbl.iterrows():
        mark = ("*" if r.det == best_v else " ") + (">" if r.det == p1 else " ") + ("E" if r.det == pem else " ")
        print(f"    {mark} {r.det[:16]:16s} {r.fam}  a1={r.a1:.3f}  a4={r.a4:.3f}  ap_norm={r.ap_norm:+.3f}  em={r.em:.3f}")
