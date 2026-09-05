# -*- coding: utf-8 -*-
"""Does an AP-based synthetic probe beat the AUC probe? For each detector grade its synthetic separation
BOTH ways: AUC (current) and ap_norm (aligns the probe metric with the target). Compare the resulting
selectors (beta1, matched, corr_str, combined) under each probe. Classical pool, bundle splits
(construct-matched). Incremental per-dataset CSV + resume."""
import os, sys, io, contextlib, warnings
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")
S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
D = os.path.join(S, "scratchpad", "streamline"); sys.path.insert(0, os.path.join(S, "scratchpad")); sys.path.insert(0, r"E:\Projects\Submitted\ADRank\src")
from dev_common import TAB_POOL
LOCAL = ("LOF", "KNN", "CBLOF"); GLOBAL = ("HBOS", "COPOD", "ECOD", "PCA")
fam = lambda v: "local" if str(v).startswith(LOCAL) else ("global" if str(v).startswith(GLOBAL) else "other")
TAU = -0.14; OUT = os.path.join(D, "STREAM_APPROBE.csv"); BUN = os.path.join(D, "bundle")
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
def apn(y, s):
    bb = y.mean(); return (average_precision_score(y, s) - bb) / (1 - bb + 1e-12) if 0 < bb < 1 else np.nan
idx = pd.read_csv(os.path.join(BUN, "_index.csv")); mine = pd.read_csv(os.path.join(D, "STREAM_MINE.csv")); mine["key"] = mine.corpus + "__" + mine.dataset
corr = dict(zip(mine.key, mine.corr_str))
done = set(pd.read_csv(OUT).dataset) if os.path.exists(OUT) else set(); fo = open(OUT, "a")
if not done: fo.write("key,truefam,cs,reg_b1_auc,reg_b1_ap,reg_m_auc,reg_m_ap,reg_p1,reg_pe_auc,reg_pe_ap,lev_auc,lev_ap\n"); fo.flush()
for _, rr in idx.iterrows():
    key = rr.key
    if key in done: continue
    try: z = np.load(os.path.join(BUN, key + ".npz")); Xtr, Xval, Xtn, Xh = z["Xtr"], z["Xval"], z["Xtn"], z["Xh"]
    except Exception: continue
    Xte = np.vstack([Xtn, Xh]); yte = np.r_[np.zeros(len(Xtn)), np.ones(len(Xh))]; base = yte.mean()
    syn1, syn4 = gen_beta(Xval, 200, 1.0), gen_beta(Xval, 200, -4.0); ye1 = np.r_[np.zeros(len(Xval)), np.ones(200)]
    names, ap, a1u, a4u, a1p, a4p = [], {}, {}, {}, {}, {}
    for vn, ct in TAB_POOL:
        try:
            m = ct()
            with contextlib.redirect_stdout(io.StringIO()): m.fit(Xtr)
            st = np.asarray(m.decision_function(Xte), float); sv = np.asarray(m.decision_function(Xval), float)
            s1 = np.asarray(m.decision_function(syn1), float); s4 = np.asarray(m.decision_function(syn4), float)
            if not (np.all(np.isfinite(st)) and np.nanstd(st) > 1e-12): continue
            names.append(vn); ap[vn] = (average_precision_score(yte, st) - base) / (1 - base + 1e-12)
            a1u[vn] = roc_auc_score(ye1, np.r_[sv, s1]); a4u[vn] = roc_auc_score(ye1, np.r_[sv, s4])
            a1p[vn] = apn(ye1, np.r_[sv, s1]); a4p[vn] = apn(ye1, np.r_[sv, s4])
        except Exception: pass
    if len(names) < 6: continue
    av = pd.Series(ap); best = av.max()
    p1u = max(names, key=lambda v: a1u[v]); peu = max(names, key=lambda v: a4u[v]); levu = max(a1u.values()) - max(a4u.values())
    p1p = max(names, key=lambda v: a1p[v]); pep = max(names, key=lambda v: a4p[v]); levp = max(a1p.values()) - max(a4p.values())
    loc = [ap[v] for v in names if fam(v) == "local"]; glo = [ap[v] for v in names if fam(v) == "global"]
    tf = "local" if (loc and (not glo or max(loc) > max(glo))) else "global"
    fo.write(f"{key},{tf},{corr.get(key,np.nan)},{best-av[p1u]:.4f},{best-av[p1p]:.4f},{best-av[p1u if levu>TAU else peu]:.4f},{best-av[p1p if levp>TAU else pep]:.4f},{best-av[p1u]:.4f},{best-av[peu]:.4f},{best-av[pep]:.4f},{levu:.4f},{levp:.4f}\n"); fo.flush()
    print(f"  {rr.dataset[:22]} levAUC={levu:+.2f} levAP={levp:+.2f}", flush=True)
fo.close()
df = pd.read_csv(OUT); med = df.cs.median()
df["reg_comb_auc"] = np.where((df.cs > med) | (df.lev_auc > TAU), df.reg_p1, df.reg_pe_auc)
df["reg_comb_ap"] = np.where((df.cs > med) | (df.lev_ap > TAU), df.reg_b1_ap, df.reg_pe_ap)   # p1 under AP probe = reg_b1_ap
def macro(c): return np.mean([df[df.truefam == fa][c].mean() for fa in ["local", "global"] if (df.truefam == fa).any()])
print(f"\n=== AUC probe vs AP probe (classical pool, {len(df)} datasets) ===")
print(f"  {'selector':22s} {'AUC-probe':>10s} {'AP-probe':>10s}")
for nm, ca, cp in [("beta=1", "reg_b1_auc", "reg_b1_ap"), ("matched", "reg_m_auc", "reg_m_ap"), ("combined", "reg_comb_auc", "reg_comb_ap")]:
    print(f"  {nm:22s} {df[ca].mean():.3f}/{macro(ca):.3f}  {df[cp].mean():.3f}/{macro(cp):.3f}  (micro/macro)")
print("saved streamline/STREAM_APPROBE.csv")
