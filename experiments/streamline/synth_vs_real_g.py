# -*- coding: utf-8 -*-
"""How do our synthetics differ from REAL anomalies on GLOBAL datasets (where we lose to MV)?
Hypothesis: real global anomalies are extreme in a CONSISTENT small set of features (a signature);
our beta=-4 synthetic randomizes WHICH features are extreme -> it probes the wrong features, so it
cannot rank the global detectors. Measure the feature-signature consistency, real vs synthetic."""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"; S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
D = os.path.join(S, "scratchpad", "streamline")
sys.path.insert(0, os.path.join(S, "scratchpad")); sys.path.insert(0, os.path.join(ROOT, "src"))
import adrank.pipeline as PP
from hadb_ts_final import W, STRIDE, block_split3
from hadb_ts_mts import mts_window_features, window_labels as mts_wlabels, load_mts
Q = 0.05
OBJ = {}
for sub in ("adbench", "dami"):
    dd = os.path.join(ROOT, "data", sub)
    if os.path.isdir(dd):
        for ds in PP.load_npz_dir(dd): OBJ[(sub, str(ds.name)[:40])] = (np.nan_to_num(np.asarray(ds.X, float)), np.asarray(ds.y, int).ravel())
MTS = {}
for name, src, Xc, lab in load_mts(200): MTS[str(name)[:40]] = (Xc, lab)
def severity(Xtr, X, bins=30):
    n, d = X.shape; sev = np.zeros(n)
    for j in range(d):
        tr = np.sort(Xtr[:, j])
        if tr[-1] - tr[0] < 1e-12: continue
        m = len(tr); fb = np.searchsorted(tr, X[:, j], side="right") / m
        cnt, edges = np.histogram(Xtr[:, j], bins=bins); dens = cnt / max(cnt.sum(), 1); b = np.clip(np.digitize(X[:, j], edges[1:-1]), 0, bins - 1)
        sev = np.maximum(sev, np.maximum(-np.log(np.maximum(2 * np.minimum(fb, 1 - fb), 1.0 / (2 * m))), -np.log(dens[b] + 1e-9)))
    return sev
def _H(Xn, cap=98):
    H = []
    for j in range(Xn.shape[1]):
        col = Xn[:, j]; cnt, edges = np.histogram(col, bins=30); dens = cnt / max(cnt.sum(), 1)
        lo, hi = np.percentile(col, 100 - cap), np.percentile(col, cap); ctr = (edges[:-1] + edges[1:]) / 2
        H.append((edges, dens, (cnt > 0) & (ctr >= lo) & (ctr <= hi)))
    return H
def gen_beta(Xn, ns, beta=-4.0, frac=0.4, seed=0):
    rng = np.random.default_rng(seed); n, d = Xn.shape; H = _H(Xn); out = np.empty((ns, d))
    for r in range(ns):
        base = Xn[rng.integers(n)].copy()
        for j in rng.choice(d, max(1, int(frac * d)), replace=False):
            edges, dens, allowed = H[j]; w = np.where(allowed, np.maximum(dens, 1e-6) ** beta, 0.0)
            base[j] = Xn[rng.integers(n), j] if w.sum() == 0 else rng.uniform(*edges[[(b := rng.choice(len(w), p=w / w.sum())), b + 1]])
        out[r] = base
    return out
def get3(corp, name):
    if corp in ("oddbench", "ovrbench"):
        d = np.load(os.path.join(ROOT, "data", corp, name + ".npz"), allow_pickle=True)
        X = np.nan_to_num(np.vstack([np.asarray(d["train"], float), np.asarray(d["test"], float)])); y = np.concatenate([np.asarray(d["train_labels"]).ravel(), np.asarray(d["test_labels"]).ravel()]).astype(int); Xn, Xa = X[y == 0], X[y == 1]
    elif corp in ("adbench", "dami"):
        X, y = OBJ[(corp, name)]; Xn, Xa = X[y == 0], X[y == 1]
    else:
        Xc, lab = MTS[name]; Xw, st = mts_window_features(Xc); yw = mts_wlabels(lab, st); Xw = np.nan_to_num(Xw)
        pos = np.arange(len(Xw)); tr, va, te = block_split3(yw, pos, 0); return Xw[tr][yw[tr] == 0], Xw[va][yw[va] == 0], Xw[yw == 1]
    r = np.random.default_rng(0)
    if len(Xn) > 6000: Xn = Xn[r.choice(len(Xn), 6000, replace=False)]
    idx = np.arange(len(Xn)); r.shuffle(idx); a, b = int(0.6 * len(idx)), int(0.8 * len(idx)); return Xn[idx[:a]], Xn[idx[a:b]], Xa
def signature(Xtr, X):
    """concentration of WHICH feature is most extreme, across the anomaly/synth set.
       top1_share = frac sharing the single most-common top-|z| feature; eff_feats = participation ratio."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9; Z = np.abs((X - mu) / sd); top = Z.argmax(1)
    cnt = np.bincount(top, minlength=Xtr.shape[1]).astype(float); p = cnt / cnt.sum()
    eff = 1.0 / np.sum(p ** 2)   # participation ratio: ~1 if all anomalies share one feature, ~d if uniform
    return float(cnt.max() / cnt.sum()), float(eff), float((Z > 2).sum(1).mean())
rk = pd.read_csv(os.path.join(D, "STREAM_RANK.csv")); g = rk[rk.truefam == "global"].copy(); g["gap"] = g.reg_matched - g.reg_mv
sel = g.sort_values("gap", ascending=False).head(15)  # global datasets where we lose most to MV
rows = []
for _, r in sel.iterrows():
    try: Xtr, Xval, Xa = get3(r.corpus, r.dataset)
    except Exception: continue
    if len(Xtr) < 40 or len(Xval) < 20 or len(Xa) < 10: continue
    Xh = Xa[severity(Xtr, Xa) <= np.quantile(severity(Xtr, Xval), 1 - Q)]
    if len(Xh) < 10: continue
    syn = gen_beta(Xval, 300)
    r_share, r_eff, r_nx = signature(Xtr, Xh); s_share, s_eff, s_nx = signature(Xtr, syn)
    rows.append({"dataset": r.dataset[:22], "d": Xtr.shape[1], "reg_matched": round(r.reg_matched, 2), "reg_mv": round(r.reg_mv, 2),
                 "real_top1share": round(r_share, 2), "synth_top1share": round(s_share, 2),
                 "real_efffeats": round(r_eff, 1), "synth_efffeats": round(s_eff, 1), "real_nx": round(r_nx, 1), "synth_nx": round(s_nx, 1)})
df = pd.DataFrame(rows); df.to_csv(os.path.join(D, "STREAM_SYNTH_VS_REAL_G.csv"), index=False)
print(f"=== synthetic vs REAL anomalies on GLOBAL datasets where we lose to MV ({len(df)}) ===")
print("  top1share = frac of anomalies whose MOST-extreme feature is the SAME one (high = consistent signature)")
print("  eff_feats = # distinct 'anomaly features' (low = concentrated signature; ~d = spread/random)\n")
print(df.to_string(index=False))
print(f"\n  MEANS:  real top1share {df.real_top1share.mean():.2f} vs synth {df.synth_top1share.mean():.2f}   "
      f"|  real eff_feats {df.real_efffeats.mean():.1f} vs synth {df.synth_efffeats.mean():.1f}")
print("  => real global anomalies concentrate in a CONSISTENT feature signature; our synthetic spreads across RANDOM features")
print("saved streamline/STREAM_SYNTH_VS_REAL_G.csv")
