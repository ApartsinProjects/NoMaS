# -*- coding: utf-8 -*-
"""Few-shot alpha calibration: use 10% of the hardened anomalies to CALIBRATE, evaluate regret on the
other 90% (no leakage). Compare: binary matched (unsup), alpha-via-10% (10% picks which alpha's probe-
pick is best), direct-10% (best detector on the 10% anomalies, no synthetics), oracle (on 90%).
Incremental per-dataset CSV + resume."""
import os, sys, io, contextlib, warnings
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"; S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
D = os.path.join(S, "scratchpad", "streamline"); sys.path.insert(0, os.path.join(S, "scratchpad")); sys.path.insert(0, os.path.join(ROOT, "src"))
import adrank.pipeline as PP
from dev_common import TAB_POOL
from hadb_ts_final import W, STRIDE, block_split3
from hadb_ts_mts import mts_window_features, window_labels as mts_wlabels, load_mts
LOCAL = ("LOF", "KNN", "CBLOF"); GLOBAL = ("HBOS", "COPOD", "ECOD", "PCA")
fam = lambda v: "local" if str(v).startswith(LOCAL) else ("global" if str(v).startswith(GLOBAL) else "other")
Q = 0.05; TAU = -0.14; ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]; FRAC = 0.10; OUT = os.path.join(D, "STREAM_SEMISUP.csv")
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
def harden(Xtr, Xhold, Xa):
    tho = np.quantile(severity(Xtr, Xhold), 1 - Q); so_a = severity(Xtr, Xa)
    try:
        sc = StandardScaler().fit(Xtr); pca = PCA(n_components=0.95, whiten=True, random_state=0).fit(sc.transform(Xtr))
        Ptr, Ph, Pa = pca.transform(sc.transform(Xtr)), pca.transform(sc.transform(Xhold)), pca.transform(sc.transform(Xa))
        thp = np.quantile(severity(Ptr, Ph), 1 - Q); sp_a = severity(Ptr, Pa)
    except Exception:
        sp_a = np.zeros(len(Xa)); thp = np.inf
    return Xa[~((so_a > tho) | (sp_a > thp))]
def _H(Xn, cap=98):
    H = []
    for j in range(Xn.shape[1]):
        col = Xn[:, j]; cnt, edges = np.histogram(col, bins=30); dens = cnt / max(cnt.sum(), 1)
        lo, hi = np.percentile(col, 100 - cap), np.percentile(col, cap); ctr = (edges[:-1] + edges[1:]) / 2
        H.append((edges, dens, (cnt > 0) & (ctr >= lo) & (ctr <= hi)))
    return H
def gen_alpha(Xn, ns, alpha, seed=0):
    rng = np.random.default_rng(seed); n, d = Xn.shape; H = _H(Xn); out = np.empty((ns, d))
    k = max(1, int(round(1 + alpha * (d - 1)))); beta = -4.0 + 6.0 * alpha
    for r in range(ns):
        base = Xn[rng.integers(n)].copy()
        for j in rng.choice(d, k, replace=False):
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
        pos = np.arange(len(Xw)); tr, va, te = block_split3(yw, pos, 0); return Xw[tr][yw[tr] == 0], Xw[va][yw[va] == 0], Xw[te][yw[te] == 0], Xw[yw == 1]
    r = np.random.default_rng(0)
    if len(Xn) > 6000: Xn = Xn[r.choice(len(Xn), 6000, replace=False)]
    idx = np.arange(len(Xn)); r.shuffle(idx); a, b = int(0.6 * len(idx)), int(0.8 * len(idx)); return Xn[idx[:a]], Xn[idx[a:b]], Xn[idx[b:]], Xa
def apn(y, s):
    bb = y.mean(); return (average_precision_score(y, s) - bb) / (1 - bb + 1e-12) if 0 < bb < 1 else np.nan
f = pd.read_csv(os.path.join(D, "STREAM_FINAL2_SET.csv"))
done = set(pd.read_csv(OUT).dataset) if os.path.exists(OUT) else set()
fo = open(OUT, "a")
if not done: fo.write("corpus,dataset,truefam,n_calib,reg_binary,reg_alpha10,reg_direct10,reg_oracle\n"); fo.flush()
for i, (_, r) in enumerate(f.iterrows()):
    if r.dataset in done: continue
    try: Xtr, Xval, Xtn, Xa = get3(r.corpus, r.dataset)
    except Exception: continue
    if len(Xtr) < 40 or len(Xtn) < 10 or len(Xval) < 20 or len(Xa) < 5: continue
    Xh = harden(Xtr, np.vstack([Xval, Xtn]), Xa)
    if len(Xh) < 40: continue
    rng = np.random.default_rng(0); perm = rng.permutation(len(Xh)); ncal = max(5, int(FRAC * len(Xh)))
    Hc, He = Xh[perm[:ncal]], Xh[perm[ncal:]]                       # 10% calib, 90% eval (disjoint)
    yE = np.r_[np.zeros(len(Xtn)), np.ones(len(He))]; yC = np.r_[np.zeros(len(Xtn)), np.ones(len(Hc))]
    syns = {al: gen_alpha(Xval, 200, al) for al in ALPHAS}
    names, apE, apC, sig = [], {}, {}, {}
    for vn, ct in TAB_POOL:
        try:
            m = ct()
            with contextlib.redirect_stdout(io.StringIO()): m.fit(Xtr)
            sn = np.asarray(m.decision_function(Xtn), float); se = np.asarray(m.decision_function(He), float); scl = np.asarray(m.decision_function(Hc), float); sv = np.asarray(m.decision_function(Xval), float)
            if not (np.all(np.isfinite(np.r_[sn, se])) and np.nanstd(np.r_[sn, se]) > 1e-12): continue
            names.append(vn); apE[vn] = apn(yE, np.r_[sn, se]); apC[vn] = apn(yC, np.r_[sn, scl])
            row = []
            for al in ALPHAS:
                ss = np.asarray(m.decision_function(syns[al]), float); yv = np.r_[np.zeros(len(sv)), np.ones(len(ss))]
                row.append(roc_auc_score(yv, np.r_[sv, ss]) if np.nanstd(np.r_[sv, ss]) > 1e-12 else 0.5)
            sig[vn] = np.array(row)
        except Exception: pass
    names = [v for v in names if np.isfinite(apE.get(v, np.nan)) and np.isfinite(apC.get(v, np.nan))]
    if len(names) < 6: continue
    apE_ = {v: apE[v] for v in names}; best = max(apE_.values()); SIG = {v: sig[v] for v in names}
    def pick_at(ai): return max(names, key=lambda v: SIG[v][ai])
    lev = max(SIG[v][-1] for v in names) - max(SIG[v][0] for v in names)
    p_bin = pick_at(len(ALPHAS) - 1) if lev > TAU else pick_at(0)
    astar = int(np.argmax([apC[pick_at(ai)] for ai in range(len(ALPHAS))]))   # 10% picks the alpha
    p_a10 = pick_at(astar)
    p_dir = max(names, key=lambda v: apC[v])                                    # direct 10% detector pick
    tf = "local" if any(fam(v) == "local" for v in names) and (max([apE_[v] for v in names if fam(v) == "local"], default=-9) >= max([apE_[v] for v in names if fam(v) == "global"], default=-9)) else "global"
    fo.write(f"{r.corpus},{r.dataset},{tf},{ncal},{best-apE_[p_bin]:.4f},{best-apE_[p_a10]:.4f},{best-apE_[p_dir]:.4f},0.0\n"); fo.flush()
    print(f"  {i}/{len(f)} {r.dataset[:22]} ncal={ncal}", flush=True)
fo.close()
df = pd.read_csv(OUT)
def macro(c): return np.mean([df[df.truefam == fa][c].mean() for fa in ["local", "global"] if (df.truefam == fa).any()])
print(f"\n=== 10%-anomaly alpha calibration ({len(df)} datasets, median n_calib {int(df.n_calib.median())}) ===")
print(f"  {'method':26s} {'micro':>6s} {'macro':>6s}")
for nm, c in [("binary matched (unsup)", "reg_binary"), ("alpha-via-10% (semi)", "reg_alpha10"), ("direct-10% (few-shot)", "reg_direct10")]:
    print(f"  {nm:26s} {df[c].mean():6.3f} {macro(c):6.3f}")
print("saved streamline/STREAM_SEMISUP.csv")
