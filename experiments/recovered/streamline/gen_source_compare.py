# -*- coding: utf-8 -*-
"""Does the synthetic SOURCE matter (train has 3x more normals than val)? Compare matched-selector regret
with synthetics from: V=val/ref-val (current), T=train/ref-train (more coverage but detector fit on train
-> possible memorizer leak), TV=train-synth/ref-val (coverage + clean reference). Focus on low-n_val
datasets. Incremental per-dataset CSV append + resume (observability rule)."""
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
Q = 0.05; TAU = -0.14; OUT = os.path.join(D, "STREAM_SRC_COMPARE.csv")
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
def gen_beta(Xn, ns, beta, frac=0.4, seed=0):
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
        pos = np.arange(len(Xw)); tr, va, te = block_split3(yw, pos, 0); return Xw[tr][yw[tr] == 0], Xw[va][yw[va] == 0], Xw[te][yw[te] == 0], Xw[yw == 1]
    r = np.random.default_rng(0)
    if len(Xn) > 6000: Xn = Xn[r.choice(len(Xn), 6000, replace=False)]
    idx = np.arange(len(Xn)); r.shuffle(idx); a, b = int(0.6 * len(idx)), int(0.8 * len(idx)); return Xn[idx[:a]], Xn[idx[a:b]], Xn[idx[b:]], Xa
f = pd.read_csv(os.path.join(D, "STREAM_FINAL2_SET.csv"))
done = set(pd.read_csv(OUT).dataset) if os.path.exists(OUT) else set()
fo = open(OUT, "a")
if not done: fo.write("corpus,dataset,truefam,n_val,n_tr,reg_V,reg_T,reg_TV\n"); fo.flush()
for i, (_, r) in enumerate(f.iterrows()):
    if r.dataset in done: continue
    try: Xtr, Xval, Xtn, Xa = get3(r.corpus, r.dataset)
    except Exception: continue
    if len(Xtr) < 40 or len(Xtn) < 10 or len(Xval) < 20 or len(Xa) < 5: continue
    Xh = harden(Xtr, np.vstack([Xval, Xtn]), Xa)
    if len(Xh) < 20: continue
    Xte = np.vstack([Xtn, Xh]); yte = np.r_[np.zeros(len(Xtn)), np.ones(len(Xh))]; base = yte.mean()
    s1v, s4v = gen_beta(Xval, 200, 1.0, seed=1), gen_beta(Xval, 200, -4.0, seed=2)
    s1t, s4t = gen_beta(Xtr, 200, 1.0, seed=1), gen_beta(Xtr, 200, -4.0, seed=2)
    names, ap = [], {}
    a1V, a4V, a1T, a4T, a1TV, a4TV = {}, {}, {}, {}, {}, {}
    yV = np.r_[np.zeros(len(Xval)), np.ones(200)]; yT = np.r_[np.zeros(len(Xtr)), np.ones(200)]
    for vn, ct in TAB_POOL:
        try:
            m = ct()
            with contextlib.redirect_stdout(io.StringIO()): m.fit(Xtr)
            st = np.asarray(m.decision_function(Xte), float)
            if not (np.all(np.isfinite(st)) and np.nanstd(st) > 1e-12): continue
            sv = np.asarray(m.decision_function(Xval), float); str_ = np.asarray(m.decision_function(Xtr), float)
            names.append(vn); ap[vn] = (average_precision_score(yte, st) - base) / (1 - base + 1e-12)
            a1V[vn] = roc_auc_score(yV, np.r_[sv, m.decision_function(s1v)]); a4V[vn] = roc_auc_score(yV, np.r_[sv, m.decision_function(s4v)])
            a1T[vn] = roc_auc_score(yT, np.r_[str_, m.decision_function(s1t)]); a4T[vn] = roc_auc_score(yT, np.r_[str_, m.decision_function(s4t)])
            a1TV[vn] = roc_auc_score(yV, np.r_[sv, m.decision_function(s1t)]); a4TV[vn] = roc_auc_score(yV, np.r_[sv, m.decision_function(s4t)])
        except Exception: pass
    if len(names) < 6: continue
    av = pd.Series(ap); best = av.max()
    def matched(a1, a4):
        lev = max(a1[v] for v in names) - max(a4[v] for v in names)
        return best - av[max(names, key=lambda v: a1[v]) if lev > TAU else max(names, key=lambda v: a4[v])]
    tf = "local" if any(fam(v) == "local" for v in names) and (max([av[v] for v in names if fam(v) == "local"], default=-9) >= max([av[v] for v in names if fam(v) == "global"], default=-9)) else "global"
    fo.write(f"{r.corpus},{r.dataset},{tf},{len(Xval)},{len(Xtr)},{matched(a1V,a4V):.4f},{matched(a1T,a4T):.4f},{matched(a1TV,a4TV):.4f}\n"); fo.flush()
    print(f"  {i}/{len(f)} {r.dataset[:22]} n_val={len(Xval)}", flush=True)
fo.close()
df = pd.read_csv(OUT)
def macro(c): return np.mean([df[df.truefam == fa][c].mean() for fa in ["local", "global"] if (df.truefam == fa).any()])
print(f"\n=== synthetic SOURCE comparison ({len(df)} datasets) ===")
print(f"  {'variant':28s} {'micro':>6s} {'macro':>6s}")
for nm, c in [("V  val-synth / val-ref (current)", "reg_V"), ("T  train-synth / train-ref", "reg_T"), ("TV train-synth / val-ref", "reg_TV")]:
    print(f"  {nm:28s} {df[c].mean():6.3f} {macro(c):6.3f}")
lo = df[df.n_val < df.n_val.median()]
print(f"\n  on LOW-n_val half ({len(lo)} datasets, n_val<{int(df.n_val.median())}):  V {lo.reg_V.mean():.3f}  T {lo.reg_T.mean():.3f}  TV {lo.reg_TV.mean():.3f}")
print("saved streamline/STREAM_SRC_COMPARE.csv")
