# -*- coding: utf-8 -*-
"""Extend the synthetic to a GLOBAL-anomaly type: consistent-single-feature ELEVATED synthetics (each
batch elevated in ONE feature; detector criterion = mean separation over feature-batches, rewarding
LOW-DILUTION detectors). Does it rank the GLOBAL detectors better than our current random beta=-4?
Metric: within-global Spearman(criterion, true ap_norm) on global datasets."""
import os, sys, io, contextlib, warnings
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"; S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
D = os.path.join(S, "scratchpad", "streamline")
sys.path.insert(0, os.path.join(S, "scratchpad")); sys.path.insert(0, os.path.join(ROOT, "src"))
import adrank.pipeline as PP
from dev_common import TAB_POOL
from hadb_ts_final import W, STRIDE, block_split3
from hadb_ts_mts import mts_window_features, window_labels as mts_wlabels, load_mts
GLOBAL = ("HBOS", "COPOD", "ECOD", "PCA"); fam = lambda v: "global" if str(v).startswith(GLOBAL) else "other"
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
def harden(Xtr, Xhold, Xa):
    tho = np.quantile(severity(Xtr, Xhold), 1 - Q); so_a = severity(Xtr, Xa)
    try:
        sc = StandardScaler().fit(Xtr); pca = PCA(n_components=0.95, whiten=True, random_state=0).fit(sc.transform(Xtr))
        Ptr, Ph, Pa = pca.transform(sc.transform(Xtr)), pca.transform(sc.transform(Xhold)), pca.transform(sc.transform(Xa))
        thp = np.quantile(severity(Ptr, Ph), 1 - Q); sp_a = severity(Ptr, Pa)
    except Exception:
        sp_a = np.zeros(len(Xa)); thp = np.inf
    return Xa[~((so_a > tho) | (sp_a > thp))]
def gen_rand_ext(Xn, ns, beta=-4.0, frac=0.4, seed=0):  # current probe (random features)
    rng = np.random.default_rng(seed); n, d = Xn.shape; out = np.empty((ns, d)); H = []
    for j in range(d):
        col = Xn[:, j]; cnt, edges = np.histogram(col, bins=30); dens = cnt / max(cnt.sum(), 1)
        lo, hi = np.percentile(col, 2), np.percentile(col, 98); ctr = (edges[:-1] + edges[1:]) / 2
        allowed = (cnt > 0) & (ctr >= lo) & (ctr <= hi); w = np.where(allowed, np.maximum(dens, 1e-6) ** beta, 0.0)
        H.append((edges, w / w.sum() if w.sum() > 0 else None))
    for r in range(ns):
        base = Xn[rng.integers(n)].copy()
        for j in rng.choice(d, max(1, int(frac * d)), replace=False):
            edges, w = H[j]; base[j] = Xn[rng.integers(n), j] if w is None else rng.uniform(*edges[[(b := rng.choice(len(w), p=w)), b + 1]])
        out[r] = base
    return out
def gen_single_feat(Xn, ns, j, seed=0, plo=85, phi=99):  # NEW: consistent single-feature elevated
    rng = np.random.default_rng(seed); n = len(Xn); out = Xn[rng.integers(n, size=ns)].copy()
    lo, hi = np.percentile(Xn[:, j], plo), np.percentile(Xn[:, j], phi)
    side = rng.random(ns) < 0.5; loL, hiL = np.percentile(Xn[:, j], 100 - phi), np.percentile(Xn[:, j], 100 - plo)
    out[:, j] = np.where(side, rng.uniform(lo, hi, ns), rng.uniform(loL, hiL, ns)); return out
def get4(corp, name):
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
    bb = y.mean(); return (average_precision_score(y, s) - bb) / (1 - bb + 1e-12)
rk = pd.read_csv(os.path.join(D, "STREAM_RANK.csv")); g = rk[rk.truefam == "global"]
sel = g.iloc[np.linspace(0, len(g) - 1, min(35, len(g))).astype(int)]; rows = []
for i, (_, r) in enumerate(sel.iterrows()):
    try: Xtr, Xval, Xtn, Xa = get4(r.corpus, r.dataset)
    except Exception: continue
    if len(Xtr) < 40 or len(Xval) < 30 or len(Xtn) < 10 or len(Xa) < 10: continue
    Xh = harden(Xtr, np.vstack([Xval, Xtn]), Xa)
    if len(Xh) < 15: continue
    Xte = np.vstack([Xtn, Xh]); yte = np.r_[np.zeros(len(Xtn)), np.ones(len(Xh))]; base = yte.mean(); d = Xtr.shape[1]
    randext = gen_rand_ext(Xval, 200); feats = np.random.default_rng(0).choice(d, min(d, 15), replace=False)
    sfbatches = [gen_single_feat(Xval, 60, int(j)) for j in feats]
    ap, cur, newp = {}, {}, {}
    yv = np.r_[np.zeros(len(Xval)), np.ones(len(randext))]
    for vn, ct in TAB_POOL:
        try:
            m = ct()
            with contextlib.redirect_stdout(io.StringIO()): m.fit(Xtr)
            sv = np.asarray(m.decision_function(Xval), float); st = np.asarray(m.decision_function(Xte), float)
            if not (np.all(np.isfinite(st)) and np.nanstd(st) > 1e-12 and np.nanstd(sv) > 1e-12): continue
            ap[vn] = apn(yte, st); cur[vn] = roc_auc_score(yv, np.r_[sv, m.decision_function(randext)])
            aucs = []
            for sb in sfbatches:
                ss = np.asarray(m.decision_function(sb), float)
                if np.nanstd(np.r_[sv, ss]) > 1e-12: aucs.append(roc_auc_score(np.r_[np.zeros(len(sv)), np.ones(len(ss))], np.r_[sv, ss]))
            newp[vn] = np.mean(aucs) if aucs else 0.5
        except Exception: pass
    gc = [v for v in ap if fam(v) == "global" and v in cur and v in newp]
    if len(gc) < 4: continue
    av = pd.Series({v: ap[v] for v in gc})
    def rho(cr):
        cc = [cr[v] for v in gc]; return spearmanr(cc, av[gc]).statistic if np.std(cc) > 0 else np.nan
    rows.append({"dataset": r.dataset[:22], "rho_current_rand": rho(cur), "rho_new_singlefeat": rho(newp),
                 "reg_current": av.max() - av[max(gc, key=lambda v: cur[v])], "reg_new": av.max() - av[max(gc, key=lambda v: newp[v])]})
    if i % 10 == 0: print(f"  ..{i}/{len(sel)}", flush=True)
df = pd.DataFrame(rows); df.to_csv(os.path.join(D, "STREAM_EXTEND_G.csv"), index=False)
print(f"\n=== consistent-single-feature probe vs current random beta=-4, WITHIN-GLOBAL ranking ({len(df)} global datasets) ===")
print(f"  within-global Spearman(criterion, true ap_norm):  current random {df.rho_current_rand.mean():+.3f}   NEW single-feature {df.rho_new_singlefeat.mean():+.3f}")
print(f"  within-global SELECTION regret:                   current {df.reg_current.mean():.3f}   NEW {df.reg_new.mean():.3f}")
print(f"  new probe ranks global detectors better on {int((df.rho_new_singlefeat>df.rho_current_rand).sum())}/{len(df)} datasets")
print("saved streamline/STREAM_EXTEND_G.csv")
