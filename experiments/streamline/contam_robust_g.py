# -*- coding: utf-8 -*-
"""CONTAMINATION-ROBUSTNESS probe (unsupervised-AD lens): inject synthetic anomalies INTO the training
set, fit each detector on the contaminated train, and measure whether it STILL separates fresh
anomalies (robust) vs absorbs them (fragile). Does robustness rank the GLOBAL detectors better than
clean-separation? Metric: within-global Spearman(criterion, true ap_norm)."""
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
Q = 0.05; CONTAM = 0.05  # inject anomalies at 5% of train
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
def gen_beta(Xn, ns, beta=-4.0, frac=0.4, seed=0):
    rng = np.random.default_rng(seed); n, d = Xn.shape; H = _H(Xn); out = np.empty((ns, d))
    for r in range(ns):
        base = Xn[rng.integers(n)].copy()
        for j in rng.choice(d, max(1, int(frac * d)), replace=False):
            edges, dens, allowed = H[j]; w = np.where(allowed, np.maximum(dens, 1e-6) ** beta, 0.0)
            base[j] = Xn[rng.integers(n), j] if w.sum() == 0 else rng.uniform(*edges[[(b := rng.choice(len(w), p=w / w.sum())), b + 1]])
        out[r] = base
    return out
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
    if len(Xtr) < 60 or len(Xval) < 30 or len(Xtn) < 10 or len(Xa) < 10: continue
    Xh = harden(Xtr, np.vstack([Xval, Xtn]), Xa)
    if len(Xh) < 15: continue
    Xte = np.vstack([Xtn, Xh]); yte = np.r_[np.zeros(len(Xtn)), np.ones(len(Xh))]
    ninj = max(5, int(CONTAM * len(Xtr))); syn_inj = gen_beta(Xval, ninj, seed=1); syn_probe = gen_beta(Xval, 200, seed=2)
    Xtr_c = np.vstack([Xtr, syn_inj]); yv = np.r_[np.zeros(len(Xval)), np.ones(len(syn_probe))]
    ap, clean, contam = {}, {}, {}
    for vn, ct in TAB_POOL:
        try:
            m = ct()
            with contextlib.redirect_stdout(io.StringIO()): m.fit(Xtr)
            st = np.asarray(m.decision_function(Xte), float); sv = np.asarray(m.decision_function(Xval), float); sp = np.asarray(m.decision_function(syn_probe), float)
            if not (np.all(np.isfinite(st)) and np.nanstd(st) > 1e-12 and np.nanstd(np.r_[sv, sp]) > 1e-12): continue
            ap[vn] = apn(yte, st); clean[vn] = roc_auc_score(yv, np.r_[sv, sp])
            mc = ct()
            with contextlib.redirect_stdout(io.StringIO()): mc.fit(Xtr_c)
            svc = np.asarray(mc.decision_function(Xval), float); spc = np.asarray(mc.decision_function(syn_probe), float)
            contam[vn] = roc_auc_score(yv, np.r_[svc, spc]) if np.nanstd(np.r_[svc, spc]) > 1e-12 else 0.5
        except Exception: pass
    gc = [v for v in ap if fam(v) == "global" and v in clean and v in contam]
    if len(gc) < 4: continue
    av = pd.Series({v: ap[v] for v in gc})
    def rho(cr):
        cc = [cr[v] for v in gc]; return spearmanr(cc, av[gc]).statistic if np.std(cc) > 0 else np.nan
    rows.append({"dataset": r.dataset[:22], "rho_clean": rho(clean), "rho_contam": rho(contam),
                 "reg_clean": av.max() - av[max(gc, key=lambda v: clean[v])], "reg_contam": av.max() - av[max(gc, key=lambda v: contam[v])]})
    if i % 10 == 0: print(f"  ..{i}/{len(sel)}", flush=True)
df = pd.DataFrame(rows); df.to_csv(os.path.join(D, "STREAM_CONTAM_G.csv"), index=False)
print(f"\n=== contamination-robustness vs clean-separation, WITHIN-GLOBAL ranking ({len(df)} global datasets) ===")
print(f"  within-global Spearman(criterion, true ap_norm):  clean-separation {df.rho_clean.mean():+.3f}   contamination-robust {df.rho_contam.mean():+.3f}")
print(f"  within-global SELECTION regret:                    clean {df.reg_clean.mean():.3f}   contam {df.reg_contam.mean():.3f}")
print(f"  contamination-robustness ranks global better on {int((df.rho_contam>df.rho_clean).sum())}/{len(df)} datasets")
print("saved streamline/STREAM_CONTAM_G.csv")
