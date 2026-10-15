# -*- coding: utf-8 -*-
"""Band-filtered generate-at-alpha selector. For each alpha, generate synthetics then KEEP only those in
the difficulty BAND: nearest-normal distance > normal-NN-scale (OOD floor) AND severity < hardening
threshold (not trivially extreme). This difficulty-matches synthetics to the real hard anomalies so alpha
controls only TYPE. Then selectors: max-disagreement, alpha-from-local_ev, best-alpha ORACLE, vs binary.
Incremental per-dataset CSV append + resume (per the observability rule)."""
import os, sys, io, contextlib, warnings
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
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
Q = 0.05; TAU = -0.14; ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]; OUT = os.path.join(D, "STREAM_GENALPHA_BAND.csv")
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
f = pd.read_csv(os.path.join(D, "STREAM_FINAL2_SET.csv"))
done = set(pd.read_csv(OUT).dataset) if os.path.exists(OUT) else set()   # RESUME
fo = open(OUT, "a");
if not done: fo.write("corpus,dataset,truefam,reg_binary,reg_maxdisagree,reg_alpha_localev,reg_bestalpha_oracle,mean_band_frac\n"); fo.flush()
for i, (_, r) in enumerate(f.iterrows()):
    if r.dataset in done: continue
    try: Xtr, Xval, Xtn, Xa = get3(r.corpus, r.dataset)
    except Exception: continue
    if len(Xtr) < 40 or len(Xtn) < 10 or len(Xa) < 5: continue
    Xh = harden(Xtr, np.vstack([Xval, Xtn]), Xa)
    if len(Xh) < 20: continue
    Xte = np.vstack([Xtn, Xh]); yte = np.r_[np.zeros(len(Xtn)), np.ones(len(Xh))]; base = yte.mean()
    ceiling = np.quantile(severity(Xtr, Xval), 1 - Q)                    # hardening threshold (upper band edge)
    sc = StandardScaler().fit(Xtr); Zv = sc.transform(Xval); scale = np.median(NearestNeighbors(n_neighbors=2).fit(Zv).kneighbors(Zv)[0][:, 1]) + 1e-9; nn = NearestNeighbors(n_neighbors=1).fit(Zv)
    syns, bandfracs = {}, []
    for al in ALPHAS:
        s = gen_alpha(Xval, 400, al); sev = severity(Xtr, s); dd = nn.kneighbors(sc.transform(s))[0][:, 0]
        band = (dd > scale) & (sev < ceiling)                           # BAND: OOD floor & below hardening ceiling
        bandfracs.append(float(band.mean()))
        syns[al] = s[band][:200] if band.sum() >= 30 else s[:200]        # fallback if band too small
    names, ap, sig = [], {}, {}
    for vn, ct in TAB_POOL:
        try:
            m = ct()
            with contextlib.redirect_stdout(io.StringIO()): m.fit(Xtr)
            st = np.asarray(m.decision_function(Xte), float); sv = np.asarray(m.decision_function(Xval), float)
            if not (np.all(np.isfinite(st)) and np.nanstd(st) > 1e-12): continue
            names.append(vn); ap[vn] = (average_precision_score(yte, st) - base) / (1 - base + 1e-12)
            row = []
            for al in ALPHAS:
                ss = np.asarray(m.decision_function(syns[al]), float); yv = np.r_[np.zeros(len(sv)), np.ones(len(ss))]
                row.append(roc_auc_score(yv, np.r_[sv, ss]) if np.nanstd(np.r_[sv, ss]) > 1e-12 else 0.5)
            sig[vn] = np.array(row)
        except Exception: pass
    if len(names) < 6: continue
    av = pd.Series(ap); best = av.max(); SIG = np.vstack([sig[v] for v in names])
    lev = SIG[:, -1].max() - SIG[:, 0].max()
    p_bin = names[SIG[:, -1].argmax()] if lev > TAU else names[SIG[:, 0].argmax()]
    p_dis = names[SIG[:, int(np.argmax(SIG.std(0)))].argmax()]
    alpha_e = 1.0 / (1.0 + np.exp(-(lev + 0.14) * 4.0)); ai = int(np.argmin([abs(a - alpha_e) for a in ALPHAS])); p_le = names[SIG[:, ai].argmax()]
    reg_bestalpha = min(best - av[names[SIG[:, j].argmax()]] for j in range(len(ALPHAS)))
    tf = "local" if any(fam(v) == "local" for v in names) and (max([av[v] for v in names if fam(v) == "local"], default=-9) >= max([av[v] for v in names if fam(v) == "global"], default=-9)) else "global"
    fo.write(f"{r.corpus},{r.dataset},{tf},{best-av[p_bin]:.4f},{best-av[p_dis]:.4f},{best-av[p_le]:.4f},{reg_bestalpha:.4f},{np.mean(bandfracs):.3f}\n"); fo.flush()
    print(f"  {i}/{len(f)} {r.dataset[:24]} bandfrac={np.mean(bandfracs):.2f}", flush=True)
fo.close()
df = pd.read_csv(OUT)
def macro(c): return np.mean([df[df.truefam == fa][c].mean() for fa in ["local", "global"] if (df.truefam == fa).any()])
print(f"\n=== BAND-filtered generate-at-alpha ({len(df)} datasets, mean band-frac {df.mean_band_frac.mean():.2f}) ===")
print(f"  {'selector':22s} {'micro':>6s} {'macro':>6s}")
for nm, c in [("binary matched", "reg_binary"), ("max-disagreement", "reg_maxdisagree"), ("alpha from local_ev", "reg_alpha_localev"), ("best-alpha ORACLE", "reg_bestalpha_oracle")]:
    print(f"  {nm:22s} {df[c].mean():6.3f} {macro(c):6.3f}")
print("saved streamline/STREAM_GENALPHA_BAND.csv")
