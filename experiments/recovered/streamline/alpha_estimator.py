# -*- coding: utf-8 -*-
"""Try to close the alpha-estimator gap. Pass 1 (incremental JSONL): per dataset save label-free features
(local_ev, silhouette, corr-void, PC1 dip, d, logn), the detector signature SIG[det x alpha], ap, and the
oracle alpha. Pass 2: compare estimators - binary, max-disagreement, MARGIN (top1-top2), CONSENSUS
(detector winning most alphas), STRUCTURAL (features->alpha heuristic), and LEARNED (leave-one-out
regressor features->oracle_alpha). Reports regret vs the best-alpha oracle. Incremental + resume."""
import os, sys, io, json, contextlib, warnings, glob
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, average_precision_score, silhouette_score
from scipy.stats import kurtosis
warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"; S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
D = os.path.join(S, "scratchpad", "streamline"); sys.path.insert(0, os.path.join(S, "scratchpad")); sys.path.insert(0, os.path.join(ROOT, "src"))
import adrank.pipeline as PP
from dev_common import TAB_POOL
from hadb_ts_final import W, STRIDE, block_split3
from hadb_ts_mts import mts_window_features, window_labels as mts_wlabels, load_mts
LOCAL = ("LOF", "KNN", "CBLOF"); GLOBAL = ("HBOS", "COPOD", "ECOD", "PCA")
fam = lambda v: "local" if str(v).startswith(LOCAL) else ("global" if str(v).startswith(GLOBAL) else "other")
Q = 0.05; TAU = -0.14; ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]; JL = os.path.join(D, "STREAM_ALPHAEST.jsonl")
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
def struct_features(Xtr):
    sc = StandardScaler().fit(Xtr); Z = sc.transform(Xtr)
    rng = np.random.default_rng(0); sub = Z[rng.choice(len(Z), min(len(Z), 1500), replace=False)]
    try:
        pca = PCA().fit(sub); ev = pca.explained_variance_ratio_; corr_void = float(ev[len(ev)//2:].sum())  # var in low PCs (correlation/voids)
    except Exception: corr_void = 0.5
    try:
        sil = max(silhouette_score(sub, KMeans(k, n_init=3, random_state=0).fit_predict(sub)) for k in (2, 3, 4)) if len(sub) > 20 else 0.0
    except Exception: sil = 0.0
    try:
        pc1 = PCA(2, random_state=0).fit_transform(sub)[:, 0]; kurt = float(kurtosis(pc1))  # low/neg kurt ~ multimodal on PC1
    except Exception: kurt = 0.0
    return {"corr_void": corr_void, "silh": float(sil), "pc1_kurt": kurt, "d": int(Xtr.shape[1]), "logn": float(np.log(len(Xtr)))}
# ---- PASS 1: per-dataset compute + incremental save ----
done = set()
if os.path.exists(JL):
    for ln in open(JL):
        try: done.add(json.loads(ln)["key"])
        except Exception: pass
f = pd.read_csv(os.path.join(D, "STREAM_FINAL2_SET.csv")); fo = open(JL, "a")
for i, (_, r) in enumerate(f.iterrows()):
    key = f"{r.corpus}__{r.dataset}"
    if key in done: continue
    try: Xtr, Xval, Xtn, Xa = get3(r.corpus, r.dataset)
    except Exception: continue
    if len(Xtr) < 40 or len(Xtn) < 10 or len(Xval) < 20 or len(Xa) < 5: continue
    Xh = harden(Xtr, np.vstack([Xval, Xtn]), Xa)
    if len(Xh) < 20: continue
    Xte = np.vstack([Xtn, Xh]); yte = np.r_[np.zeros(len(Xtn)), np.ones(len(Xh))]; base = yte.mean()
    syns = {al: gen_alpha(Xval, 200, al) for al in ALPHAS}
    names, ap, sig = [], {}, {}
    for vn, ct in TAB_POOL:
        try:
            m = ct()
            with contextlib.redirect_stdout(io.StringIO()): m.fit(Xtr)
            st = np.asarray(m.decision_function(Xte), float); sv = np.asarray(m.decision_function(Xval), float)
            if not (np.all(np.isfinite(st)) and np.nanstd(st) > 1e-12): continue
            names.append(vn); ap[vn] = float((average_precision_score(yte, st) - base) / (1 - base + 1e-12))
            row = []
            for al in ALPHAS:
                ss = np.asarray(m.decision_function(syns[al]), float); yv = np.r_[np.zeros(len(sv)), np.ones(len(ss))]
                row.append(float(roc_auc_score(yv, np.r_[sv, ss])) if np.nanstd(np.r_[sv, ss]) > 1e-12 else 0.5)
            sig[vn] = row
        except Exception: pass
    if len(names) < 6: continue
    tf = "local" if any(fam(v) == "local" for v in names) and (max([ap[v] for v in names if fam(v) == "local"], default=-9) >= max([ap[v] for v in names if fam(v) == "global"], default=-9)) else "global"
    feats = struct_features(Xtr)
    fo.write(json.dumps({"key": key, "truefam": tf, "names": names, "ap": ap, "sig": sig, "feats": feats}) + "\n"); fo.flush()
    print(f"  {i}/{len(f)} {r.dataset[:22]}", flush=True)
fo.close()
# ---- PASS 2: estimators ----
rows = [json.loads(ln) for ln in open(JL) if ln.strip()]
print(f"\nloaded {len(rows)} datasets")
def pick_at(rec, ai):
    return rec["names"][int(np.argmax([rec["sig"][v][ai] for v in rec["names"]]))]
def reg_of(rec, det): return max(rec["ap"].values()) - rec["ap"][det]
for rec in rows:
    S_ = np.array([[rec["sig"][v][a] for a in range(len(ALPHAS))] for v in rec["names"]])
    rec["_S"] = S_; rec["_lev"] = S_[:, -1].max() - S_[:, 0].max()
    rec["_oracle_ai"] = int(np.argmin([reg_of(rec, pick_at(rec, ai)) for ai in range(len(ALPHAS))]))
# feature matrix for learned estimator
FKEYS = ["corr_void", "silh", "pc1_kurt", "d", "logn"]
Xf = np.array([[rec["feats"][k] for k in FKEYS] + [rec["_lev"]] for rec in rows]); yA = np.array([rec["_oracle_ai"] for rec in rows])
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
loo_pred = np.zeros(len(rows), int)
Xfn = (Xf - Xf.mean(0)) / (Xf.std(0) + 1e-9)
for tr_i, te_i in LeaveOneOut().split(Xfn):
    rf = RandomForestRegressor(n_estimators=200, random_state=0).fit(Xfn[tr_i], yA[tr_i])
    loo_pred[te_i[0]] = int(np.clip(round(rf.predict(Xfn[te_i])[0]), 0, len(ALPHAS) - 1))
def macro(regs, fams): return np.mean([np.mean([r for r, fm in zip(regs, fams) if fm == fa]) for fa in ["local", "global"] if any(fm == fa for fm in fams)])
fams = [rec["truefam"] for rec in rows]
def evaluate(name, picker):
    regs = [reg_of(rec, picker(rec, idx)) for idx, rec in enumerate(rows)]
    print(f"  {name:26s} micro {np.mean(regs):.3f}  macro {macro(regs, fams):.3f}")
print("\n=== alpha estimators (regret vs best-alpha oracle) ===")
evaluate("binary matched", lambda rec, i: pick_at(rec, len(ALPHAS) - 1) if rec["_lev"] > TAU else pick_at(rec, 0))
evaluate("max-disagreement", lambda rec, i: pick_at(rec, int(np.argmax(rec["_S"].std(0)))))
evaluate("MARGIN (top1-top2)", lambda rec, i: pick_at(rec, int(np.argmax([np.sort(rec["_S"][:, a])[-1] - np.sort(rec["_S"][:, a])[-2] for a in range(len(ALPHAS))]))))
def consensus(rec, i):
    from collections import Counter
    c = Counter(pick_at(rec, a) for a in range(len(ALPHAS))); return c.most_common(1)[0][0]
evaluate("CONSENSUS detector", consensus)
evaluate("LEARNED alpha (LOO)", lambda rec, i: pick_at(rec, loo_pred[i]))
evaluate("best-alpha ORACLE", lambda rec, i: pick_at(rec, rec["_oracle_ai"]))
print("saved streamline/STREAM_ALPHAEST.jsonl")
