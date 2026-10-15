# -*- coding: utf-8 -*-
"""Are the GLOBAL MV-wins datasets single-feature-RANK-trivial after hardening? Compute best single-
feature ap_norm on the hardened test vs the best-detector oracle. Small gap => a single feature ranks
the anomalies as well as any detector => rank-trivial (survives severity-hardening because 'consistently
elevated' != 'top-5% extreme'). Answers whether to add a single-feature-ap_norm triviality filter."""
import os, sys, warnings
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"; S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
D = os.path.join(S, "scratchpad", "streamline")
sys.path.insert(0, os.path.join(S, "scratchpad")); sys.path.insert(0, os.path.join(ROOT, "src"))
import adrank.pipeline as PP
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
def harden(Xtr, Xhold, Xa):
    tho = np.quantile(severity(Xtr, Xhold), 1 - Q); so_a = severity(Xtr, Xa)
    try:
        sc = StandardScaler().fit(Xtr); pca = PCA(n_components=0.95, whiten=True, random_state=0).fit(sc.transform(Xtr))
        Ptr, Ph, Pa = pca.transform(sc.transform(Xtr)), pca.transform(sc.transform(Xhold)), pca.transform(sc.transform(Xa))
        thp = np.quantile(severity(Ptr, Ph), 1 - Q); sp_a = severity(Ptr, Pa)
    except Exception:
        sp_a = np.zeros(len(Xa)); thp = np.inf
    return Xa[~((so_a > tho) | (sp_a > thp))]
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
rk = pd.read_csv(os.path.join(D, "STREAM_RANK.csv")); sc = pd.read_csv(os.path.join(D, "STREAM_SCORE2.csv"))[["corpus", "dataset", "oracle_apnorm"]]
g = rk[rk.truefam == "global"].merge(sc, on=["corpus", "dataset"], how="left"); g["gap"] = g.reg_matched - g.reg_mv
sel = g.sort_values("gap", ascending=False).head(15); rows = []
for _, r in sel.iterrows():
    try: Xtr, Xval, Xtn, Xa = get4(r.corpus, r.dataset)
    except Exception: continue
    if len(Xtr) < 40 or len(Xtn) < 10 or len(Xa) < 10 or r.oracle_apnorm != r.oracle_apnorm: continue
    Xh = harden(Xtr, np.vstack([Xval, Xtn]), Xa)
    if len(Xh) < 10: continue
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9; Zn = np.abs((Xtn - mu) / sd); Za = np.abs((Xh - mu) / sd); y = np.r_[np.zeros(len(Zn)), np.ones(len(Za))]
    f1 = max((apn(y, np.r_[Zn[:, j], Za[:, j]]) for j in range(Xtr.shape[1]) if np.std(np.r_[Zn[:, j], Za[:, j]]) > 1e-9), default=0.0)
    rows.append({"dataset": r.dataset[:22], "d": Xtr.shape[1], "oracle_ap": round(r.oracle_apnorm, 2),
                 "best_1feat_ap": round(f1, 2), "gap": round(r.oracle_apnorm - f1, 2), "reg_matched": round(r.reg_matched, 2), "reg_mv": round(r.reg_mv, 2)})
df = pd.DataFrame(rows); df.to_csv(os.path.join(D, "STREAM_DEEP_GLOBAL.csv"), index=False)
print(f"=== single-feature ap_norm on HARDENED anomalies vs oracle - global MV-wins datasets ({len(df)}) ===")
print("  gap = oracle_ap - best_1feat_ap;  small gap => a single feature ranks as well as any detector = RANK-TRIVIAL\n")
print(df.sort_values("gap").to_string(index=False))
triv = df[df.gap < 0.05]
print(f"\n  single-feature-rank-trivial (gap<0.05): {len(triv)}/{len(df)}  -> {list(triv.dataset)}")
print(f"  mean gap {df.gap.mean():.2f}  (low = these globals are single-feature-separable despite severity-hardening)")
print("saved streamline/STREAM_DEEP_GLOBAL.csv")
