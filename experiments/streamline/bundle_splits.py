# -*- coding: utf-8 -*-
"""Bundle per-dataset hardened splits (Xtr, Xtn=test-normals, Xh=hard-anomalies) for all 174 datasets
into compact .npz files, so a Modal harness can fit classical+deep detectors on the SAME hardened test
(construct-matched). Same get3/harden as pool_spread. Tabular + MTS use the window-feature space."""
import os, sys, warnings
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"; S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
D = os.path.join(S, "scratchpad", "streamline"); OUTB = os.path.join(D, "bundle")
os.makedirs(OUTB, exist_ok=True)
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
def harden(Xtr, Xhold, Xa):
    tho = np.quantile(severity(Xtr, Xhold), 1 - Q); so_a = severity(Xtr, Xa)
    try:
        sc = StandardScaler().fit(Xtr); pca = PCA(n_components=0.95, whiten=True, random_state=0).fit(sc.transform(Xtr))
        Ptr, Ph, Pa = pca.transform(sc.transform(Xtr)), pca.transform(sc.transform(Xhold)), pca.transform(sc.transform(Xa))
        thp = np.quantile(severity(Ptr, Ph), 1 - Q); sp_a = severity(Ptr, Pa)
    except Exception:
        sp_a = np.zeros(len(Xa)); thp = np.inf
    return Xa[~((so_a > tho) | (sp_a > thp))]
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
f = pd.read_csv(os.path.join(D, "STREAM_FINAL2_SET.csv")); n_ok = 0; idx_rows = []
for i, (_, r) in enumerate(f.iterrows()):
    try: Xtr, Xval, Xtn, Xa = get3(r.corpus, r.dataset)
    except Exception: continue
    if len(Xtr) < 40 or len(Xtn) < 10 or len(Xa) < 5: continue
    Xh = harden(Xtr, np.vstack([Xval, Xtn]), Xa)
    if len(Xh) < 20: continue
    key = f"{r.corpus}__{str(r.dataset)[:36]}".replace("/", "_").replace(" ", "_")
    np.savez_compressed(os.path.join(OUTB, key + ".npz"), Xtr=Xtr.astype(np.float32), Xval=Xval.astype(np.float32), Xtn=Xtn.astype(np.float32), Xh=Xh.astype(np.float32))
    idx_rows.append({"key": key, "corpus": r.corpus, "dataset": r.dataset, "d": Xtr.shape[1], "n_tr": len(Xtr), "n_val": len(Xval), "n_tn": len(Xtn), "n_h": len(Xh)}); n_ok += 1
    if i % 30 == 0: print(f"  ..{i}/{len(f)} bundled {n_ok}", flush=True)
pd.DataFrame(idx_rows).to_csv(os.path.join(OUTB, "_index.csv"), index=False)
tot = sum(os.path.getsize(os.path.join(OUTB, x)) for x in os.listdir(OUTB)) / 1e6
print(f"\nbundled {n_ok} datasets into {OUTB}  ({tot:.0f} MB total)")
print("saved bundle/_index.csv")
