# -*- coding: utf-8 -*-
"""DEEP GATE (tabular): before dropping 'all-bad' datasets (best shallow ap_norm<=0.10), test whether
DEEP detectors crack them. Same hardened test as the pool. Deep models: DeepOD DeepSVDD + Deep Isolation
Forest, PyOD AutoEncoder + VAE. Keep a dataset if any deep model lifts ap_norm meaningfully above shallow."""
import os, sys, io, contextlib, warnings
import numpy as np, pandas as pd
os.environ["PYTHONWARNINGS"] = "ignore"
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score
warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"; S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
D = os.path.join(S, "scratchpad", "streamline"); sys.path.insert(0, os.path.join(S, "scratchpad")); sys.path.insert(0, os.path.join(ROOT, "src"))
import adrank.pipeline as PP
Q = 0.05; DEV = "cpu"
OBJ = {}
for sub in ("adbench", "dami"):
    dd = os.path.join(ROOT, "data", sub)
    if os.path.isdir(dd):
        for ds in PP.load_npz_dir(dd): OBJ[(sub, str(ds.name)[:40])] = (np.nan_to_num(np.asarray(ds.X, float)), np.asarray(ds.y, int).ravel())
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
    else:
        X, y = OBJ[(corp, name)]; Xn, Xa = X[y == 0], X[y == 1]
    r = np.random.default_rng(0)
    if len(Xn) > 6000: Xn = Xn[r.choice(len(Xn), 6000, replace=False)]
    idx = np.arange(len(Xn)); r.shuffle(idx); a, b = int(0.6 * len(idx)), int(0.8 * len(idx)); return Xn[idx[:a]], Xn[idx[a:b]], Xn[idx[b:]], Xa
def apn(y, s):
    bb = y.mean(); return (average_precision_score(y, s) - bb) / (1 - bb + 1e-12)
def make_deep():
    M = []
    try:
        from deepod.models import DeepSVDD; M.append(("DeepSVDD", lambda: DeepSVDD(epochs=50, device=DEV, random_state=0)))
    except Exception as e: print("DeepSVDD import fail", e)
    try:
        from deepod.models import ICL; M.append(("ICL", lambda: ICL(epochs=50, device=DEV, random_state=0)))
    except Exception as e: print("ICL import fail", e)
    try:
        from deepod.models import RCA; M.append(("RCA", lambda: RCA(epochs=50, device=DEV, random_state=0)))
    except Exception as e: print("RCA import fail", e)
    try:
        from pyod.models.auto_encoder import AutoEncoder; M.append(("AE", lambda: AutoEncoder(epoch_num=60, verbose=0)))
    except Exception as e: print("AE import fail", e)
    try:
        from pyod.models.vae import VAE; M.append(("VAE", lambda: VAE(epoch_num=60, verbose=0)))
    except Exception as e: print("VAE import fail", e)
    return M
DEEP = make_deep(); print("deep models:", [n for n, _ in DEEP], flush=True)
sp = pd.read_csv(os.path.join(D, "STREAM_POOL_SPREAD.csv")); ab = sp[(sp.best <= 0.10) & (sp.corpus != "tsbad_m")]
print(f"tabular all-bad datasets: {len(ab)}\n", flush=True)
rows = []
for i, (_, r) in enumerate(ab.iterrows()):
    try: Xtr, Xval, Xtn, Xa = get3(r.corpus, r.dataset)
    except Exception: continue
    if len(Xtr) < 40 or len(Xtn) < 10 or len(Xa) < 5: continue
    Xh = harden(Xtr, np.vstack([Xval, Xtn]), Xa)
    if len(Xh) < 20: continue
    Xte = np.vstack([Xtn, Xh]); yte = np.r_[np.zeros(len(Xtn)), np.ones(len(Xh))]
    ss = StandardScaler().fit(Xtr); Ztr, Zte = ss.transform(Xtr), ss.transform(Xte)
    dres = {}
    for nm, ctor in DEEP:
        try:
            m = ctor()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                m.fit(Ztr); sc_ = np.asarray(m.decision_function(Zte), float)
            dres[nm] = apn(yte, sc_) if np.all(np.isfinite(sc_)) and np.nanstd(sc_) > 1e-12 else np.nan
        except Exception: dres[nm] = np.nan
    deepbest = np.nanmax(list(dres.values())) if any(np.isfinite(list(dres.values()))) else np.nan
    rec = {"corpus": r.corpus, "dataset": r.dataset[:24], "shallow_best": round(r.best, 3), "deep_best": round(float(deepbest), 3),
           "lift": round(float(deepbest - r.best), 3)}
    rec.update({k: round(float(v), 3) for k, v in dres.items()})
    rows.append(rec)
    print(f"  {i:2d} {r.dataset[:24]:26s} shallow {r.best:+.3f} -> deep {deepbest:+.3f}  lift {deepbest-r.best:+.3f}", flush=True)
df = pd.DataFrame(rows); df.to_csv(os.path.join(D, "STREAM_DEEP_GATE_TAB.csv"), index=False)
cracked = df[df.deep_best > 0.10]; lifted = df[df.lift > 0.05]
print(f"\n=== DEEP GATE (tabular, {len(df)} all-bad datasets) ===")
print(f"  deep lifts ceiling above 0.10 on: {len(cracked)}/{len(df)}  -> {list(cracked.dataset)}")
print(f"  deep improves ap_norm by >0.05 on: {len(lifted)}/{len(df)}")
print(f"  median deep_best {df.deep_best.median():.3f} vs shallow_best {df.shallow_best.median():.3f}   mean lift {df.lift.mean():+.3f}")
print("saved streamline/STREAM_DEEP_GATE_TAB.csv")
