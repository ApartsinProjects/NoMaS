# -*- coding: utf-8 -*-
"""Relate best-alpha (anomaly type) to normal VOID structure and anomaly PLACEMENT. Per dataset:
 best_alpha (oracle, from STREAM_ALPHAEST.jsonl signatures) ; n_modes = GMM-BIC clusters in normals
 (interior voids) ; anomaly placement fractions: hull_ext (a feature beyond normal Q1-Q99 -> marginal),
 void_fill (inside envelope but nearest-normal > normal spacing -> combination), embedded (nn<=spacing).
Hypothesis: high alpha <-> void-filling anomalies + multimodal normals; low alpha <-> hull-exterior."""
import os, sys, json, warnings
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"; S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
D = os.path.join(S, "scratchpad", "streamline"); sys.path.insert(0, os.path.join(S, "scratchpad")); sys.path.insert(0, os.path.join(ROOT, "src"))
import adrank.pipeline as PP
from hadb_ts_final import W, STRIDE, block_split3
from hadb_ts_mts import mts_window_features, window_labels as mts_wlabels, load_mts
Q = 0.05; ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]; OUT = os.path.join(D, "STREAM_VOID.csv")
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
# oracle alpha per key from the estimator signatures
ORC = {}
jl = os.path.join(D, "STREAM_ALPHAEST.jsonl")
if os.path.exists(jl):
    for ln in open(jl):
        r = json.loads(ln); names = r["names"]; ap = r["ap"]; sig = r["sig"]; best = max(ap.values())
        def pick(ai): return names[int(np.argmax([sig[v][ai] for v in names]))]
        oai = int(np.argmin([best - ap[pick(ai)] for ai in range(len(ALPHAS))])); ORC[r["key"]] = ALPHAS[oai]
f = pd.read_csv(os.path.join(D, "STREAM_FINAL2_SET.csv"))
done = set(pd.read_csv(OUT).dataset) if os.path.exists(OUT) else set(); fo = open(OUT, "a")
if not done: fo.write("corpus,dataset,best_alpha,n_modes,void_mass,ll_gap,hull_ext,void_fill,embedded\n"); fo.flush()
for _, r in f.iterrows():
    key = f"{r.corpus}__{r.dataset}"
    if r.dataset in done or key not in ORC: continue
    try: Xtr, Xval, Xtn, Xa = get3(r.corpus, r.dataset)
    except Exception: continue
    if len(Xtr) < 40 or len(Xa) < 5: continue
    Xh = harden(Xtr, np.vstack([Xval, Xtn]), Xa)
    if len(Xh) < 20: continue
    sc = StandardScaler().fit(Xtr); Ztr = sc.transform(Xtr); Zh = sc.transform(Xh)
    rng = np.random.default_rng(0); sub = Ztr[rng.choice(len(Ztr), min(len(Ztr), 1500), replace=False)]
    scale = np.median(NearestNeighbors(n_neighbors=2).fit(sub).kneighbors(sub)[0][:, 1]) + 1e-9
    dd = NearestNeighbors(n_neighbors=1).fit(sub).kneighbors(Zh)[0][:, 0]
    qlo, qhi = np.percentile(Xtr, 1, 0), np.percentile(Xtr, 99, 0)
    outside = ((Xh < qlo) | (Xh > qhi)).any(1)                     # any feature beyond normal Q1-Q99 envelope
    inside = ~outside
    hull_ext = float(outside.mean())
    void_fill = float((inside & (dd > scale)).mean())              # inside envelope, low-density interior gap
    embedded = float((inside & (dd <= scale)).mean())              # inside envelope, near a normal
    try:
        models = [GaussianMixture(k, covariance_type="diag", random_state=0, reg_covar=1e-4).fit(sub) for k in range(1, 9)]
        n_modes = int(np.argmin([m.bic(sub) for m in models]) + 1)
        g1, gk = models[0], models[n_modes - 1]
        # void_mass: fraction of single-Gaussian mass landing where the mixture says low-density (<5th pct of real normals)
        samp, _ = g1.sample(2000); thr = np.quantile(gk.score_samples(sub), 0.05)
        void_mass = float(np.mean(gk.score_samples(samp) < thr))
        ll_gap = float(gk.score(sub) - g1.score(sub))              # mean log-lik improvement of mixture over single Gaussian
    except Exception:
        n_modes, void_mass, ll_gap = 1, 0.0, 0.0
    fo.write(f"{r.corpus},{r.dataset},{ORC[key]:.2f},{n_modes},{void_mass:.3f},{ll_gap:.3f},{hull_ext:.3f},{void_fill:.3f},{embedded:.3f}\n"); fo.flush()
    print(f"  {r.dataset[:22]} a*={ORC[key]:.1f} modes={n_modes} voidmass={void_mass:.2f} hull={hull_ext:.2f} void={void_fill:.2f}", flush=True)
fo.close()
df = pd.read_csv(OUT)
print(f"\n=== best-alpha vs void structure & anomaly placement ({len(df)} datasets) ===")
print(f"  mean: best_alpha {df.best_alpha.mean():.2f}  n_modes {df.n_modes.mean():.1f}  void_mass {df.void_mass.mean():.2f}  ll_gap {df.ll_gap.mean():.2f}  hull_ext {df.hull_ext.mean():.2f}  void_fill {df.void_fill.mean():.2f}  embedded {df.embedded.mean():.2f}\n")
for a, b in [("best_alpha", "void_mass"), ("best_alpha", "ll_gap"), ("best_alpha", "n_modes"), ("best_alpha", "void_fill"), ("best_alpha", "hull_ext"), ("best_alpha", "embedded"), ("void_mass", "void_fill"), ("n_modes", "void_fill")]:
    rho, p = spearmanr(df[a], df[b]); print(f"  Spearman({a:11s}, {b:10s}) = {rho:+.2f}  (p={p:.3f})")
print("\n  hypothesis: best_alpha POS with void_mass/void_fill/n_modes (combination/local), NEG with hull_ext (marginal/global)")
print("saved streamline/STREAM_VOID.csv")
