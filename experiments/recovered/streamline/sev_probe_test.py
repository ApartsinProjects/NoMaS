# -*- coding: utf-8 -*-
"""Test the severity-calibration hypothesis: does a MILDER synthetic probe (lower frac = fewer features
perturbed = closer to the real anomaly hardness) predict real detector performance better than the
default frac=0.4? Re-fit CLASSICAL detectors on the worst-failing tabular datasets, grade them on probes
at frac in {0.1,0.2,0.4} x beta in {+1,-4}, and report probe->truth rank correlation per config."""
import numpy as np, json, glob, os, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
def classical_pool():
    from pyod.models.iforest import IForest; from pyod.models.lof import LOF; from pyod.models.knn import KNN
    from pyod.models.hbos import HBOS; from pyod.models.ecod import ECOD; from pyod.models.copod import COPOD
    from pyod.models.pca import PCA as PPCA; from pyod.models.cblof import CBLOF; from pyod.models.loda import LODA
    p = [('IForest', lambda: IForest(n_estimators=100, random_state=0)), ('LOF', lambda: LOF(n_neighbors=20)), ('KNN', lambda: KNN(n_neighbors=5)),
         ('ECOD', lambda: ECOD()), ('COPOD', lambda: COPOD()), ('HBOS', lambda: HBOS(n_bins=10)), ('PCA', lambda: PPCA(n_components=0.5, random_state=0)),
         ('CBLOF', lambda: CBLOF(n_clusters=8, random_state=0)), ('LODA', lambda: LODA(n_bins=10))]
    for k in [3, 10, 35, 100, 200]: p += [(f'LOF_k{k}', lambda k=k: LOF(n_neighbors=k)), (f'KNN_k{k}', lambda k=k: KNN(n_neighbors=k))]
    for nb in [5, 20, 50]: p.append((f'HBOS_b{nb}', lambda nb=nb: HBOS(n_bins=nb)))
    for nc in [0.3, 0.7, 0.9]: p.append((f'PCA_c{nc}', lambda nc=nc: PPCA(n_components=nc, random_state=0)))
    return dict(p)
def gen_beta(Xn, ns, beta, frac, seed=0):
    rng = np.random.default_rng(seed); n, d = Xn.shape; out = np.empty((ns, d)); Hs = []
    for j in range(d):
        col = Xn[:, j]; cnt, edges = np.histogram(col, bins=30); dens = cnt / max(cnt.sum(), 1)
        lo, hi = np.percentile(col, 2), np.percentile(col, 98); ctr = (edges[:-1] + edges[1:]) / 2
        allowed = (cnt > 0) & (ctr >= lo) & (ctr <= hi); w = np.where(allowed, np.maximum(dens, 1e-6) ** beta, 0.0)
        Hs.append((edges, w / w.sum() if w.sum() > 0 else None))
    for r in range(ns):
        base = Xn[rng.integers(n)].copy()
        for j in rng.choice(d, max(1, int(frac * d)), replace=False):
            edges, w = Hs[j]; base[j] = Xn[rng.integers(n), j] if w is None else rng.uniform(*edges[[(b := rng.choice(len(w), p=w)), b + 1]])
        out[r] = base
    return out
J = {}
for fn in glob.glob('deepres_fixed/shard_*.jsonl'):
    for ln in open(fn):
        try:
            r = json.loads(ln)
            if 'ap' in r: J[r['key']] = r
        except Exception: pass
lb = pd.read_csv('STREAM_FINAL_LEADERBOARD_FIXED.csv').drop_duplicates('key', keep='last'); lb['corpus'] = lb.key.str.split('__').str[0]
tab = lb[lb.corpus.isin(['oddbench', 'ovrbench'])].copy(); tab['gap'] = tab.reg_combined - tab.reg_em
fails = list(tab.sort_values('gap', ascending=False).head(14).key)
FRACS = [0.1, 0.2, 0.4]; BETAS = [('+1', 1.0), ('-4', -4.0)]
agg = {f'{bl}_{fr}': [] for bl, _ in BETAS for fr in FRACS}
pool = classical_pool()
for key in fails:
    if key not in J or not os.path.exists(f'bundle_fixed/{key}.npz'): continue
    z = np.load(f'bundle_fixed/{key}.npz'); Xtr, Xval = z['Xtr'].astype(float), z['Xval'].astype(float)
    r = J[key]; names = [n for n in r['ap'] if not n.startswith('D_') and n in pool]
    ap = [r['ap'][n] for n in names]
    probes = {f'{bl}_{fr}': gen_beta(Xval, 200, bv, fr) for bl, bv in BETAS for fr in FRACS}
    y1 = np.r_[np.zeros(len(Xval)), np.ones(200)]
    grades = {pk: [] for pk in probes}
    for nm in names:
        m = pool[nm](); m.fit(Xtr); sv = m.decision_function(Xval)
        for pk, syn in probes.items():
            ss = m.decision_function(syn); v = np.r_[sv, ss]
            grades[pk].append(roc_auc_score(y1, v) if np.nanstd(v) > 1e-12 else 0.5)
    for pk in probes:
        rho = spearmanr(grades[pk], ap).correlation
        if np.isfinite(rho): agg[pk].append(rho)
print(f'probe->truth rank corr (mean over {len(fails)} worst-failing tabular datasets), by (beta, frac):')
print('  frac:       0.1     0.2     0.4  (0.4=current default)')
for bl, _ in BETAS:
    vals = '  '.join(f'{np.mean(agg[bl + "_" + str(fr)]):.3f}' for fr in FRACS)
    print(f'  beta={bl:>2}:   {vals}')
print('higher = probe grade better predicts real detector performance')
