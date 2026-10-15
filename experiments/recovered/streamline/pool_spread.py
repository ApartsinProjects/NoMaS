# -*- coding: utf-8 -*-
"""How many datasets carry NO model-selection signal? For each dataset compute the pool ap_norm
distribution on the hardened test (best/worst/mean/spread). 'Every detector very good' = worst >= GOOD;
'every detector very bad' = best <= BAD; 'selection moot' = best-mean < FLAT. A selection benchmark
should keep only datasets where the detector choice matters. Count them and show the leaderboard if dropped."""
import os, sys, io, contextlib, warnings
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score
warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"; S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
D = os.path.join(S, "scratchpad", "streamline")
sys.path.insert(0, os.path.join(S, "scratchpad")); sys.path.insert(0, os.path.join(ROOT, "src"))
import adrank.pipeline as PP
from dev_common import TAB_POOL
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
f = pd.read_csv(os.path.join(D, "STREAM_FINAL2_SET.csv")); rk = pd.read_csv(os.path.join(D, "STREAM_RANK.csv"))[["corpus", "dataset", "truefam"]]
rows = []
for i, (_, r) in enumerate(f.iterrows()):
    try: Xtr, Xval, Xtn, Xa = get3(r.corpus, r.dataset)
    except Exception: continue
    if len(Xtr) < 40 or len(Xtn) < 10 or len(Xa) < 5: continue
    Xh = harden(Xtr, np.vstack([Xval, Xtn]), Xa)
    if len(Xh) < 20: continue
    Xte = np.vstack([Xtn, Xh]); yte = np.r_[np.zeros(len(Xtn)), np.ones(len(Xh))]; base = yte.mean(); aps = []
    for vn, ct in TAB_POOL:
        try:
            m = ct()
            with contextlib.redirect_stdout(io.StringIO()): m.fit(Xtr)
            st = np.asarray(m.decision_function(Xte), float)
            if np.all(np.isfinite(st)) and np.nanstd(st) > 1e-12: aps.append((average_precision_score(yte, st) - base) / (1 - base + 1e-12))
        except Exception: pass
    if len(aps) < 6: continue
    a = np.array(aps); rows.append({"corpus": r.corpus, "dataset": r.dataset, "best": a.max(), "worst": a.min(),
                 "mean": a.mean(), "p10": np.percentile(a, 10), "p90": np.percentile(a, 90), "spread": a.max() - a.min(), "oracle_gain": a.max() - a.mean()})
    if i % 30 == 0: print(f"  ..{i}/{len(f)} {len(rows)} kept", flush=True)
df = pd.DataFrame(rows).merge(rk, on=["corpus", "dataset"], how="left"); df.to_csv(os.path.join(D, "STREAM_POOL_SPREAD.csv"), index=False)
N = len(df)
print(f"\n=== pool ap_norm spread across {N} datasets ===")
print(f"  best (ceiling):  median {df.best.median():.2f}   worst (floor): median {df.worst.median():.2f}   oracle_gain(best-mean): median {df.oracle_gain.median():.2f}")
for GOOD in (0.7, 0.8):
    print(f"  EVERY detector very good (worst>={GOOD}): {int((df.worst>=GOOD).sum())}")
for BAD in (0.05, 0.10, 0.15):
    print(f"  EVERY detector very bad  (best <={BAD}): {int((df.best<=BAD).sum())}")
for FLAT in (0.03, 0.05, 0.10):
    print(f"  selection MOOT (oracle_gain<{FLAT}):     {int((df.oracle_gain<FLAT).sum())}")
# candidate drop: all-good OR all-bad OR flat, at moderate thresholds
GOOD, BAD, FLAT = 0.8, 0.10, 0.05
drop = (df.worst >= GOOD) | (df.best <= BAD) | (df.oracle_gain < FLAT)
print(f"\n  COMBINED drop (worst>={GOOD} OR best<={BAD} OR gain<{FLAT}): {int(drop.sum())}/{N}  "
      f"(allgood {int((df.worst>=GOOD).sum())}, allbad {int((df.best<=BAD).sum())}, flat {int((df.oracle_gain<FLAT).sum())})")
print(f"  by family among dropped: {df[drop].truefam.value_counts().to_dict()}")
keep = df[~drop]
print(f"  kept: {len(keep)}  ({int((keep.truefam=='local').sum())} local / {int((keep.truefam=='global').sum())} global)")
# leaderboard effect on the kept subset
full = pd.read_csv(os.path.join(D, "STREAM_RANK.csv")); kept_keys = set(zip(keep.corpus, keep.dataset))
sub = full[[ (c, d) in kept_keys for c, d in zip(full.corpus, full.dataset)]]
def macro(frame, c): return np.mean([frame[frame.truefam == fa][c].mean() for fa in ["local", "global"] if (frame.truefam == fa).any()])
print(f"\n  --- leaderboard on selection-informative subset ({len(sub)}) ---")
e, mv = sub.reg_em.values, sub.reg_mv.values
for nm, c in [("NoMaS matched", "reg_matched"), ("EM", "reg_em"), ("MV", "reg_mv"), ("random", "reg_random")]:
    a = sub[c].values; pm = wilcoxon(a, mv).pvalue if c != "reg_mv" and (a != mv).any() else np.nan
    print(f"    {nm:16s} micro {a.mean():.3f}  macro {macro(sub,c):.3f}  vsMV {pm:.3f}")
print("saved streamline/STREAM_POOL_SPREAD.csv")
