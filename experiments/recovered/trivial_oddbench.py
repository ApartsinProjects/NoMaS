# -*- coding: utf-8 -*-
"""Easy/Difficult anomaly stratification on ODDBENCH (187 datasets, MacrOData-CMU, CC BY 4.0).

The decisive-scale version of the trivial-anomaly test. The local tabular pass had only 36
usable datasets and found NO family flips (0/36); OddBench gives 5x the power on the same
question, and is the paper's frozen external test set.

Trivial definition (Apartsin & Aperstein, arXiv:2607.06094 sec 5.3): an anomaly is EASY if
max_k |z_k| exceeds the Pth percentile of TRAIN-NORMAL, DIFFICULT otherwise. Statistics from
normal points only.

Protocol: STRATIFY, do not filter. For each subset the positives are that subset's anomalies
and the negatives are the FULL normal set.

Downloads are cached to data/oddbench/ so reruns are free.
"""
import os, sys, io, json, time, urllib.request, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"
sys.path.insert(0, os.path.join(ROOT, "src"))
from adrank.pipeline import fit_and_score  # noqa: E402

HF = "https://huggingface.co/datasets/MacrOData-CMU/OddBench/resolve/main/public/"
CACHE = os.path.join(ROOT, "data", "oddbench")
os.makedirs(CACHE, exist_ok=True)
DETS = ["IForest", "LOF", "KNN", "ECOD", "COPOD", "HBOS", "PCA", "CBLOF", "LODA"]
LOCAL = {"LOF", "KNN", "CBLOF"}
GLOBAL = {"HBOS", "COPOD", "ECOD", "PCA"}
fam = lambda d: "local" if d in LOCAL else ("global" if d in GLOBAL else "other")
PCTS = [95.0, 99.0, 99.5]
MAX_N = 6000
OUT = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d\trivial_oddbench.csv"


def load(name):
    """Paper's loader (scripts/modal_oddbench.py): concatenate train+test."""
    p = os.path.join(CACHE, name)
    if not os.path.exists(p):
        b = urllib.request.urlopen(HF + name + "?download=true", timeout=180).read()
        with open(p, "wb") as f:
            f.write(b)
    d = np.load(p, allow_pickle=True)
    tr, trl = np.asarray(d["train"], dtype=np.float64), np.asarray(d["train_labels"]).ravel()
    te, tel = np.asarray(d["test"], dtype=np.float64), np.asarray(d["test_labels"]).ravel()
    return np.vstack([tr, te]), np.concatenate([trl, tel]).astype(int)


names = sorted(pd.read_csv(os.path.join(ROOT, "results", "oddbench.csv"))
               .query("status == 'ok'").dataset.unique())
print(f"OddBench datasets evaluated in the paper: {len(names)}", flush=True)

rows, t0 = [], time.time()
for i, name in enumerate(names, 1):
    try:
        X, y = load(name)
    except Exception as e:
        print(f"[{i}/{len(names)}] {name}: DOWNLOAD FAILED {e}", flush=True)
        continue
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    rng = np.random.RandomState(0)
    if len(X) > MAX_N:
        keep = rng.choice(len(X), MAX_N, replace=False)
        X, y = X[keep], y[keep]
    if len(X) < 200 or y.sum() < 20 or y.sum() == len(y):
        print(f"[{i}/{len(names)}] {name}: skip (n={len(X)} anom={int(y.sum())})", flush=True)
        continue

    S = {}
    for d_ in DETS:
        s = fit_and_score(d_, X, X)
        if s is not None:
            S[d_] = s
    if len(S) < 6:
        print(f"[{i}/{len(names)}] {name}: skip (detectors failed)", flush=True)
        continue

    nm = y == 0
    mu, sd = X[nm].mean(0), X[nm].std(0) + 1e-9
    maxz = np.abs((X - mu) / sd).max(1)
    anom, norm_idx = np.where(y == 1)[0], np.where(nm)[0]

    def auc_on(idx):
        yy = np.r_[np.ones(len(idx)), np.zeros(len(norm_idx))]
        return {k: roc_auc_score(yy, np.r_[v[idx], v[norm_idx]]) for k, v in S.items()}

    all_auc = auc_on(anom)
    ba = max(all_auc, key=all_auc.get)
    for pct in PCTS:
        thr = float(np.percentile(maxz[nm], pct))
        easy = maxz[anom] > thr
        hard = anom[~easy]
        rec = dict(dataset=name, n=len(X), d=X.shape[1], pct=pct, n_anom=len(anom),
                   n_hard=int(len(hard)), frac_trivial=float(easy.mean()),
                   best_all=ba, fam_all=fam(ba), auc_best_all=all_auc[ba],
                   mean_auc_all=float(np.mean(list(all_auc.values()))),
                   spread_all=all_auc[ba] - float(np.mean(list(all_auc.values()))))
        if len(hard) >= 10:
            h = auc_on(hard)
            bh = max(h, key=h.get)
            rec.update(best_hard=bh, fam_hard=fam(bh), auc_best_hard=h[bh],
                       mean_auc_hard=float(np.mean(list(h.values()))),
                       spread_hard=h[bh] - float(np.mean(list(h.values()))),
                       auc_hard_json=json.dumps({k: round(v, 4) for k, v in h.items()}))
        rows.append(rec)
    if i % 10 == 0 or i == len(names):
        pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"[{i}/{len(names)}] {name}: n={len(X)} d={X.shape[1]} anom={len(anom)} "
          f"best={ba}({fam(ba)}) | {time.time()-t0:.0f}s", flush=True)

pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"\nwrote {OUT} rows={len(rows)} elapsed={time.time()-t0:.0f}s")
