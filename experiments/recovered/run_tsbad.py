# -*- coding: utf-8 -*-
"""Evaluate NoMaS on TSB-AD-U: an EXTERNALLY CURATED HARD benchmark.

TSB-AD (Liu & Paparrizos, NeurIPS 2024 D&B, Apache-2.0) re-curated 1,070 series from 40
source datasets specifically to remove flawed/trivial exemplars. Using it avoids building
our own hard benchmark and gives an independent, third-party difficulty guarantee.

Pipeline (uses the PAPER'S OWN time-series treatment for consistency):
  raw univariate series -> src/adrank/ts._window_features (w=64, stride=16, 28 descriptors)
  -> window labels (anomalous if the window contains >=1 anomalous point)
  -> 9-detector panel: true AUC (labels) and NoMaS pseudo-AUC (cluster holdout, no labels)
  -> regret = best true AUC - true AUC of the NoMaS pick; random-pick regret for reference

Reported per SOURCE dataset too (YAHOO / UCR / NAB / SMD ...), because TSB-AD retains
series from Yahoo, NAB and SMAP - the benchmarks Wu & Keogh urged abandoning - and we
should see whether curated-Yahoo behaves like curated-UCR.
"""
import os, sys, io, zipfile, time, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"
sys.path.insert(0, os.path.join(ROOT, "src"))
from adrank.pipeline import embed, cluster, fit_and_score  # noqa: E402
from adrank.ts import _window_features, _window_labels     # noqa: E402

DETS = ["IForest", "LOF", "KNN", "ECOD", "COPOD", "HBOS", "PCA", "CBLOF", "LODA"]
LOCAL = {"LOF", "KNN", "CBLOF"}
GLOBAL = {"HBOS", "COPOD", "ECOD", "PCA"}
fam = lambda d: "local" if d in LOCAL else ("global" if d in GLOBAL else "other")
SEED, K, M = 0, 30, 12          # M = number of held-out-cluster regimes
W, STRIDE = 64, 16
MAX_LEN = 60000                 # cap raw series length for runtime
MAX_SERIES = int(sys.argv[1]) if len(sys.argv) > 1 else 200
ZIP = os.path.join(ROOT, "data", "tsbad", "TSB-AD-U.zip")
OUT = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d\tsbad_nomas.csv"

z = zipfile.ZipFile(ZIP)
files = sorted(n for n in z.namelist() if n.lower().endswith(".csv"))
rng0 = np.random.RandomState(SEED)
if len(files) > MAX_SERIES:                      # hash-sample, NOT alphabetical (bias lesson)
    import hashlib
    files = sorted(files, key=lambda s: hashlib.sha1(s.encode()).hexdigest())[:MAX_SERIES]

rows, t0 = [], time.time()
for i, fn in enumerate(files, 1):
    base = os.path.basename(fn)
    source = base.split("_")[1] if "_" in base else "?"
    try:
        df = pd.read_csv(io.BytesIO(z.read(fn)))
        x = df.iloc[:, 0].to_numpy(dtype=float)
        lab = df.iloc[:, 1].to_numpy(dtype=int)
    except Exception as e:
        print(f"[{i}/{len(files)}] {base}: parse fail {e}", flush=True); continue
    if len(x) > MAX_LEN:
        x, lab = x[:MAX_LEN], lab[:MAX_LEN]
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        Xw, starts = _window_features(x, w=W, stride=STRIDE)
        yw = _window_labels(lab, starts, w=W, min_count=1)
    except Exception as e:
        print(f"[{i}/{len(files)}] {base}: window fail {e}", flush=True); continue
    Xw = np.nan_to_num(Xw, nan=0.0, posinf=0.0, neginf=0.0)
    na = int(yw.sum())
    nnorm = int((yw == 0).sum())
    # A contiguous anomaly region makes every OVERLAPPING window anomalous, so on
    # high-density series almost all windows are positive and too few normals remain to
    # cluster. That is Wu & Keogh's "unrealistic anomaly density" flaw; count it.
    if len(Xw) < 200 or na < 10 or nnorm < 150:
        print(f"[{i}/{len(files)}] {base}: skip (win={len(Xw)} anom={na} norm={nnorm} "
              f"win_anom_rate={na/max(len(yw),1):.3f})", flush=True)
        rows_skipped.append(dict(series=base, source=source, n_windows=len(Xw),
                                 n_anom_win=na, n_norm_win=nnorm,
                                 win_anom_rate=na / max(len(yw), 1), reason="density"))
        continue

    true = {}
    for d in DETS:
        s = fit_and_score(d, Xw, Xw)
        if s is not None:
            true[d] = float(roc_auc_score(yw, s))
    if len(true) < 6:
        print(f"[{i}/{len(files)}] {base}: detectors failed", flush=True); continue

    # ---- NoMaS: cluster the NORMAL windows, hold clusters out, no labels used ----
    Xn = Xw[yw == 0]
    Z = embed(Xn, dim=16)
    labc = cluster(Z, K, seed=SEED)
    acc = {d: [] for d in DETS}
    used = 0
    rng = np.random.RandomState(SEED)
    for j in np.unique(labc):
        if used >= M:
            break
        m = labc == j
        nj = int(m.sum())
        if nj < 15 or nj > 0.25 * len(Xn):
            continue
        rest = np.where(~m)[0]; rng.shuffle(rest)
        nh = max(20, int(0.2 * len(rest)))
        hold, tr = rest[:nh], rest[nh:]
        Xsc = np.vstack([Xn[m], Xn[hold]])
        yt = np.r_[np.ones(nj), np.zeros(len(hold))]
        for d in DETS:
            s = fit_and_score(d, Xn[tr], Xsc)
            if s is not None and d in true:
                acc[d].append(roc_auc_score(yt, s))
        used += 1
    score = {d: float(np.mean(v)) for d, v in acc.items() if v}
    if len(score) < 5 or used < 3:
        print(f"[{i}/{len(files)}] {base}: too few regimes ({used})", flush=True); continue

    best = max(true.values())
    pick = max(score, key=score.get)
    reg = best - true[pick]
    rand = best - float(np.mean(list(true.values())))
    bd = max(true, key=true.get)
    rows.append(dict(series=base, source=source, n_windows=len(Xw), n_anom_win=na,
                     n_regimes=used, best_det=bd, best_fam=fam(bd), best_auc=best,
                     pick=pick, pick_fam=fam(pick), regret=reg, random_regret=rand,
                     spread=best - float(np.mean(list(true.values())))))
    if i % 20 == 0:
        pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"[{i}/{len(files)}] {base[:44]} src={source} best={bd} pick={pick} "
          f"regret={reg:.4f} rand={rand:.4f} | {time.time()-t0:.0f}s", flush=True)

pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"\nwrote {OUT} rows={len(rows)} elapsed={time.time()-t0:.0f}s")
