# -*- coding: utf-8 -*-
"""Build HADB - the Hard Anomaly Detection Benchmark.

A unified collection of tabular anomaly-detection datasets with TRIVIAL anomalies removed,
so that detector ranking is measured on operationally meaningful anomalies rather than on
easy single-feature excursions.

TRIVIAL DEFINITION (adopted verbatim from Apartsin & Aperstein, "Modeling Normal Is All You
Need", arXiv:2607.06094, sec. 5.3): a trivial detector scores each point by the maximum
absolute standardized per-feature value, max_k |z_k|. An anomaly is EASY (trivial) if this
score exceeds the Pth percentile of TRAIN-NORMAL, and DIFFICULT otherwise. Statistics come
from normal points only, so a cluster of anomalies cannot inflate its own threshold.

HANDLING: trivial anomaly ROWS ARE DROPPED, not relabelled as normal. Relabelling would
contaminate the normal class and change what every detector fits. Dropping yields a
self-consistent standalone benchmark. (The source paper instead STRATIFIES, keeping all
normals as negatives; that evaluation remains available from the manifest.)

INCLUSION CRITERIA (all must hold):
  >= MIN_HARD difficult anomalies retained     (paper's WADI has 19, so 20 is the floor)
  >= 200 rows after filtering
  retained anomaly rate within [0.001, 0.25]
  >= 6 detectors run successfully

Outputs:
  <out>/<dataset>.npz      filtered X, y
  <out>/HADB_manifest.csv  per-dataset provenance + baseline detector AUCs
"""
import os, sys, glob, json, time, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"
sys.path.insert(0, os.path.join(ROOT, "src"))
from adrank.pipeline import fit_and_score  # noqa: E402

DETS = ["IForest", "LOF", "KNN", "ECOD", "COPOD", "HBOS", "PCA", "CBLOF", "LODA"]
LOCAL = {"LOF", "KNN", "CBLOF"}
GLOBAL = {"HBOS", "COPOD", "ECOD", "PCA"}
fam = lambda d: "local" if d in LOCAL else ("global" if d in GLOBAL else "other")

PCT = float(sys.argv[1]) if len(sys.argv) > 1 else 99.0
MIN_HARD = 20
MIN_ROWS = 200
RATE_LO, RATE_HI = 0.001, 0.25
OUTDIR = os.path.join(ROOT, "data", "hadb")
os.makedirs(OUTDIR, exist_ok=True)

SETS = []
for sub in ("dami", "adbench"):
    for p in sorted(glob.glob(os.path.join(ROOT, "data", sub, "*.npz"))):
        SETS.append((sub, os.path.splitext(os.path.basename(p))[0], p))

rows, t0 = [], time.time()
for sub, name, path in SETS:
    a = np.load(path)
    X, y = np.asarray(a["X"], float), np.asarray(a["y"], int).ravel()
    n0, d0, na0 = len(X), X.shape[1], int(y.sum())
    rec = dict(dataset=name, source=sub, n_orig=n0, d=d0, n_anom_orig=na0,
               rate_orig=round(na0 / max(n0, 1), 5), pct=PCT)

    if n0 < MIN_ROWS or na0 < 5:
        rec.update(status="excluded", reason="too_small")
        rows.append(rec); continue

    nm = y == 0
    Xn = X[nm]
    mu, sd = Xn.mean(0), Xn.std(0) + 1e-9
    maxz = np.abs((X - mu) / sd).max(1)
    thr = float(np.percentile(maxz[nm], PCT))
    anom = np.where(y == 1)[0]
    easy = maxz[anom] > thr
    hard_idx = anom[~easy]
    rec.update(trivial_thr=round(thr, 4), n_trivial=int(easy.sum()),
               frac_trivial=round(float(easy.mean()), 4), n_hard=int(len(hard_idx)))

    if len(hard_idx) < MIN_HARD:
        rec.update(status="excluded", reason=f"only_{len(hard_idx)}_hard_anomalies")
        rows.append(rec); continue

    keep = np.r_[np.where(nm)[0], hard_idx]
    keep.sort()
    Xh, yh = X[keep], y[keep]
    rate = float(yh.sum() / len(yh))
    rec.update(n_hadb=int(len(yh)), n_anom_hadb=int(yh.sum()), rate_hadb=round(rate, 5))
    if len(yh) < MIN_ROWS or not (RATE_LO <= rate <= RATE_HI):
        rec.update(status="excluded", reason=f"rate_{rate:.4f}_or_too_few_rows")
        rows.append(rec); continue

    # baseline detector AUCs ON HADB (unsupervised: fit on the filtered data itself)
    auc = {}
    for dd in DETS:
        s = fit_and_score(dd, Xh, Xh)
        if s is not None:
            auc[dd] = float(roc_auc_score(yh, s))
    if len(auc) < 6:
        rec.update(status="excluded", reason="detectors_failed")
        rows.append(rec); continue

    b = max(auc, key=auc.get)
    mean_auc = float(np.mean(list(auc.values())))
    rec.update(status="included", best=b, best_fam=fam(b), best_auc=round(auc[b], 4),
               mean_auc=round(mean_auc, 4), spread=round(auc[b] - mean_auc, 4),
               auc_json=json.dumps({k: round(v, 4) for k, v in auc.items()}))
    np.savez_compressed(os.path.join(OUTDIR, name + ".npz"), X=Xh, y=yh)
    rows.append(rec)
    print(f"[keep] {name}: {n0}->{len(yh)} rows, anom {na0}->{int(yh.sum())} "
          f"(trivial {int(easy.sum())} dropped), best={b}({fam(b)}) "
          f"spread={auc[b]-mean_auc:.3f} | {time.time()-t0:.0f}s", flush=True)

M = pd.DataFrame(rows)
M.to_csv(os.path.join(OUTDIR, "HADB_manifest.csv"), index=False)
inc = M[M.status == "included"]
print(f"\n=== HADB v1 (trivial threshold = {PCT}th pct of train-normal) ===")
print(f"  datasets scanned : {len(M)}")
print(f"  INCLUDED         : {len(inc)}")
print(f"  excluded         : {int((M.status=='excluded').sum())}")
if len(M[M.status == 'excluded']):
    print(M[M.status == "excluded"].reason.value_counts().to_string())
if len(inc):
    print(f"\n  anomalies dropped as trivial: {int(inc.n_trivial.sum())} of "
          f"{int(inc.n_anom_orig.sum())} ({100*inc.n_trivial.sum()/inc.n_anom_orig.sum():.1f}%)")
    print(f"  mean best-detector AUC on HADB : {inc.best_auc.mean():.3f}")
    print(f"  mean best-vs-mean spread       : {inc.spread.mean():.4f}")
    print(f"  winning family distribution    : {inc.best_fam.value_counts().to_dict()}")
print(f"\nwrote {OUTDIR}")
