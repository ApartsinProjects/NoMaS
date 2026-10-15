# -*- coding: utf-8 -*-
"""HADB time-series arm, final: TSB-AD curated data + TSB-AD's own 28-detector library.

WHY THIS SUPERSEDES THE EARLIER ATTEMPTS:
  - Detectors are SUBSEQUENCE-NATIVE (Sub_LOF, Sub_KNN, Sub_IForest, Sub_PCA, Sub_HBOS)
    plus genuinely TS-specific methods (MatrixProfile, NORMA, SR, FFT, POLY, Series2Graph),
    from the same group that curated the data. My earlier arm hand-rolled 28 summary
    statistics and ran TABULAR detectors on them - the paper's own acknowledged weak point
    ("a richer window descriptor is needed than a handful of summary statistics").
  - Scores are PER-POINT, so evaluation is at point level against point labels. No
    windowing, no window-label aggregation, no descriptor bottleneck.
  - The candidate pool is modality-specific, which is legitimate: the selection method
    ranks whatever candidates it is given, and different datasets may carry different pools.

FILTERS (same chain as the tabular arm where meaningful):
  S1 TRIVIALITY = Wu & Keogh's one-liner test at the SERIES level (their Definition 1),
     the canonical criterion for time series. Not max|z|, which is a tabular tool.
  S2 enough non-trivial anomalies must remain.
  S3 evaluate the pool; TSB-AD filenames encode a normal-only training region (`_tr_N_`),
     so scores are read on the TEST portion only.
  S4/S5 drop ceiling, rescue floor by pool expansion.
  S7 keep only datasets with real spread.

NOTE ON PROTOCOL: TSB-AD's unsupervised wrapper scores the whole series transductively,
which is that community's standard and matches TSB-AD's own benchmark. Metrics are computed
on the post-train-split portion only.
"""
import os, re, sys, io, zipfile, hashlib, time, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"
ZIP = os.path.join(ROOT, "data", "tsbad", "TSB-AD-U.zip")
N_SERIES = int(sys.argv[1]) if len(sys.argv) > 1 else 120
MAX_LEN, MIN_ANOM = 30000, 20
OUT = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d\hadb_ts_tsbad.csv"

from TSB_AD.model_wrapper import run_Unsupervise_AD


def POOL(period):
    """Subsequence-native + TS-specific + tabular-on-raw, with the window/period swept."""
    p = []
    for per in (period, max(4, period // 2), period * 2):
        p += [(f"Sub_LOF_p{per}", "subseq", "Sub_LOF", dict(periodicity=per, n_neighbors=20)),
              (f"Sub_KNN_p{per}", "subseq", "Sub_KNN", dict(periodicity=per, n_neighbors=10)),
              (f"Sub_IForest_p{per}", "subseq", "Sub_IForest", dict(periodicity=per)),
              (f"Sub_PCA_p{per}", "subseq", "Sub_PCA", dict(periodicity=per)),
              (f"Sub_HBOS_p{per}", "subseq", "Sub_HBOS", dict(periodicity=per)),
              (f"MatrixProfile_p{per}", "ts_native", "MatrixProfile", dict(periodicity=per))]
    p += [("NORMA", "ts_native", "NORMA", dict(periodicity=period)),
          ("SR", "ts_native", "SR", {}), ("FFT", "ts_native", "FFT", {}),
          ("POLY", "ts_native", "POLY", dict(periodicity=period)),
          ("KShapeAD", "ts_native", "KShapeAD", dict(periodicity=period)),
          ("IForest", "point", "IForest", {}), ("LOF", "point", "LOF", dict(n_neighbors=20)),
          ("KNN", "point", "KNN", dict(n_neighbors=10)), ("HBOS", "point", "HBOS", {}),
          ("PCA", "point", "PCA", {}), ("COPOD", "point", "COPOD", {}),
          ("EIF", "point", "EIF", {})]
    return p


# ---------------- Wu & Keogh one-liner triviality ----------------
def _movmean(a, k):
    c = np.cumsum(np.insert(a, 0, 0.0)); out = np.empty(len(a)); h = k // 2
    for i in range(len(a)):
        lo, hi = max(0, i - h), min(len(a), i + h + 1)
        out[i] = (c[hi] - c[lo]) / (hi - lo)
    return out


def _movstd(a, k):
    m = _movmean(a, k); m2 = _movmean(a * a, k)
    return np.sqrt(np.maximum(m2 - m * m, 0.0))


def keogh_trivial(x, lab):
    anom = np.where(lab == 1)[0]
    if len(anom) == 0 or len(x) < 50:
        return False, "no_anomaly"
    d = np.diff(x, prepend=x[0])
    for base, tag in ((np.abs(d), "absdiff"), (d, "diff")):
        for k in (0, 8, 32, 128, 512):
            for u in (0, 1):
                if u and k == 0:
                    continue
                for c in (0.0, 1.0, 3.0):
                    if k == 0 and c > 0:
                        continue
                    thr = np.zeros(len(base))
                    if u:
                        thr = thr + _movmean(base, k)
                    if c and k:
                        thr = thr + c * _movstd(base, k)
                    st = base - thr
                    if np.all(np.isfinite(st)) and lab[int(np.nanargmax(st))] == 1:
                        return True, f"{tag}_u{u}_k{k}_c{c:g}"
    return False, "survived"


z = zipfile.ZipFile(ZIP)
files = sorted(n for n in z.namelist() if n.lower().endswith(".csv"))
files = sorted(files, key=hashlib.sha1(b"").hexdigest().__class__ and
               (lambda s: hashlib.sha1(s.encode()).hexdigest()))[:N_SERIES]
print(f"series={len(files)} (hash-sampled)", flush=True)

rows, meta, t0 = [], [], time.time()
for i, fn in enumerate(files, 1):
    base = os.path.basename(fn)
    src = base.split("_")[1] if "_" in base else "?"
    rec = dict(dataset=base, corpus="tsbad", source=src)
    try:
        df = pd.read_csv(io.BytesIO(z.read(fn)))
        x = np.nan_to_num(df.iloc[:, 0].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)
        lab = df.iloc[:, 1].to_numpy(int)
    except Exception:
        rec.update(status="dropped", reason="parse"); meta.append(rec); continue
    if len(x) > MAX_LEN:
        x, lab = x[:MAX_LEN], lab[:MAX_LEN]

    m = re.search(r"_tr_(\d+)_", base)
    tr_end = min(int(m.group(1)), len(x) // 2) if m else len(x) // 5
    rec.update(n=len(x), n_anom=int(lab.sum()), tr_end=tr_end)

    triv, how = keogh_trivial(x, lab)
    rec.update(keogh_trivial=bool(triv), keogh_rule=how)
    if triv:
        rec.update(status="dropped", reason="keogh_trivial"); meta.append(rec)
    else:
        te = np.arange(tr_end, len(x))
        yte = lab[te]
        if yte.sum() < MIN_ANOM or yte.sum() == len(yte):
            rec.update(status="dropped", reason="too_few_test_anomalies"); meta.append(rec)
        else:
            period = 50 if len(x) > 2000 else 20
            n_ok = 0
            for vname, fam, model, kw in POOL(period):
                try:
                    s = run_Unsupervise_AD(model, x.reshape(-1, 1), **kw)
                except Exception:
                    continue
                if not isinstance(s, np.ndarray):     # TSB-AD returns a STRING on failure
                    continue
                s = np.asarray(s, float).ravel()
                if len(s) != len(x) or not np.all(np.isfinite(s)) or np.nanstd(s) < 1e-12:
                    continue
                st = s[te]
                rows.append(dict(dataset=base, corpus="tsbad", source=src, variant=vname,
                                 family=fam,
                                 auc=float(roc_auc_score(yte, st)),
                                 pauc10=float(roc_auc_score(yte, st, max_fpr=0.10)),
                                 ap=float(average_precision_score(yte, st)),
                                 base_rate=float(yte.mean())))
                n_ok += 1
            rec.update(status="scored" if n_ok >= 6 else "dropped",
                       n_variants=n_ok,
                       reason=None if n_ok >= 6 else f"only_{n_ok}_variants")
            meta.append(rec)
    if i % 5 == 0 or i == len(files):
        pd.DataFrame(rows).to_csv(OUT, index=False)
        pd.DataFrame(meta).to_csv(OUT.replace(".csv", "_steps.csv"), index=False)
        sc = sum(1 for r in meta if r.get("status") == "scored")
        tv = sum(1 for r in meta if r.get("keogh_trivial"))
        print(f"[{i}/{len(files)}] scored={sc} keogh_trivial={tv} | {time.time()-t0:.0f}s",
              flush=True)

D = pd.DataFrame(rows); D.to_csv(OUT, index=False)
M0 = pd.DataFrame(meta); M0.to_csv(OUT.replace(".csv", "_steps.csv"), index=False)
print(f"\nwrote {OUT} rows={len(D)} series={D.dataset.nunique() if len(D) else 0}")

if len(M0):
    print(f"\n=== S1: Wu & Keogh one-liner triviality ===")
    tv = M0.keogh_trivial.fillna(False)
    print(f"  TRIVIAL: {int(tv.sum())}/{len(M0)} series ({100*tv.mean():.0f}%)")
    if tv.any():
        print(M0[tv].keogh_rule.value_counts().head(5).to_string())
    print(f"  drops: {M0[M0.status=='dropped'].reason.value_counts().to_dict()}")
if len(D):
    print(f"\n=== pool ===  " + str(D.groupby('family').variant.nunique().to_dict()))
    wins = [g.loc[g.auc.idxmax(), "family"] for _, g in D.groupby("dataset")
            if g.family.nunique() > 1]
    if wins:
        print("  which family WINS per series:",
              pd.Series(wins).value_counts(normalize=True).round(3).to_dict())
    for metric in ["auc", "ap"]:
        M = D.groupby("dataset")[metric].agg(["max", "mean"])
        M["spread"] = M["max"] - M["mean"]
        zone = lambda b, s: "ceiling" if (b >= 0.95 and s < 0.03) else ("floor" if b < 0.70 else "live")
        M["zone"] = [zone(b, s) for b, s in zip(M["max"], M.spread)]
        print(f"\n=== inclusion on {metric.upper()} === zones {M.zone.value_counts().to_dict()}")
        for thr in [0.05, 0.10, 0.15]:
            print(f"  live AND spread>={thr:.2f}: "
                  f"{len(M[(M.zone=='live')&(M.spread>=thr)]):3d}/{len(M)}")
        if metric == "auc":
            M.reset_index().to_csv(OUT.replace(".csv", "_manifest.csv"), index=False)
