# -*- coding: utf-8 -*-
"""HADB time-series arm, done properly: Wu & Keogh triviality + TS-NATIVE detector families.

TWO CHANGES over the first attempt:

1. TRIVIALITY USES WU & KEOGH'S OWN TEST, not max|z| on window statistics.
   Wu & Keogh (IEEE TKDE 35(3):2421-2429, 2023; arXiv:2009.13807) Definition 1:
     "A time series anomaly detection problem is trivial if it can be solved with a single
      line of standard library MATLAB code ... limited to basic vectorized primitive
      operations, such as mean, max, std, diff."
   Their universal one-liners, which we brute-force over (u, k, c, b):
     abs(diff(TS)) > u*movmean(abs(diff(TS)),k) + c*movstd(abs(diff(TS)),k) + b
         diff(TS)  > u*movmean(diff(TS),k)      + c*movstd(diff(TS),k)      + b
   A SERIES is dropped if any parameterisation flags the labelled anomaly region.
   This is dataset-level (their construction) and is the domain-appropriate criterion:
   86.1% of Yahoo series fall to it. It replaces max|z|, which was designed for tabular
   features and means little on window summary statistics.

2. THE CANDIDATE POOL IS TS-NATIVE.
   A selection method ranks WHATEVER candidates it is given, so per-modality pools are
   legitimate and different datasets may carry different candidates. For time series the
   right families are subsequence/matrix-profile methods, not tabular detectors:
     aeon (BSD-3): STOMP (matrix profile), LeftSTAMPi, MERLIN, KMeansAD, STRAY
     plus windowed-PyOD variants as a second family GROUP for contrast.
   window_size is swept, which is the hyperparameter axis that matters most for TS.
   (stumpy and dtaianomaly have no Python 3.14 wheels on this machine.)
"""
import os, sys, io, zipfile, hashlib, time, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"
sys.path.insert(0, os.path.join(ROOT, "src"))
from adrank.ts import _window_features, _window_labels  # noqa: E402

ZIP = os.path.join(ROOT, "data", "tsbad", "TSB-AD-U.zip")
N_SERIES = int(sys.argv[1]) if len(sys.argv) > 1 else 150
SEEDS = [0, 1, 2]
W, STRIDE, MAX_LEN = 64, 16, 40000
MIN_WIN, MIN_NORM_WIN, MIN_HARD = 200, 150, 20
OUT = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d\hadb_ts_v2.csv"


# ---------------- Wu & Keogh one-liner triviality (series level) ----------------
def _movmean(a, k):
    if k <= 1:
        return a.copy()
    c = np.cumsum(np.insert(a, 0, 0.0))
    out = np.full_like(a, np.nan, dtype=float)
    half = k // 2
    for i in range(len(a)):
        lo, hi = max(0, i - half), min(len(a), i + half + 1)
        out[i] = (c[hi] - c[lo]) / (hi - lo)
    return out


def _movstd(a, k):
    if k <= 1:
        return np.zeros_like(a, dtype=float)
    m = _movmean(a, k)
    m2 = _movmean(a * a, k)
    return np.sqrt(np.maximum(m2 - m * m, 0.0))


def keogh_trivial(x, lab):
    """True if ANY one-liner parameterisation isolates the anomaly region.
    Score = the one-liner statistic; 'solved' if its argmax falls inside a labelled
    anomaly (their success criterion is finding the anomaly, not a full ROC)."""
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
                    thr = np.zeros_like(base, dtype=float)
                    if u:
                        thr = thr + _movmean(base, k)
                    if c and k:
                        thr = thr + c * _movstd(base, k)
                    stat = base - thr
                    if not np.all(np.isfinite(stat)):
                        continue
                    if lab[int(np.nanargmax(stat))] == 1:      # one line finds it
                        return True, f"{tag}_u{u}_k{k}_c{c:g}"
    return False, "survived"


# ---------------- TS-NATIVE pool (aeon) + windowed-PyOD contrast group ----------------
def build_pools():
    ts_native, windowed = [], []
    try:
        from aeon.anomaly_detection.series.distance_based import STOMP, KMeansAD, MERLIN, LeftSTAMPi
        for w in (32, 64, 128):
            ts_native.append((f"STOMP_w{w}", "ts_native",
                              lambda w=w: STOMP(window_size=w)))
            ts_native.append((f"KMeansAD_w{w}", "ts_native",
                              lambda w=w: KMeansAD(window_size=w, n_clusters=8)))
        for w in (64, 128):
            ts_native.append((f"LeftSTAMPi_w{w}", "ts_native",
                              lambda w=w: LeftSTAMPi(window_size=w)))
        ts_native.append(("MERLIN", "ts_native", lambda: MERLIN(min_length=32, max_length=128)))
    except Exception as e:
        print(f"  aeon TS-native unavailable: {e}", flush=True)
    from pyod.models.lof import LOF
    from pyod.models.knn import KNN
    from pyod.models.iforest import IForest
    from pyod.models.hbos import HBOS
    from pyod.models.ecod import ECOD
    from pyod.models.copod import COPOD
    from pyod.models.pca import PCA as PPCA
    windowed = [("w_IForest", "windowed", lambda: IForest(n_estimators=100, random_state=0)),
                ("w_ECOD", "windowed", lambda: ECOD()), ("w_COPOD", "windowed", lambda: COPOD()),
                ("w_HBOS", "windowed", lambda: HBOS(n_bins=10)),
                ("w_PCA", "windowed", lambda: PPCA(n_components=0.5, random_state=0))]
    for k in (5, 20, 50, 100):
        windowed.append((f"w_LOF_k{k}", "windowed", lambda k=k: LOF(n_neighbors=k)))
        windowed.append((f"w_KNN_k{k}", "windowed", lambda k=k: KNN(n_neighbors=k)))
    return ts_native, windowed


TS_NATIVE, WINDOWED = build_pools()
print(f"pools: {len(TS_NATIVE)} TS-native (aeon) + {len(WINDOWED)} windowed-PyOD", flush=True)


def score_windowed(ctor, Xtr, Xte):
    import contextlib
    try:
        m = ctor()
        with contextlib.redirect_stdout(io.StringIO()):
            m.fit(Xtr)
        s = np.asarray(m.decision_function(Xte), float)
        return s if np.all(np.isfinite(s)) else None
    except Exception:
        return None


def score_ts_native(ctor, x_raw, starts, w):
    """Run a raw-series detector, then aggregate its point scores to WINDOW scores so it is
    comparable with the windowed-PyOD group on the identical evaluation set."""
    import contextlib
    try:
        m = ctor()
        with contextlib.redirect_stdout(io.StringIO()):
            p = np.asarray(m.fit_predict(x_raw), float)
        if not np.all(np.isfinite(p)) or np.nanstd(p) < 1e-12:
            return None
        return np.array([np.nanmax(p[s:s + w]) if s + w <= len(p) else np.nan
                         for s in starts], float)
    except Exception:
        return None


z = zipfile.ZipFile(ZIP)
files = sorted(n for n in z.namelist() if n.lower().endswith(".csv"))
files = sorted(files, key=lambda s: hashlib.sha1(s.encode()).hexdigest())[:N_SERIES]

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

    # ---- S1: Wu & Keogh triviality, at the SERIES level ----
    triv, how = keogh_trivial(x, lab)
    rec.update(keogh_trivial=bool(triv), keogh_rule=how)
    if triv:
        rec.update(status="dropped", reason="keogh_trivial"); meta.append(rec)
        if i % 10 == 0:
            print(f"[{i}/{len(files)}] trivial so far="
                  f"{sum(1 for m in meta if m.get('keogh_trivial'))} | {time.time()-t0:.0f}s",
                  flush=True)
        continue

    try:
        Xw, starts = _window_features(x, w=W, stride=STRIDE)
        yw = _window_labels(lab, starts, w=W, min_count=1)
    except Exception:
        rec.update(status="dropped", reason="window"); meta.append(rec); continue
    Xw = np.nan_to_num(Xw, nan=0.0, posinf=0.0, neginf=0.0)
    _, uq = np.unique(Xw, axis=0, return_index=True); uq = np.sort(uq)
    keep_mask = np.zeros(len(Xw), bool); keep_mask[uq] = True
    Xw, yw, starts = Xw[keep_mask], yw[keep_mask], np.asarray(starts)[keep_mask]
    rec.update(n_win=len(Xw), n_anom_win=int(yw.sum()), n_norm_win=int((yw == 0).sum()))
    if len(Xw) < MIN_WIN or yw.sum() < MIN_HARD or (yw == 0).sum() < MIN_NORM_WIN:
        rec.update(status="dropped", reason="too_short_or_dense"); meta.append(rec); continue

    norm_idx = np.where(yw == 0)[0]
    anom_idx = np.where(yw == 1)[0]
    for seed in SEEDS:
        g = np.random.default_rng(seed)
        idx = np.arange(len(norm_idx)); g.shuffle(idx)
        cut = int(0.8 * len(idx))
        tr_i, te_i = norm_idx[idx[:cut]], norm_idx[idx[cut:]]
        if len(tr_i) < 50:
            continue
        eval_i = np.r_[te_i, anom_idx]
        yte = np.r_[np.zeros(len(te_i)), np.ones(len(anom_idx))]
        for name, fam, ctor in WINDOWED:
            s = score_windowed(ctor, Xw[tr_i], Xw[eval_i])
            if s is None or np.nanstd(s) < 1e-12:
                continue
            rows.append(dict(dataset=base, corpus="tsbad", source=src, seed=seed,
                             variant=name, family=fam,
                             auc=float(roc_auc_score(yte, s)),
                             pauc10=float(roc_auc_score(yte, s, max_fpr=0.10)),
                             ap=float(average_precision_score(yte, s)),
                             base_rate=float(yte.mean())))
        if seed == SEEDS[0]:      # TS-native detectors are unsupervised on the raw series
            for name, fam, ctor in TS_NATIVE:
                sw = score_ts_native(ctor, x, starts, W)
                if sw is None:
                    continue
                s = sw[eval_i]
                if not np.all(np.isfinite(s)) or np.nanstd(s) < 1e-12:
                    continue
                rows.append(dict(dataset=base, corpus="tsbad", source=src, seed=seed,
                                 variant=name, family=fam,
                                 auc=float(roc_auc_score(yte, s)),
                                 pauc10=float(roc_auc_score(yte, s, max_fpr=0.10)),
                                 ap=float(average_precision_score(yte, s)),
                                 base_rate=float(yte.mean())))
    rec.update(status="scored"); meta.append(rec)
    if i % 10 == 0 or i == len(files):
        pd.DataFrame(rows).to_csv(OUT, index=False)
        pd.DataFrame(meta).to_csv(OUT.replace(".csv", "_steps.csv"), index=False)
        print(f"[{i}/{len(files)}] scored={sum(1 for m in meta if m.get('status')=='scored')} "
              f"trivial={sum(1 for m in meta if m.get('keogh_trivial'))} | {time.time()-t0:.0f}s",
              flush=True)

D = pd.DataFrame(rows); D.to_csv(OUT, index=False)
M0 = pd.DataFrame(meta); M0.to_csv(OUT.replace(".csv", "_steps.csv"), index=False)
print(f"\nwrote {OUT} rows={len(D)} series={D.dataset.nunique() if len(D) else 0}")

if len(M0):
    print(f"\n=== S1: Wu & Keogh one-liner triviality (series level) ===")
    print(f"  TRIVIAL: {int(M0.keogh_trivial.fillna(False).sum())}/{len(M0)} series")
    tv = M0[M0.keogh_trivial.fillna(False)]
    if len(tv):
        print("  which one-liner solved them:")
        print(tv.keogh_rule.value_counts().head(6).to_string())
    print(f"\n  other drops: {M0[M0.status=='dropped'].reason.value_counts().to_dict()}")
if len(D):
    print(f"\n=== pool coverage ===")
    print(D.groupby('family').variant.nunique().to_string())
    print(f"\n=== does the TS-NATIVE family ever win? ===")
    wins = []
    for (ds, sd), g in D.groupby(["dataset", "seed"]):
        if g.family.nunique() > 1:
            wins.append(g.loc[g.auc.idxmax(), "family"])
    if wins:
        print(pd.Series(wins).value_counts(normalize=True).round(3).to_string())
    for metric in ["auc", "ap"]:
        gg = D.groupby(["dataset", "seed"])[metric].agg(["max", "mean"])
        M = gg.groupby("dataset").mean(); M["spread"] = M["max"] - M["mean"]
        zone = lambda b, s: "ceiling" if (b >= 0.95 and s < 0.03) else ("floor" if b < 0.70 else "live")
        M["zone"] = [zone(b, s) for b, s in zip(M["max"], M.spread)]
        print(f"\n=== inclusion on {metric.upper()} === zones {M.zone.value_counts().to_dict()}")
        for thr in [0.05, 0.10, 0.15]:
            print(f"  live AND spread>={thr:.2f}: "
                  f"{len(M[(M.zone=='live')&(M.spread>=thr)]):3d}/{len(M)}")
        if metric == "auc":
            M.reset_index().to_csv(OUT.replace(".csv", "_manifest.csv"), index=False)
