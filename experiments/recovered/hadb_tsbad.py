# -*- coding: utf-8 -*-
"""Fold TIME SERIES (TSB-AD) into HADB using the IDENTICAL filter chain.

WHY TIME SERIES FITS HADB CLEANLY (and image/text do not):
windowing turns a series into a tabular matrix of 28 INTERPRETABLE summary statistics
(mean, std, min, max, slope, ...). Those axes carry standalone meaning, so the max|z|
triviality criterion means what it is supposed to mean. On deep embeddings (ResNet/BERT)
a per-axis z-score corresponds to nothing a practitioner would call a trivial anomaly,
which is why CV/NLP need a solution-based criterion instead.

Source: TSB-AD-U (Liu & Paparrizos, NeurIPS 2024 D&B, Apache-2.0), 870 univariate series
re-curated from 40 datasets specifically to remove flawed exemplars.

Filters identical to build_hadb_v3.py: dedup -> double-hard triviality (max|z| AND LinRes
at the 99th pct of train-normal) -> min hard anomalies -> 41-variant pool on the paper's
target protocol -> spread-based inclusion. Stability recorded, never used to filter.

KNOWN LIMITATION, stated up front: the 28-statistic window descriptor is the paper's own
acknowledged weak point ("a richer window descriptor is needed than a handful of summary
statistics"). A weak descriptor pushes every detector toward the floor, so it will SHRINK
the time-series contribution to HADB rather than inflate it. Series excluded for
floor/low-spread should not be read as "no detector can do this" but as "not with this
representation".
"""
import os, sys, io, zipfile, hashlib, time, warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"
sys.path.insert(0, os.path.join(ROOT, "src"))
from adrank.ts import _window_features, _window_labels  # noqa: E402

from pyod.models.lof import LOF
from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from pyod.models.hbos import HBOS
from pyod.models.ecod import ECOD
from pyod.models.copod import COPOD
from pyod.models.pca import PCA as PPCA
from pyod.models.cblof import CBLOF
from pyod.models.loda import LODA

ZIP = os.path.join(ROOT, "data", "tsbad", "TSB-AD-U.zip")
TRIV_PCT, MIN_HARD = 99.0, 20
SEEDS = [0, 1, 2]
W, STRIDE, MAX_LEN = 64, 16, 60000
MIN_WIN, MIN_NORM_WIN = 200, 150
N_SERIES = int(sys.argv[1]) if len(sys.argv) > 1 else 300
OUT = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d\hadb_tsbad.csv"

CLASSIC = [("IForest", lambda: IForest(n_estimators=100, random_state=0)),
           ("LOF", lambda: LOF(n_neighbors=20)), ("KNN", lambda: KNN(n_neighbors=5)),
           ("ECOD", lambda: ECOD()), ("COPOD", lambda: COPOD()),
           ("HBOS", lambda: HBOS(n_bins=10)),
           ("PCA", lambda: PPCA(n_components=0.5, random_state=0)),
           ("CBLOF", lambda: CBLOF(n_clusters=8, random_state=0)),
           ("LODA", lambda: LODA(n_bins=10))]

def build_pool():
    p = list(CLASSIC)
    for k in [3, 10, 35, 50, 100, 200]:
        p.append((f"LOF_k{k}", lambda k=k: LOF(n_neighbors=k)))
        p.append((f"KNN_k{k}", lambda k=k: KNN(n_neighbors=k)))
    for nb in [5, 20, 50, 100]:
        p.append((f"HBOS_b{nb}", lambda nb=nb: HBOS(n_bins=nb)))
    for nc in [0.3, 0.7, 0.9, 0.99]:
        p.append((f"PCA_c{nc}", lambda nc=nc: PPCA(n_components=nc, random_state=0)))
    for ncl in [4, 16, 32]:
        p.append((f"CBLOF_c{ncl}", lambda ncl=ncl: CBLOF(n_clusters=ncl, random_state=0)))
    for ne in [50, 300]:
        p.append((f"IF_n{ne}", lambda ne=ne: IForest(n_estimators=ne, random_state=0)))
    for nb in [20, 50]:
        p.append((f"LODA_b{nb}", lambda nb=nb: LODA(n_bins=nb)))
    return p

POOL = build_pool()
CLASSIC_NAMES = {n for n, _ in CLASSIC}


def score(ctor, Xtr, Xte):
    import contextlib
    try:
        m = ctor()
        with contextlib.redirect_stdout(io.StringIO()):
            m.fit(Xtr)
        s = np.asarray(m.decision_function(Xte), float)
        return s if np.all(np.isfinite(s)) else None
    except Exception:
        return None


def linres(X, nm):
    d = X.shape[1]
    if d < 2 or d > 120:
        return None
    Xn = X[nm]; R = np.zeros((len(X), d))
    for k in range(d):
        oth = [i for i in range(d) if i != k]
        try:
            m = Ridge(alpha=1.0).fit(Xn[:, oth], Xn[:, k])
            res = X[:, k] - m.predict(X[:, oth])
            R[:, k] = np.abs(res) / (res[nm].std() + 1e-9)
        except Exception:
            R[:, k] = 0.0
    return R.max(1)


z = zipfile.ZipFile(ZIP)
files = sorted(n for n in z.namelist() if n.lower().endswith(".csv"))
files = sorted(files, key=lambda s: hashlib.sha1(s.encode()).hexdigest())[:N_SERIES]
print(f"TSB-AD series sampled (hash, not alphabetical): {len(files)}; pool={len(POOL)}",
      flush=True)

rows, meta, t0 = [], [], time.time()
for i, fn in enumerate(files, 1):
    base = os.path.basename(fn)
    src = base.split("_")[1] if "_" in base else "?"
    rec = dict(dataset=base, corpus="tsbad", source=src)
    try:
        df = pd.read_csv(io.BytesIO(z.read(fn)))
        x = np.nan_to_num(df.iloc[:, 0].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)
        lab = df.iloc[:, 1].to_numpy(int)
    except Exception as e:
        rec.update(status="dropped", reason=f"parse_{type(e).__name__}"); meta.append(rec); continue
    if len(x) > MAX_LEN:
        x, lab = x[:MAX_LEN], lab[:MAX_LEN]
    try:
        Xw, starts = _window_features(x, w=W, stride=STRIDE)
        yw = _window_labels(lab, starts, w=W, min_count=1)
    except Exception as e:
        rec.update(status="dropped", reason=f"window_{type(e).__name__}"); meta.append(rec); continue
    Xw = np.nan_to_num(Xw, nan=0.0, posinf=0.0, neginf=0.0)
    n_raw = len(Xw)
    _, uq = np.unique(Xw, axis=0, return_index=True); uq = np.sort(uq)
    Xw, yw = Xw[uq], yw[uq]
    rec.update(n_win=len(Xw), dup_frac=round(1 - len(Xw) / max(n_raw, 1), 4),
               n_anom_win=int(yw.sum()), n_norm_win=int((yw == 0).sum()),
               win_anom_rate=round(float(yw.mean()), 4))
    if len(Xw) < MIN_WIN or yw.sum() < 10 or (yw == 0).sum() < MIN_NORM_WIN:
        rec.update(status="dropped", reason="too_short_or_dense"); meta.append(rec); continue

    nm = yw == 0
    mu, sd = Xw[nm].mean(0), Xw[nm].std(0) + 1e-9
    maxz = np.abs((Xw - mu) / sd).max(1)
    lr = linres(Xw, nm)
    anom = np.where(yw == 1)[0]
    kz = maxz[anom] <= np.percentile(maxz[nm], TRIV_PCT)
    kl = (lr[anom] <= np.percentile(lr[nm], TRIV_PCT)) if lr is not None else True
    hard = anom[kz & kl]
    rec.update(n_hard=int(len(hard)), frac_trivial=round(1 - len(hard) / len(anom), 4),
               frac_trivial_maxz_only=round(1 - int(np.sum(kz)) / len(anom), 4))
    if len(hard) < MIN_HARD:
        rec.update(status="dropped", reason=f"only_{len(hard)}_double_hard")
        meta.append(rec); continue

    norm_idx = np.where(nm)[0]
    for seed in SEEDS:
        g = np.random.default_rng(seed)
        idx = np.arange(len(norm_idx)); g.shuffle(idx)
        cut = int(0.8 * len(idx))
        tr_i, te_i = norm_idx[idx[:cut]], norm_idx[idx[cut:]]
        if len(tr_i) < 50:
            continue
        Xtr = Xw[tr_i]
        Xte = np.vstack([Xw[te_i], Xw[hard]])
        yte = np.r_[np.zeros(len(te_i)), np.ones(len(hard))]
        h = len(tr_i) // 2
        A, B = Xw[tr_i[:h]], Xw[tr_i[h:]]
        for vname, ctor in POOL:
            s = score(ctor, Xtr, Xte)
            if s is None or np.nanstd(s) < 1e-12:
                continue
            sa, sb = score(ctor, A, Xte), score(ctor, B, Xte)
            st = np.nan
            if sa is not None and sb is not None and np.nanstd(sa) > 1e-12 and np.nanstd(sb) > 1e-12:
                st = spearmanr(sa, sb).statistic
            rows.append(dict(dataset=base, corpus="tsbad", source=src, seed=seed,
                             variant=vname, is_classic=vname in CLASSIC_NAMES,
                             auc=float(roc_auc_score(yte, s)),
                             pauc10=float(roc_auc_score(yte, s, max_fpr=0.10)),
                             ap=float(average_precision_score(yte, s)),
                             base_rate=float(yte.mean()),
                             stability=float(st) if np.isfinite(st) else np.nan))
    rec.update(status="scored"); meta.append(rec)
    if i % 10 == 0 or i == len(files):
        pd.DataFrame(rows).to_csv(OUT, index=False)
        pd.DataFrame(meta).to_csv(OUT.replace(".csv", "_steps.csv"), index=False)
        print(f"[{i}/{len(files)}] scored={sum(1 for m in meta if m.get('status')=='scored')} "
              f"| {time.time()-t0:.0f}s", flush=True)

D = pd.DataFrame(rows); D.to_csv(OUT, index=False)
M0 = pd.DataFrame(meta); M0.to_csv(OUT.replace(".csv", "_steps.csv"), index=False)
print(f"\nwrote {OUT} rows={len(D)} series={D.dataset.nunique() if len(D) else 0}")

if len(D):
    sc = M0[M0.status == "scored"]
    print(f"\n=== TSB-AD through the HADB filters ===")
    print(f"  scored {len(sc)}/{len(M0)}")
    print(M0[M0.status == "dropped"].reason.value_counts().head(6).to_string())
    print(f"  double-hard removes {sc.frac_trivial.mean():.3f} "
          f"(max|z| alone {sc.frac_trivial_maxz_only.mean():.3f})")
    print(f"  source mix of scored: {sc.source.value_counts().head(8).to_dict()}")
    for metric in ["auc", "ap"]:
        gg = D.groupby(["dataset", "seed"])[metric].agg(["max", "mean"])
        M = gg.groupby("dataset").mean(); M["spread"] = M["max"] - M["mean"]
        zone = lambda b, s: "ceiling" if (b >= 0.95 and s < 0.03) else ("floor" if b < 0.70 else "live")
        M["zone"] = [zone(b, s) for b, s in zip(M["max"], M.spread)]
        print(f"\n=== inclusion on {metric.upper()} ===  zones: {M.zone.value_counts().to_dict()}")
        for thr in [0.05, 0.10, 0.15]:
            k = M[(M.zone == "live") & (M.spread >= thr)]
            print(f"  live AND spread>={thr:.2f}: {len(k):3d}/{len(M)}")
        if metric == "auc":
            M.reset_index().to_csv(OUT.replace(".csv", "_manifest.csv"), index=False)
