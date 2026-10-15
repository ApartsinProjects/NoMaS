# -*- coding: utf-8 -*-
"""THE STATED HYPOTHESIS: can label-free GEOMETRY of the normal data predict
whether the true-best detector is global-family or local-family?

Features come only from X (never y, never the detector scores). Target is the
true-best detector's family, majority over 5 seeds. 59 dev datasets have raw
matrices locally.
"""
import glob, os, warnings
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"
LOCAL = {"LOF", "KNN", "CBLOF"}
GLOBAL = {"HBOS", "COPOD", "ECOD", "PCA"}
RNG = np.random.RandomState(0)


def fam(d):
    return "local" if d in LOCAL else ("global" if d in GLOBAL else "other")


def descriptors(X):
    """Label-free geometry of the (unlabeled) data matrix."""
    X = np.asarray(X, float)
    X = X[np.isfinite(X).all(1)]
    if len(X) > 5000:
        X = X[RNG.choice(len(X), 5000, replace=False)]
    Xs = StandardScaler().fit_transform(X)
    n, d = Xs.shape
    k = kurtosis(Xs, axis=0, fisher=True)
    sk = np.abs(skew(Xs, axis=0))
    f = {}
    f["kurt_mean"] = float(np.nanmean(k))
    f["kurt_max"] = float(np.nanmax(k))
    f["kurt_frac_heavy"] = float(np.mean(k > 3))
    f["skew_mean"] = float(np.nanmean(sk))
    p = PCA(n_components=min(d, 20, n - 1)).fit(Xs)
    ev = p.explained_variance_ratio_
    f["pc1"] = float(ev[0])
    f["part_ratio"] = float((ev.sum() ** 2) / (np.sum(ev ** 2) + 1e-12))
    # multivariate tail heaviness: 99th vs median distance to centroid
    dist = np.linalg.norm(Xs - Xs.mean(0), axis=1)
    f["tail_ratio"] = float(np.percentile(dist, 99) / (np.median(dist) + 1e-9))
    # clusterability: how much structure k-means finds
    kk = min(10, max(2, n // 50))
    km = MiniBatchKMeans(n_clusters=kk, n_init=3, random_state=0).fit(Xs)
    inertia_ratio = km.inertia_ / (np.sum(dist ** 2) + 1e-9)
    f["cluster_gain"] = float(1 - inertia_ratio)
    f["log_n"] = float(np.log10(n))
    f["log_d"] = float(np.log10(d))
    return f


t = pd.read_parquet(os.path.join(ROOT, "results/raw/modal_true_allseeds.parquet"))
files = {}
for sub in ["adbench", "dami", "cv", "nlp"]:
    for p in glob.glob(os.path.join(ROOT, "data", sub, "*.npz")):
        files[os.path.splitext(os.path.basename(p))[0]] = p

recs = []
for ds, g in t.groupby("dataset"):
    if ds not in files:
        continue
    # true best detector per seed -> family; majority across seeds
    fams = []
    for s, gs in g.groupby("seed"):
        gs = gs.dropna(subset=["true_auc"])
        if len(gs) < 3:
            continue
        fams.append(fam(gs.loc[gs.true_auc.idxmax(), "detector"]))
    if not fams:
        continue
    target = pd.Series(fams).mode()[0]
    try:
        z = np.load(files[ds])
        f = descriptors(z["X"])
    except Exception as e:
        print(f"  skip {ds}: {e}")
        continue
    f["dataset"] = ds
    f["fam"] = target
    f["modality"] = g.modality.iloc[0]
    recs.append(f)

D = pd.DataFrame(recs)
print(f"datasets with geometry + target: {len(D)}")
print("target distribution:", D.fam.value_counts().to_dict())

FEATS = ["kurt_mean", "kurt_max", "kurt_frac_heavy", "skew_mean", "pc1",
         "part_ratio", "tail_ratio", "cluster_gain", "log_n", "log_d"]
X = D[FEATS].replace([np.inf, -np.inf], np.nan).fillna(0).values
y = (D.fam == "global").astype(int).values
print(f"global-best base rate: {y.mean():.3f}")

# univariate screen: which descriptor separates the families at all?
print("\nunivariate AUC per descriptor (global-vs-not):")
for i, f in enumerate(FEATS):
    a = roc_auc_score(y, X[:, i])
    print(f"  {f:16s} AUC={a:.3f}  (|.5-AUC|={abs(a-.5):.3f})")

# leave-one-dataset-out multivariate
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.5))
p = np.zeros(len(X))
for tr, te in LeaveOneOut().split(X):
    clf.fit(X[tr], y[tr])
    p[te] = clf.predict_proba(X[te])[:, 1]
print(f"\nLOO multivariate logistic AUC = {roc_auc_score(y, p):.3f}")
print(f"KILL CRITERION was AUC < 0.65 -> "
      f"{'PASS, routing is predictable' if roc_auc_score(y,p) >= 0.65 else 'FAIL, geometry does not predict family'}")
D.to_csv(r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d\scratchpad\geom.csv", index=False)
