# -*- coding: utf-8 -*-
"""Can the true-best FAMILY be predicted label-free?

Features are strictly label-free: the NoMaS pseudo-score profile per detector,
dataset size/dim, and regime variance. The anomaly `rate` column is LABEL-DERIVED
and is deliberately excluded.

Evaluation is grouped by dataset (3 seeds per dataset would otherwise leak).
Reports router AUC and, more importantly, the END metric: regret when routing
by the PREDICTED family.
"""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

CSV = r"E:\Projects\Submitted\ADRank\results\oddbench.csv"
LOCAL = {"LOF", "KNN", "CBLOF"}
GLOBAL = {"HBOS", "COPOD", "ECOD", "PCA"}
DETS = ["IForest", "LOF", "KNN", "ECOD", "COPOD", "HBOS", "PCA", "CBLOF", "LODA"]


def fam(d):
    return "local" if d in LOCAL else ("global" if d in GLOBAL else "other")


df = pd.read_csv(CSV)
df = df[df.status == "ok"]

X, y3, meta = [], [], []
for _, r in df.iterrows():
    ta = json.loads(r.true_auc_json); sc = json.loads(r.score_json)
    dets = [d for d in DETS if d in ta and d in sc and not pd.isna(ta[d])]
    if len(dets) < 9:
        continue
    s = np.array([sc[d] for d in DETS], float)
    sz = (s - s.mean()) / (s.std() + 1e-9)          # shape only, scale-free
    gl = np.mean([sz[DETS.index(d)] for d in GLOBAL])
    lo = np.mean([sz[DETS.index(d)] for d in LOCAL])
    feats = list(sz) + [gl - lo, s.std() / (abs(s.mean()) + 1e-9),
                        np.log10(r.n), np.log10(max(r.d, 1)),
                        float(r.max_regime_var) if not pd.isna(r.max_regime_var) else 0.0]
    best_det = max(dets, key=lambda d: ta[d])
    X.append(feats); y3.append(fam(best_det))
    meta.append((r.dataset, ta, sc, max(ta[d] for d in dets), dets))

X = np.array(X); y3 = np.array(y3)
groups = np.array([m[0] for m in meta])
print(f"rows={len(X)}  datasets={len(set(groups))}  features={X.shape[1]}")
print("class balance:", pd.Series(y3).value_counts(normalize=True).round(3).to_dict())

y_glob = (y3 == "global").astype(int)

# ---- Is there ANY signal? binary global-vs-not, grouped CV ----
gkf = GroupKFold(n_splits=5)
for name, clf in [("logreg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))),
                  ("rf", RandomForestClassifier(n_estimators=400, min_samples_leaf=5, random_state=0))]:
    p = np.zeros(len(X))
    for tr, te in gkf.split(X, y_glob, groups):
        clf.fit(X[tr], y_glob[tr])
        p[te] = clf.predict_proba(X[te])[:, 1]
    print(f"{name:7s} global-vs-not grouped-CV AUC = {roc_auc_score(y_glob, p):.3f}")
    if name == "rf":
        p_rf = p

# ---- 3-class router -> END METRIC: regret under predicted routing ----
clf3 = RandomForestClassifier(n_estimators=400, min_samples_leaf=5, random_state=0)
pred3 = np.empty(len(X), dtype=object)
for tr, te in gkf.split(X, y3, groups):
    clf3.fit(X[tr], y3[tr])
    pred3[te] = clf3.predict(X[te])
print(f"\n3-class router accuracy (grouped CV): {(pred3 == y3).mean():.3f} "
      f"(majority baseline {pd.Series(y3).value_counts(normalize=True).max():.3f})")

reg_now, reg_pred, reg_oracle = [], [], []
for i, (ds, ta, sc, best_auc, dets) in enumerate(meta):
    pick_all = max(dets, key=lambda d: sc[d])
    reg_now.append(best_auc - ta[pick_all])
    inf = [d for d in dets if fam(d) == pred3[i]] or dets
    reg_pred.append(best_auc - ta[max(inf, key=lambda d: sc[d])])
    ino = [d for d in dets if fam(d) == y3[i]]
    reg_oracle.append(best_auc - ta[max(ino, key=lambda d: sc[d])])

reg_now = np.array(reg_now); reg_pred = np.array(reg_pred); reg_oracle = np.array(reg_oracle)
print(f"\n--- END METRIC (regret) ---")
print(f"current NoMaS        : {reg_now.mean():.4f}")
print(f"PREDICTED routing    : {reg_pred.mean():.4f}   ({100*(1-reg_pred.mean()/reg_now.mean()):+.1f}% vs current)")
print(f"oracle routing       : {reg_oracle.mean():.4f}")
from scipy.stats import wilcoxon
d = reg_now - reg_pred
if np.any(d != 0):
    print(f"paired Wilcoxon (now vs predicted): p = {wilcoxon(reg_now, reg_pred).pvalue:.2e}  "
          f"wins={int((d>0).sum())} losses={int((d<0).sum())} ties={int((d==0).sum())}")
