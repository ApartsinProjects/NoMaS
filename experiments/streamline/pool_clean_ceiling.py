# -*- coding: utf-8 -*-
"""ANALYSIS ONLY (changes nothing): are MTS datasets 'all-bad' only because ambiguous (near-normal)
windows contaminate the anomaly set? For each MTS dataset fit the pool ONCE and compare the detector
CEILING (best ap_norm) on the CURRENT anomaly set (>=1pt, ambiguous included) vs a CLEANED set
(purity>=T, ambiguous dropped). If cleaning lifts the ceiling above 0.10, the dataset was difficult
only due to ambiguous windows -> clean, don't drop. Also flags datasets with no clean anomaly left."""
import os, sys, io, contextlib, warnings
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score
warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"; S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
D = os.path.join(S, "scratchpad", "streamline"); sys.path.insert(0, os.path.join(S, "scratchpad")); sys.path.insert(0, os.path.join(ROOT, "src"))
from dev_common import TAB_POOL
from hadb_ts_final import W, STRIDE, block_split3
from hadb_ts_mts import mts_window_features, load_mts
Q = 0.05; T = float(os.environ.get("PURITY", "0.5"))
def severity(Xtr, X, bins=30):
    n, d = X.shape; sev = np.zeros(n)
    for j in range(d):
        tr = np.sort(Xtr[:, j])
        if tr[-1] - tr[0] < 1e-12: continue
        m = len(tr); fb = np.searchsorted(tr, X[:, j], side="right") / m
        cnt, edges = np.histogram(Xtr[:, j], bins=bins); dens = cnt / max(cnt.sum(), 1); b = np.clip(np.digitize(X[:, j], edges[1:-1]), 0, bins - 1)
        sev = np.maximum(sev, np.maximum(-np.log(np.maximum(2 * np.minimum(fb, 1 - fb), 1.0 / (2 * m))), -np.log(dens[b] + 1e-9)))
    return sev
def hardmask(Xtr, Xhold, Xa):
    tho = np.quantile(severity(Xtr, Xhold), 1 - Q); so_a = severity(Xtr, Xa)
    try:
        sc = StandardScaler().fit(Xtr); pca = PCA(n_components=0.95, whiten=True, random_state=0).fit(sc.transform(Xtr))
        Ptr, Ph, Pa = pca.transform(sc.transform(Xtr)), pca.transform(sc.transform(Xhold)), pca.transform(sc.transform(Xa))
        thp = np.quantile(severity(Ptr, Ph), 1 - Q); sp_a = severity(Ptr, Pa)
    except Exception:
        sp_a = np.zeros(len(Xa)); thp = np.inf
    return ~((so_a > tho) | (sp_a > thp))
def apn(y, s):
    bb = y.mean(); return (average_precision_score(y, s) - bb) / (1 - bb + 1e-12)
rk = pd.read_csv(os.path.join(D, "STREAM_RANK.csv")); inbench = set(rk[rk.corpus == "tsbad_m"].dataset)
sp = pd.read_csv(os.path.join(D, "STREAM_POOL_SPREAD.csv")); best_cur = dict(zip(sp[sp.corpus == "tsbad_m"].dataset, sp[sp.corpus == "tsbad_m"].best))
rows = []
for name, src, Xc, lab in load_mts(200):
    nm = str(name)[:40]
    if nm not in inbench: continue
    lab = np.asarray(lab, int).ravel(); Xw, st = mts_window_features(Xc); Xw = np.nan_to_num(Xw)
    pur = np.array([lab[s:s + W].sum() / W for s in st]); pos = np.arange(len(Xw))
    yw = (pur > 0).astype(int); tr, va, te = block_split3(yw, pos, 0)
    Xtr, Xtn = Xw[tr][yw[tr] == 0], Xw[te][yw[te] == 0]
    Aall = Xw[pur > 0]; Aclean = Xw[pur >= T]
    if len(Xtr) < 40 or len(Xtn) < 10 or len(Aall) < 5:
        rows.append({"dataset": nm[:34], "best_current": best_cur.get(nm, np.nan), "best_clean": np.nan, "n_clean_hard": 0, "note": "skip"}); continue
    hc = hardmask(Xtr, Xtn, Aall); Ahc = Aall[hc]
    if len(Aclean) >= 5:
        hcl = hardmask(Xtr, Xtn, Aclean); Ahl = Aclean[hcl]
    else:
        Ahl = Aclean
    # single pool fit; score test-normal + both hard sets
    bc, bl = [], []
    for vn, ct in TAB_POOL:
        try:
            m = ct()
            with contextlib.redirect_stdout(io.StringIO()): m.fit(Xtr)
            sn = np.asarray(m.decision_function(Xtn), float)
            if len(Ahc) >= 10:
                sa = np.asarray(m.decision_function(Ahc), float)
                if np.all(np.isfinite(np.r_[sn, sa])) and np.nanstd(np.r_[sn, sa]) > 1e-12: bc.append(apn(np.r_[np.zeros(len(sn)), np.ones(len(Ahc))], np.r_[sn, sa]))
            if len(Ahl) >= 10:
                sa2 = np.asarray(m.decision_function(Ahl), float)
                if np.all(np.isfinite(np.r_[sn, sa2])) and np.nanstd(np.r_[sn, sa2]) > 1e-12: bl.append(apn(np.r_[np.zeros(len(sn)), np.ones(len(Ahl))], np.r_[sn, sa2]))
        except Exception: pass
    rows.append({"dataset": nm[:34], "best_current": round(float(np.max(bc)), 3) if bc else np.nan,
                 "best_clean": round(float(np.max(bl)), 3) if bl else np.nan,
                 "n_clean_hard": len(Ahl), "frac_clean": round(len(Aclean) / max(len(Aall), 1), 2)})
df = pd.DataFrame(rows); df.to_csv(os.path.join(D, "STREAM_CLEAN_CEILING.csv"), index=False)
df["lift"] = df.best_clean - df.best_current
allbad = df[df.best_current <= 0.10]
print(f"=== does cleaning ambiguous windows (purity>={T}) lift the detector ceiling? (MTS, {len(df)} datasets) ===\n")
print("  --- MTS datasets that are ALL-BAD under current labeling (best_current<=0.10) ---")
print(allbad.sort_values("best_current")[["dataset", "best_current", "best_clean", "lift", "n_clean_hard", "frac_clean"]].to_string(index=False))
cross = allbad[(allbad.best_clean > 0.10) & (allbad.n_clean_hard >= 20)]
gone = df[(df.n_clean_hard < 20)]
print(f"\n  all-bad MTS that CROSS 0.10 after cleaning (were hard only due to ambiguous windows): {len(cross)}/{len(allbad)}")
print(f"    -> {list(cross.dataset)}")
print(f"  datasets with <20 clean hard anomalies left (would DROP): {len(gone)} -> {list(gone.dataset)}")
print(f"\n  overall: mean ceiling current {df.best_current.mean():.3f} -> clean {df.best_clean.mean():.3f}  (mean lift {df.lift.mean():+.3f})")
print("saved streamline/STREAM_CLEAN_CEILING.csv")
