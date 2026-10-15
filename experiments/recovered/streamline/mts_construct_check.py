# -*- coding: utf-8 -*-
"""Does the PURITY relabeling cause collateral drops in benchmark PREPARATION? Mirror pipeline_final's
construction stages for MTS (Stage1 frac_triv<0.90, Stage5 n_norm>=800, Stage3 n_eff>=100 AFTER
hardening) under purity>=T labeling, and compare survival + per-dataset hard-anomaly counts vs current
(>=1pt) labeling. Concern: purer windows are more separable -> hardening may catch more -> frac_triv up
or n_eff down could drop HEALTHY datasets, not just CreditCard/GECCO."""
import os, sys, warnings
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"; S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
D = os.path.join(S, "scratchpad", "streamline"); sys.path.insert(0, os.path.join(S, "scratchpad"))
from hadb_ts_final import W, STRIDE, block_split3
from hadb_ts_mts import mts_window_features, window_labels as mts_wlabels, load_mts
Q = 0.05; TRIV1 = 0.90; MIN_HARD = 100; MIN_NORM = 800
PURITY = float(os.environ.get("PURITY", "0.5"))
def severity(Xtr, X, bins=30):
    n, d = X.shape; sev = np.zeros(n)
    for j in range(d):
        tr = np.sort(Xtr[:, j])
        if tr[-1] - tr[0] < 1e-12: continue
        m = len(tr); fb = np.searchsorted(tr, X[:, j], side="right") / m
        cnt, edges = np.histogram(Xtr[:, j], bins=bins); dens = cnt / max(cnt.sum(), 1); b = np.clip(np.digitize(X[:, j], edges[1:-1]), 0, bins - 1)
        sev = np.maximum(sev, np.maximum(-np.log(np.maximum(2 * np.minimum(fb, 1 - fb), 1.0 / (2 * m))), -np.log(dens[b] + 1e-9)))
    return sev
def two_stage_trivial(Xtr, Xhold, Xq):
    so_h, so_q = severity(Xtr, Xhold), severity(Xtr, Xq); tho = np.quantile(so_h, 1 - Q)
    try:
        sc = StandardScaler().fit(Xtr); pca = PCA(n_components=0.95, whiten=True, random_state=0).fit(sc.transform(Xtr))
        Ph, Pq, Ptr = pca.transform(sc.transform(Xhold)), pca.transform(sc.transform(Xq)), pca.transform(sc.transform(Xtr))
        sp_h, sp_q = severity(Ptr, Ph), severity(Ptr, Pq); thp = np.quantile(sp_h, 1 - Q)
    except Exception:
        sp_q = np.zeros(len(Xq)); thp = np.inf
    return (so_q > tho) | (sp_q > thp)
def n_eff(Xtr, Xh):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9; Zt = (Xtr - mu) / sd; Zh = (Xh - mu) / sd
    rng = np.random.default_rng(0); sub = Zt[rng.choice(len(Zt), min(len(Zt), 1500), replace=False)]
    r = np.median(NearestNeighbors(n_neighbors=2).fit(sub).kneighbors(sub)[0][:, 1]) + 1e-9
    if len(Zh) > 600: Zh = Zh[rng.choice(len(Zh), 600, replace=False)]
    nn = NearestNeighbors(radius=r).fit(Zh); cov = np.zeros(len(Zh), bool); c = 0
    for i in range(len(Zh)):
        if cov[i]: continue
        c += 1; cov[nn.radius_neighbors(Zh[i:i + 1], return_distance=False)[0]] = True
    return c
def split_lbl(Xc, lab, purity):
    Xw, st = mts_window_features(Xc); Xw = np.nan_to_num(Xw); allpos = np.arange(len(Xw))
    if purity is None:
        yw = mts_wlabels(lab, st); pos = allpos
        Xw2 = Xw
    else:
        pur = np.array([lab[s:s + W].sum() / W for s in st]); keep = (pur == 0) | (pur >= purity)
        Xw2, yw, pos = Xw[keep], (pur[keep] >= purity).astype(int), allpos[keep]
    tr, va, te = block_split3(yw, pos, 0)
    return Xw2[tr][yw[tr] == 0], Xw2[te][yw[te] == 0], Xw2[yw == 1]
rk = pd.read_csv(os.path.join(D, "STREAM_RANK.csv")); inbench = list(rk[rk.corpus == "tsbad_m"].dataset)
MTS = {}
for name, src, Xc, lab in load_mts(200):
    if str(name)[:40] in set(inbench): MTS[str(name)[:40]] = (Xc, np.asarray(lab, int).ravel())
def run(purity):
    out = {}
    for nm in inbench:
        if nm not in MTS: continue
        Xc, lab = MTS[nm]
        try:
            Xtr, Xhold, Xa = split_lbl(Xc, lab, purity)
        except Exception: continue
        if len(Xtr) < 40 or len(Xhold) < 20 or len(Xa) < 5:
            out[nm] = dict(n_norm=len(Xhold), n_anom=len(Xa), n_hard=0, frac_triv=1.0, n_eff=0, ok=False); continue
        triv = two_stage_trivial(Xtr, Xhold, Xa); hard = Xa[~triv]; ne = n_eff(Xtr, hard) if len(hard) >= 5 else len(hard)
        ok = (triv.mean() < TRIV1) and (len(Xhold) >= MIN_NORM) and (ne >= MIN_HARD)
        out[nm] = dict(n_norm=len(Xhold), n_anom=len(Xa), n_hard=int((~triv).sum()), frac_triv=round(float(triv.mean()), 3), n_eff=int(ne), ok=ok)
    return out
cur, pur = run(None), run(PURITY)
rows = []
for nm in inbench:
    if nm in cur and nm in pur:
        rows.append({"dataset": nm[:38], "cur_neff": cur[nm]["n_eff"], "pur_neff": pur[nm]["n_eff"],
                     "cur_ok": cur[nm]["ok"], "pur_ok": pur[nm]["ok"], "pur_ftriv": pur[nm]["frac_triv"], "pur_nanom": pur[nm]["n_anom"]})
df = pd.DataFrame(rows); df.to_csv(os.path.join(D, "STREAM_MTS_CONSTRUCT_CHECK.csv"), index=False)
print(f"=== MTS construction survival under purity>={PURITY} vs current (>=1pt), MIN_HARD n_eff>={MIN_HARD} ===")
print(f"  datasets checked: {len(df)}")
print(f"  survive current: {int(df.cur_ok.sum())}   survive purity: {int(df.pur_ok.sum())}")
newdrop = df[df.cur_ok & ~df.pur_ok]
print(f"  NEW drops caused by relabeling: {len(newdrop)} -> {list(newdrop.dataset)}")
print(f"  mean n_eff (surviving both): current {df[df.pur_ok].cur_neff.mean():.0f}  purity {df[df.pur_ok].pur_neff.mean():.0f}")
print(f"  datasets where purity REDUCES n_eff below current by >30%: {int(((df.pur_ok)&(df.pur_neff < 0.7*df.cur_neff)).sum())}")
print("\n  --- datasets with largest n_eff drop (surviving) ---")
d2 = df[df.pur_ok].copy(); d2["drop"] = d2.cur_neff - d2.pur_neff
print(d2.sort_values("drop", ascending=False).head(8)[["dataset", "cur_neff", "pur_neff", "pur_ftriv", "pur_nanom"]].to_string(index=False))
print("\nsaved streamline/STREAM_MTS_CONSTRUCT_CHECK.csv")
