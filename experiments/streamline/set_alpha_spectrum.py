# -*- coding: utf-8 -*-
"""Use the correlation/spectrum-shape metric (corr_str etc., the strongest family predictors, AUC 0.73)
to SET alpha / route the selector. Merge the alpha-signatures (STREAM_ALPHAEST.jsonl) with the mined
spectrum stats (STREAM_MINE.csv). Test: does corr_str correlate with best-alpha? does alpha-from-corr_str
(unsup percentile map), corr_str binary route, or a LOO-learned map from spectrum feats beat binary
matched and approach the oracle?"""
import os, sys, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
D = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d\scratchpad\streamline"
ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]; TAU = -0.14
# alpha signatures
sigrows = {}
for ln in open(os.path.join(D, "STREAM_ALPHAEST.jsonl")):
    r = json.loads(ln); sigrows[r["key"]] = r
mine = pd.read_csv(os.path.join(D, "STREAM_MINE.csv")); mine["key"] = mine.corpus + "__" + mine.dataset
recs = []
for _, mr in mine.iterrows():
    r = sigrows.get(mr["key"])
    if r is None: continue
    names = r["names"]; ap = r["ap"]; sig = r["sig"]; best = max(ap.values())
    def pick(ai): return names[int(np.argmax([sig[v][ai] for v in names]))]
    oai = int(np.argmin([best - ap[pick(ai)] for ai in range(len(ALPHAS))]))
    lev = max(sig[v][-1] for v in names) - max(sig[v][0] for v in names)
    recs.append({"key": mr["key"], "truefam": mr["fam"], "best": best, "ap": ap, "sig": sig, "names": names,
                 "oai": oai, "lev": lev, "corr_str": mr["corr_str"], "top1_var": mr["top1_var"],
                 "eff_dim": mr["eff_dim"], "eigengap": mr["eigengap"], "silh": mr["silh"]})
n = len(recs); print(f"merged {n} datasets\n")
best_alpha = np.array([ALPHAS[r["oai"]] for r in recs])
for feat in ["corr_str", "top1_var", "eff_dim", "eigengap", "silh", "lev"]:
    v = np.array([r[feat] for r in recs]); rho, p = spearmanr(v, best_alpha)
    print(f"  Spearman({feat:9s}, best_alpha) = {rho:+.2f}  (p={p:.3f})")
def reg(rec, ai): return rec["best"] - rec["ap"][rec["names"][int(np.argmax([rec["sig"][v][ai] for v in rec["names"]]))]]
def macro(regs):
    fams = [r["truefam"] for r in recs]; return np.mean([np.mean([g for g, fm in zip(regs, fams) if fm == fa]) for fa in ["local", "global"] if any(fm == fa for fm in fams)])
# alpha-from-corr_str: rank corr_str -> grid alpha (high corr -> high alpha = local)
cs = np.array([r["corr_str"] for r in recs]); rank = cs.argsort().argsort() / max(n - 1, 1)
a_cs = [int(round(rk * (len(ALPHAS) - 1))) for rk in rank]
# corr_str binary route: above median -> local pick (alpha=1), else global (alpha=0)
med = np.nanmedian(cs); a_bin = [len(ALPHAS) - 1 if r["corr_str"] > med else 0 for r in recs]
# LOO learned map from spectrum feats -> oracle alpha index
Xf = np.array([[r["corr_str"], r["top1_var"], r["eff_dim"], r["eigengap"], r["silh"], r["lev"]] for r in recs])
Xf = np.nan_to_num((Xf - np.nanmean(Xf, 0)) / (np.nanstd(Xf, 0) + 1e-9)); yA = np.array([r["oai"] for r in recs])
loo = np.zeros(n, int)
for tr, te in LeaveOneOut().split(Xf):
    rf = RandomForestRegressor(n_estimators=200, random_state=0).fit(Xf[tr], yA[tr]); loo[te[0]] = int(np.clip(round(rf.predict(Xf[te])[0]), 0, len(ALPHAS) - 1))
print(f"\n=== selectors (regret vs best-alpha oracle, {n} datasets) ===")
print(f"  {'selector':28s} {'micro':>6s} {'macro':>6s}")
def ev(nm, aidx):
    regs = [reg(recs[i], aidx[i]) for i in range(n)]; print(f"  {nm:28s} {np.mean(regs):6.3f} {macro(regs):6.3f}")
ev("binary matched (local_ev)", [len(ALPHAS) - 1 if r["lev"] > TAU else 0 for r in recs])
ev("alpha-from-corr_str (rank)", a_cs)
ev("corr_str binary route", a_bin)
ev("LOO learned (spectrum feats)", list(loo))
ev("best-alpha ORACLE", [r["oai"] for r in recs])
print("saved (analysis only)")
