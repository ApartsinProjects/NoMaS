# -*- coding: utf-8 -*-
"""Tier-3 recompute-only ($0): #1 multiplicity (Holm over the primary test family), #2 source-grouped
robustness (leave-one-source-out + cluster bootstrap over sources), #5 Table 8 confusion matrix.
Per-task classical-pool vectors reuse the tier2 sources: SPARC=probe1000 a1, Goswami=borda, fixed=LOF_k10,
EM/MV/consensus/HITS/MC=RECONCILE_UOMS_V2, random=oracle-mean(classical ap)."""
import json, glob, os, numpy as np, pandas as pd
from scipy.stats import wilcoxon
S = r"E:/tmp/claude/E--Projects-Submitted-ADRank/236d6247-42ad-4e62-9828-db625dfa055d/scratchpad"
ST = os.path.join(S, "streamline"); PF = os.path.join(S, "probefix"); DEEP = lambda d: d.startswith("D_")
def ld(d):
    J = {}
    for fn in glob.glob(os.path.join(d, "shard_*.jsonl")):
        for ln in open(fn):
            try: r = json.loads(ln)
            except: continue
            if "key" in r: J[r["key"]] = r
    return J
clstd = ld(os.path.join(ST, "deepres_v2_clstd")); p1000 = ld(os.path.join(ST, "probe1000_results"))
pf = {}
for f in glob.glob(os.path.join(PF, "*.jsonl")):
    for ln in open(f):
        try: r = json.loads(ln)
        except: continue
        if r.get("ap"): pf[r["key"]] = r
aucs = {}
for f in glob.glob(os.path.join(S, "goswami2", "*.jsonl")):
    for ln in open(f):
        try: r = json.loads(ln)
        except: continue
        if r.get("aucs"): aucs[r["key"]] = r["aucs"]
rec = pd.read_csv(os.path.join(S, "RECONCILE_UOMS_V2.csv")).set_index("key")
lb = pd.read_csv(os.path.join(ST, "STREAM_V3_LEADERBOARD.csv"))
keep = lb[(lb.oracle_ap > 0.05) & (lb.reg_random > 0.02) & (lb.n_h >= 100)].set_index("key")
oracle = keep.oracle_ap.to_dict(); corpus = keep.corpus.to_dict()
TYPES = ["spike", "noise", "scale", "contextual", "cutoff"]
def ap_of(k, d): return pf[k]["ap"].get(d) if DEEP(d) else clstd[k]["ap"].get(d)
def borda(k, dets):
    names = [d for d in dets if d in aucs.get(k, {}) and all(t in aucs[k][d] for t in TYPES)]
    if len(names) < 6: return None
    b = {d: 0.0 for d in names}
    for t in TYPES:
        for rk, d in enumerate(sorted(names, key=lambda x: aucs[k][x][t])): b[d] += rk
    return max(names, key=lambda d: b[d])
def fam(d):
    if DEEP(d): return "deep"
    if d.startswith(("LOF", "KNN", "CBLOF")): return "neighbor"
    if d.startswith(("HBOS", "ECOD", "COPOD")): return "distribution"
    if d.startswith(("PCA", "LODA")): return "projection"
    return "isolation"
keys = [k for k in keep.index if k in clstd and k in pf and k in p1000 and k in aucs and k in rec.index and "LOF_k10" in clstd[k]["ap"]]
R = {}
sp = lambda k, pool: (oracle[k] - ap_of(k, max({d: v for d, v in p1000[k]["a1"].items() if pool(d) and ap_of(k, d) is not None}, key=lambda d: p1000[k]["a1"][d])))
R["SPARC"] = np.array([sp(k, lambda d: not DEEP(d)) for k in keys])
R["Goswami"] = np.array([oracle[k] - ap_of(k, borda(k, [d for d in aucs[k] if not DEEP(d) and ap_of(k, d) is not None])) for k in keys])
R["fixed"] = np.array([oracle[k] - clstd[k]["ap"]["LOF_k10"] for k in keys])
R["random"] = np.array([oracle[k] - np.mean([v for d, v in clstd[k]["ap"].items() if not DEEP(d)]) for k in keys])
for m, c in [("EM", "EM"), ("MV", "MV"), ("consensus", "consensus"), ("HITS", "hits"), ("MC", "centrality")]:
    R[m] = rec.loc[keys, f"{c}_cls"].values
corp = np.array([corpus[k] for k in keys])
n = len(keys); print(f"tasks: {n}")

# ---- #1 multiplicity: Holm over the primary family ----
fam_tests = [("SPARC vs fixed", R["SPARC"], R["fixed"]), ("Goswami vs fixed", R["Goswami"], R["fixed"]),
             ("EM vs fixed", R["EM"], R["fixed"]), ("MV vs fixed", R["MV"], R["fixed"]),
             ("consensus vs random", R["consensus"], R["random"]), ("HITS vs random", R["HITS"], R["random"]),
             ("MC vs random", R["MC"], R["random"]), ("SPARC vs Goswami", R["SPARC"], R["Goswami"])]
ps = [(nm, wilcoxon(a, b).pvalue) for nm, a, b in fam_tests]
order = sorted(range(len(ps)), key=lambda i: ps[i][1]); mtot = len(ps)
holm = {}
for rank, i in enumerate(order): holm[ps[i][0]] = min(1.0, ps[i][1] * (mtot - rank))
print("\n=== #1 multiplicity (Holm over the 8-test primary family) ===")
for nm, p in sorted(ps, key=lambda x: x[1]):
    print(f"  {nm:<20} raw p={p:.4f}  Holm p={holm[nm]:.4f}  {'sig' if holm[nm]<0.05 else 'ns'}")

# ---- #2 source-grouped robustness ----
print("\n=== #2 source-grouped: leave-one-source-out (SPARC vs fixed) ===")
d_all = R["SPARC"] - R["fixed"]
for c in sorted(set(corp)):
    mask = corp != c; adv = d_all[mask]
    print(f"  drop {c:<10} (n={mask.sum():3d}): SPARC advantage {(-adv.mean()):+.4f}  p={wilcoxon(R['SPARC'][mask], R['fixed'][mask]).pvalue:.4f}")
# cluster bootstrap over the 5 source collections
rng = np.random.default_rng(0); srcs = sorted(set(corp)); byc = {c: np.where(corp == c)[0] for c in srcs}
boot = []
for _ in range(5000):
    chosen = rng.choice(srcs, len(srcs), replace=True)
    idx = np.concatenate([byc[c] for c in chosen])
    boot.append((R["fixed"][idx] - R["SPARC"][idx]).mean())
boot = np.array(boot)
print(f"  cluster bootstrap (resample the 5 sources): SPARC advantage {boot.mean():+.4f}  95% CI [{np.percentile(boot,2.5):+.4f}, {np.percentile(boot,97.5):+.4f}]  P(advantage>0)={np.mean(boot>0):.3f}")

# ---- #5 confusion matrix ----
FAMS = ["neighbor", "distribution", "projection", "isolation", "deep"]
M = {p_: {o_: 0 for o_ in FAMS} for p_ in FAMS}
for k in keys:
    a1 = {d: v for d, v in p1000[k]["a1"].items() if ap_of(k, d) is not None}; ap = {d: ap_of(k, d) for d in a1}
    M[fam(max(a1, key=a1.get))][fam(max(ap, key=ap.get))] += 1
print("\n=== #5 Table 8 confusion matrix (probe #1 rows x oracle cols) ===")
print("  probe\\oracle  " + " ".join(f"{o[:5]:>6}" for o in FAMS) + "   row-tot")
for p_ in FAMS: print(f"  {p_:12s} " + " ".join(f"{M[p_][o]:>6d}" for o in FAMS) + f"   {sum(M[p_].values()):>5d}")
