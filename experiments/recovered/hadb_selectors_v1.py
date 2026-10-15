# -*- coding: utf-8 -*-
"""HADB selection-method comparison, round 1: reference baselines only (no NoMaS yet).

WHAT A SELECTOR DOES: for each dataset it picks ONE detector from that dataset's candidate
pool, WITHOUT labels, and is scored by the regret of that pick against the label-derived best.

  regret(dataset)      = best_ap_norm - ap_norm(picked)          (0 = picked the best)
  norm_rank(dataset)   = rank of picked among the pool / (pool-1) (0 = best, 1 = worst)

These are the Ma et al. (UOMS) framing, ported to ap_norm and to the normals-only protocol.

BASELINES computable from the SAVED metrics (this file):
  oracle_best     the label-derived best. Regret 0 by definition - a SANITY anchor.
  anti_oracle     always the worst. Upper bound on regret - a SANITY anchor.
  random          expected pick = the pool mean. E[regret] = best - mean, E[norm_rank]=0.5.
  global_fixed    the single detector with the best MEAN ap_norm across the benchmark, applied
                  everywhere (chosen PER MODALITY, since the two pools differ). This is UOMS's
                  strong "best fixed model" reference; choosing it uses benchmark-wide labels,
                  so it is a reference bar, not a deployable method.
  iforest_random  random among IForest configs (UOMS's IFOREST-R): a cheap default.
  stability       (tabular only) pick the detector whose split-half score ranking is most
                  reproducible. Label-free and deployable. We recorded stability but never used
                  it to select; this is its first real test.

LABEL-FREE LITERATURE METHODS (EM/MV, consensus, ModelCentrality, HITS) need the raw per-point
score VECTORS, which these CSVs do not carry. They come in round 2 after a score-saving re-run.

INVARIANTS (stated before running):
  I1 oracle regret == 0 exactly.
  I2 anti_oracle regret >= every other selector, every dataset.
  I3 random norm_rank ~ 0.5.
  I4 global_fixed regret <= random regret on average (a smart fixed pick beats a blind one) -
     this is the Ma et al. result; violation = bug.
"""
import os, warnings
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")
S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
M = pd.read_csv(os.path.join(S, "HADB_MANIFEST.csv"))
inc = M[M.include]
ARMS = {"adbench_dami": "hadb_v3.csv", "oddbench": "hadb_oddbench.csv",
        "ovrbench": "hadb_ovrbench.csv", "ucr": "hadb_ts_ucr.csv",
        "tsbad_u": "hadb_ts_tsbad.csv", "tsbad_m": "hadb_ts_mts.csv"}

# ---- build per (dataset, variant) mean ap_norm and mean stability ----
rows = []
for corp, f in ARMS.items():
    D = pd.read_csv(os.path.join(S, f))
    D = D[D.dataset.isin(set(inc[inc.corpus == corp].dataset))].copy()
    if not len(D):
        continue
    D["ap_norm"] = (D.ap - D.base_rate) / (1 - D.base_rate)
    D["modality"] = "tabular" if corp in ("adbench_dami", "oddbench", "ovrbench") else "timeseries"
    D["corpus"] = corp
    rows.append(D)
A = pd.concat(rows, ignore_index=True)
has_stab = "stability" in A.columns

# per (dataset, variant): mean over seeds
g = A.groupby(["modality", "corpus", "dataset", "variant"], as_index=False).agg(
    ap_norm=("ap_norm", "mean"),
    stability=("stability", "mean") if has_stab else ("ap_norm", "mean"))

# ---- global fixed detector, chosen per modality on mean ap_norm ----
global_fixed = {}
for mod in g.modality.unique():
    sub = g[g.modality == mod]
    best_v = sub.groupby("variant").ap_norm.mean().idxmax()
    global_fixed[mod] = best_v
print("global fixed detector per modality:", global_fixed)

# ---- evaluate selectors per dataset ----
def is_iforest(v):
    return str(v).startswith(("IForest", "IF_"))

recs = []
for (mod, corp, ds), gd in g.groupby(["modality", "corpus", "dataset"]):
    gd = gd.reset_index(drop=True)
    ap = gd.set_index("variant").ap_norm
    pool = len(ap)
    best, worst, mean_ = ap.max(), ap.min(), ap.mean()
    order = ap.sort_values(ascending=False)          # best -> worst
    rankpos = {v: i for i, v in enumerate(order.index)}  # 0 = best

    def entry(name, picked_ap, picked_rank):
        return dict(dataset=ds, modality=mod, corpus=corp, selector=name, pool=pool,
                    regret=best - picked_ap,
                    norm_rank=picked_rank / (pool - 1) if pool > 1 else 0.0)

    # oracle / anti-oracle
    recs.append(entry("oracle_best", best, 0))
    recs.append(entry("anti_oracle", worst, pool - 1))
    # random: expectation over uniform pick
    recs.append(entry("random", mean_, (pool - 1) / 2.0))
    # global fixed
    vf = global_fixed[mod]
    if vf in ap.index:
        recs.append(entry("global_fixed", ap[vf], rankpos[vf]))
    # iforest-random
    ifs = [v for v in ap.index if is_iforest(v)]
    if ifs:
        recs.append(entry("iforest_random", ap[ifs].mean(),
                          np.mean([rankpos[v] for v in ifs])))
    # stability (tabular only, where recorded and non-degenerate)
    if has_stab and mod == "tabular" and gd.stability.notna().any():
        pick = gd.loc[gd.stability.idxmax(), "variant"]
        recs.append(entry("stability", ap[pick], rankpos[pick]))

R = pd.DataFrame(recs)
R.to_csv(os.path.join(S, "HADB_SELECTORS_V1.csv"), index=False)

# ---- invariants ----
print("\n=== INVARIANTS ===")
orc = R[R.selector == "oracle_best"].regret.abs().max()
print(f"  I1 oracle regret == 0            : max|regret|={orc:.2e}  "
      f"{'PASS' if orc < 1e-9 else 'FAIL'}")
piv = R.pivot_table(index="dataset", columns="selector", values="regret")
anti_ok = all((piv["anti_oracle"] >= piv[c] - 1e-9).all() for c in piv.columns)
print(f"  I2 anti_oracle >= all others     : {'PASS' if anti_ok else 'FAIL'}")
rr = R[R.selector == "random"].norm_rank.mean()
print(f"  I3 random norm_rank ~ 0.5        : {rr:.3f}  {'PASS' if abs(rr-0.5)<0.02 else 'FAIL'}")
gf = R[R.selector == "global_fixed"].regret.mean()
rnd = R[R.selector == "random"].regret.mean()
print(f"  I4 global_fixed <= random regret : {gf:.4f} vs {rnd:.4f}  "
      f"{'PASS' if gf <= rnd else 'FAIL'}")

# ---- results table ----
def summarize(sub):
    out = []
    piv = sub.pivot_table(index="dataset", columns="selector", values="regret")
    for sel in ["oracle_best", "global_fixed", "iforest_random", "stability", "random", "anti_oracle"]:
        s = sub[sub.selector == sel]
        if not len(s):
            continue
        row = dict(selector=sel, n=len(s),
                   regret=round(s.regret.mean(), 4),
                   regret_med=round(s.regret.median(), 4),
                   norm_rank=round(s.norm_rank.mean(), 3),
                   pct_optimal=round(100 * (s.regret < 1e-9).mean(), 1))
        if sel not in ("oracle_best", "random") and "random" in piv:
            common = piv[[sel, "random"]].dropna()
            if len(common) > 5 and (common[sel] != common["random"]).any():
                try:
                    row["p_vs_random"] = round(wilcoxon(common[sel], common["random"]).pvalue, 4)
                except Exception:
                    row["p_vs_random"] = np.nan
        out.append(row)
    return pd.DataFrame(out)

print("\n" + "=" * 74)
print("SELECTOR COMPARISON  (regret on ap_norm; lower = better; norm_rank 0=best 1=worst)")
print("=" * 74)
print("\n--- OVERALL (307 datasets) ---")
print(summarize(R).to_string(index=False))
for mod in ["tabular", "timeseries"]:
    print(f"\n--- {mod} ---")
    print(summarize(R[R.modality == mod]).to_string(index=False))

print("\n  written: HADB_SELECTORS_V1.csv")
print("  NOTE: EM/MV, consensus, ModelCentrality, HITS need raw score vectors (round 2).")
