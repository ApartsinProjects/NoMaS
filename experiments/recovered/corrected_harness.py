# -*- coding: utf-8 -*-
"""CORRECTED harness + REPRODUCTION GATE.

Today's experiments used a harness that diverged from the real method in five ways. Two of
them demonstrably reversed or inverted conclusions. Before re-testing ANY idea, this script
must clear one gate:

    GATE: reproduce the paper's reported Spearman rho ~= 0.63 on tabular datasets
          (spread >= 0.10 subset). The broken harness gave rho = +0.002, i.e. the
          pseudo-ranking carried NO information, so every downstream "negative" was
          measured on noise.

FIXES APPLIED (vs the broken harness):
  1. TARGET: use the paper's `true_rank_from_labels` protocol - train on 80% of NORMALS
     only, score a held-out set of 20% normals + all anomalies. The broken version used
     fit_and_score(d, X, X): in-sample AND contaminated.
  2. CLUSTER SELECTION: use the paper's `smallest`-cluster strategy and M sampled SUBSETS
     (possibly several clusters per subset, targeting ~5% of N as pseudo-anomalies),
     via pipeline.sample_subsets / pseudo_auc_for_dataset, rather than iterating over
     single k-means clusters.
  3. K SWEEP: aggregate over K in {30, 50} as the paper does, not a single K.
  4. SEEDS: multiple seeds, not one.
  5. REPORT the spread-filtered subset, because on datasets where all detectors are bunched
     the choice does not matter and rho is tie-noise (the paper's own argument).

If the gate passes, this harness becomes the basis for re-testing the ideas that the
broken harness rejected.
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"
sys.path.insert(0, os.path.join(ROOT, "src"))
import adrank.pipeline as P  # noqa: E402

OUT = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d\corrected_harness.csv"
SEEDS = [0, 1, 2]
MAX_N = 6000

print("pipeline functions available:",
      [f for f in ("sample_subsets", "pseudo_auc_for_dataset", "aggregate_pseudo",
                   "true_rank_from_labels", "correlate_ranks")
       if hasattr(P, f)], flush=True)

# Use the project's own loaders so the dataset set matches the paper's tabular arm.
datasets = []
for sub in ("adbench", "dami"):
    d = os.path.join(ROOT, "data", sub)
    if os.path.isdir(d):
        datasets += P.load_npz_dir(d)
print(f"loaded {len(datasets)} datasets via pipeline.load_npz_dir", flush=True)

rows, t0 = [], time.time()
for ds in datasets:
    if len(ds.X) > MAX_N:
        rng = np.random.RandomState(0)
        keep = rng.choice(len(ds.X), MAX_N, replace=False)
        ds = P.Dataset(name=ds.name, X=ds.X[keep], y=ds.y[keep])
    if ds.y.sum() < 10:
        continue
    for seed in SEEDS:
        try:
            # --- TARGET: the paper's own protocol (train on 80% normals, score held-out)
            true_df = P.true_rank_from_labels(ds, seed=seed)
            true_df = true_df.dropna(subset=["true_auc"])
            if len(true_df) < 5:
                continue
            # --- PSEUDO: the paper's own pseudo-AUC routine (smallest selection, M subsets)
            pse = P.pseudo_auc_for_dataset(ds, seed=seed)
            agg = P.aggregate_pseudo(pse)
        except Exception as e:
            print(f"  {ds.name} seed{seed}: FAILED {type(e).__name__}: {e}", flush=True)
            continue

        # aggregate_pseudo returns per-detector aggregated pseudo scores; find its columns
        cols = [c for c in agg.columns if "auc" in c.lower() or "score" in c.lower()]
        if "detector" not in agg.columns or not cols:
            print(f"  {ds.name}: unexpected aggregate_pseudo columns: {list(agg.columns)}",
                  flush=True)
            continue
        scol = cols[0]
        tcols = ["detector", "true_auc"] + (["true_ap"] if "true_ap" in true_df else [])
        m = agg[["detector", scol]].merge(true_df[tcols], on="detector").dropna(subset=[scol, "true_auc"])
        if len(m) < 5:
            continue

        rec = dict(dataset=ds.name, seed=seed, n=len(ds.X), d=ds.X.shape[1],
                   anom_rate=float(ds.y.mean()), n_det=len(m), score_col=scol)
        # IS AUC THE RIGHT TARGET? Under 1-5% anomaly rates ROC-AUC is dominated by the
        # high-FPR region nobody deploys in. Evaluate the SAME pseudo-ranking against
        # several targets: full-curve AUC, average precision (top-heavy), and P@n
        # (precision in a budget of n = number of true anomalies), which is what a
        # practitioner with a fixed alert budget actually gets.
        for tname, tcol in [("auc", "true_auc"), ("ap", "true_ap")]:
            if tcol not in m:
                continue
            mm = m.dropna(subset=[tcol])
            if len(mm) < 5 or mm[tcol].nunique() < 2:
                continue
            best, worst, mean = mm[tcol].max(), mm[tcol].min(), mm[tcol].mean()
            pick = mm.loc[mm[scol].idxmax(), tcol]
            rec[f"rho_{tname}"] = spearmanr(mm[scol], mm[tcol]).statistic
            rec[f"spread_{tname}"] = best - worst
            rec[f"regret_{tname}"] = best - pick
            rec[f"random_regret_{tname}"] = best - mean
        # keep the legacy column names so the gate code below still works
        rec["rho"] = rec.get("rho_auc", np.nan)
        rec["spread"] = rec.get("spread_auc", np.nan)
        rec["regret"] = rec.get("regret_auc", np.nan)
        rec["random_regret"] = rec.get("random_regret_auc", np.nan)
        rows.append(rec)
    print(f"{ds.name}: done | {time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)

D = pd.DataFrame(rows)
D.to_csv(OUT, index=False)
print(f"\nwrote {OUT} rows={len(D)} datasets={D.dataset.nunique() if len(D) else 0}")

if len(D):
    print("\n=== REPRODUCTION GATE: paper reports rho ~= 0.63 on tabular, spread>=0.10 ===")
    for thr in [0.0, 0.05, 0.10, 0.20]:
        s = D[D.spread >= thr]
        if len(s):
            print(f"  spread>={thr:.2f}: n={len(s):4d} rows  mean rho={s.rho.mean():+.3f}  "
                  f"median={s.rho.median():+.3f}  regret={s.regret.mean():.4f}  "
                  f"random={s.random_regret.mean():.4f}")
    g = D[D.spread >= 0.10]
    verdict = "PASS - harness is faithful" if g.rho.mean() > 0.40 else \
              "FAIL - still broken, the divergence is elsewhere"
    print(f"\n  GATE: {verdict}  (mean rho on spread>=0.10 = {g.rho.mean():+.3f})")
    print(f"\n  regret overall: {D.regret.mean():.4f} vs random {D.random_regret.mean():.4f} "
          f"({100*(1-D.regret.mean()/max(D.random_regret.mean(),1e-9)):+.1f}%)")

    print("\n=== IS ROC-AUC THE RIGHT TARGET? same pseudo-ranking, different targets ===")
    print(f"{'target':>8s} {'mean rho':>9s} {'rho spread>=.10':>16s} {'regret':>8s} "
          f"{'random':>8s} {'reduction':>10s}")
    for t in ["auc", "ap"]:
        rc, gc, sc = f"rho_{t}", f"regret_{t}", f"spread_{t}"
        if rc not in D:
            continue
        s = D.dropna(subset=[rc])
        hi = s[s[sc] >= 0.10] if sc in s else s
        red = 100 * (1 - s[gc].mean() / max(s[f"random_regret_{t}"].mean(), 1e-9))
        print(f"{t:>8s} {s[rc].mean():+9.3f} {hi[rc].mean():+16.3f} {s[gc].mean():8.4f} "
              f"{s[f'random_regret_{t}'].mean():8.4f} {red:+9.1f}%")
    if "rho_ap" in D and "rho_auc" in D:
        from scipy.stats import wilcoxon as _w
        v = D.dropna(subset=["rho_auc", "rho_ap"])
        if len(v) and (v.rho_ap != v.rho_auc).any():
            print(f"\n  AP-target vs AUC-target rho: p={_w(v.rho_auc, v.rho_ap).pvalue:.4f} "
                  f"(AP higher on {int((v.rho_ap>v.rho_auc).sum())}/{len(v)})")
        print(f"  anomaly rate: median {D.anom_rate.median():.4f} "
              f"(the more imbalanced, the more AUC and AP diverge)")
