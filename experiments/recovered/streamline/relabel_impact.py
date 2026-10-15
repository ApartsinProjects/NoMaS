# -*- coding: utf-8 -*-
"""STAGE 1: impact of a PURITY-THRESHOLD window-labeling rule on the MTS datasets in the streamlined
benchmark. Current rule: anomaly iff >=1 of 64 pts. New rule: anomaly iff purity>=T; 0<purity<T ->
AMBIGUOUS (excluded, NOT normal, so normals stay clean); purity==0 -> normal. Report per-dataset
anomaly-window counts under T in {current, 0.25, 0.5} and dataset survival at MIN_HARD=100."""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"; S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
D = os.path.join(S, "scratchpad", "streamline"); sys.path.insert(0, os.path.join(S, "scratchpad"))
from hadb_ts_final import W, STRIDE
from hadb_ts_mts import load_mts
MIN_HARD = 100
rk = pd.read_csv(os.path.join(D, "STREAM_RANK.csv")); inbench = set(rk[rk.corpus == "tsbad_m"].dataset)
print(f"MTS (tsbad_m) datasets in streamlined benchmark: {len(inbench)}")
rows = []
for name, src, Xc, lab in load_mts(200):
    nm = str(name)[:40]
    if nm not in inbench: continue
    lab = np.asarray(lab, int).ravel(); n = len(lab)
    starts = np.arange(0, n - W + 1, STRIDE); pur = np.array([lab[s:s + W].sum() / W for s in starts])
    cur = int((pur > 0).sum()); p25 = int((pur >= 0.25).sum()); p50 = int((pur >= 0.5).sum())
    rows.append({"dataset": nm, "n_hard_current": cur, "n_hard_p25": p25, "n_hard_p50": p50,
                 "median_purity": round(float(np.median(pur[pur > 0])), 3) if cur else 0.0,
                 "surv_cur": cur >= MIN_HARD, "surv_p25": p25 >= MIN_HARD, "surv_p50": p50 >= MIN_HARD})
df = pd.DataFrame(rows).sort_values("median_purity"); df.to_csv(os.path.join(D, "STREAM_RELABEL_IMPACT.csv"), index=False)
print(f"\n=== survival at MIN_HARD>={MIN_HARD} anomaly windows ===")
print(f"  current (>=1pt):   {int(df.surv_cur.sum())}/{len(df)} datasets")
print(f"  purity>=0.25:      {int(df.surv_p25.sum())}/{len(df)} datasets  ({int((df.surv_cur & ~df.surv_p25).sum())} drop)")
print(f"  purity>=0.50:      {int(df.surv_p50.sum())}/{len(df)} datasets  ({int((df.surv_cur & ~df.surv_p50).sum())} drop)")
drops25 = df[df.surv_cur & ~df.surv_p25]; drops50 = df[df.surv_cur & ~df.surv_p50]
print(f"\n  datasets that DROP under purity>=0.25: {list(drops25.dataset)}")
print(f"  additional drops going to >=0.50:      {list(df[df.surv_p25 & ~df.surv_p50].dataset)}")
print("\n  --- lowest-purity datasets (most affected) ---")
print(df.head(10)[["dataset", "median_purity", "n_hard_current", "n_hard_p25", "n_hard_p50"]].to_string(index=False))
print("\nsaved streamline/STREAM_RELABEL_IMPACT.csv")
