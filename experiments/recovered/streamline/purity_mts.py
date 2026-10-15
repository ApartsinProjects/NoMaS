# -*- coding: utf-8 -*-
"""Is the MTS 'hard anomaly' pattern a LABELING artifact? window_labels marks a window anomaly if it
contains >=1 of W=64 anomaly points -> edge-clipped windows (1/64 anomalous, 63 normal) are labeled
anomaly but look normal. Measure anomaly-window PURITY = (#anomaly points)/W across the MTS benchmark,
and specifically on the 2 global MV-wins MTS datasets (GECCO, CreditCard)."""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"; S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
D = os.path.join(S, "scratchpad", "streamline"); sys.path.insert(0, os.path.join(S, "scratchpad"))
from hadb_ts_final import W, STRIDE
from hadb_ts_mts import load_mts
# datasets actually IN the streamlined benchmark (tsbad_m rows)
rk = pd.read_csv(os.path.join(D, "STREAM_RANK.csv")); inbench = set(rk[rk.corpus == "tsbad_m"].dataset)
glob_mv = {"173_GECCO_id_1_Sensor_tr_16165_1st_16265"[:40], "137_CreditCard_id_1_Finance_tr_500_1st_5"[:40]}
rows = []
for name, src, Xc, lab in load_mts(200):
    lab = np.asarray(lab, int).ravel(); n = len(lab)
    if n < W + STRIDE or lab.sum() == 0: continue
    starts = np.arange(0, n - W + 1, STRIDE)
    pur = np.array([lab[s:s + W].sum() / W for s in starts])   # anomaly fraction per window
    aw = pur[pur > 0]   # windows labeled anomaly (>=1 point)
    if len(aw) == 0: continue
    rows.append({"dataset": str(name)[:40], "n_anom_win": len(aw), "median_purity": round(float(np.median(aw)), 3),
                 "frac_below_0.5": round(float((aw < 0.5).mean()), 3), "frac_below_0.25": round(float((aw < 0.25).mean()), 3),
                 "frac_single_pt": round(float((aw <= 1.0 / W + 1e-9).mean()), 3),
                 "inbench": str(name)[:40] in inbench, "glob_mv": str(name)[:40] in glob_mv})
df = pd.DataFrame(rows)
ib = df[df.inbench]
print(f"=== MTS anomaly-window PURITY (anomaly fraction of the 64-pt window); rule labels anomaly if >=1 pt ===")
print(f"  total MTS series with anomalies: {len(df)};  in streamlined benchmark: {len(ib)}\n")
print("  --- across benchmark MTS datasets ---")
print(f"  median anomaly-window purity (median over datasets): {ib.median_purity.median():.3f}")
print(f"  mean frac of anomaly windows that are <50% anomaly (edge-clipped): {ib['frac_below_0.5'].mean():.3f}")
print(f"  mean frac that are <25% anomaly:                                    {ib['frac_below_0.25'].mean():.3f}")
print(f"  mean frac that are a SINGLE anomaly point (1/64):                   {ib['frac_single_pt'].mean():.3f}")
print(f"  datasets where >50% of 'anomaly' windows are edge-clipped (<50% pure): {int((ib['frac_below_0.5']>0.5).sum())}/{len(ib)}")
print("\n  --- the 2 GLOBAL MV-wins MTS datasets ---")
gm = df[df.glob_mv]
print(gm[["dataset", "n_anom_win", "median_purity", "frac_below_0.5", "frac_below_0.25", "frac_single_pt"]].to_string(index=False) if len(gm) else "  (not matched)")
df.to_csv(os.path.join(D, "STREAM_PURITY_MTS.csv"), index=False)
print("\nsaved streamline/STREAM_PURITY_MTS.csv")
