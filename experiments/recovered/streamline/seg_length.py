# -*- coding: utf-8 -*-
"""Define & identify POINT/SHORT-anomaly MTS datasets directly from raw label runs. A window (W=64) can
only be majority-anomaly if the anomaly SEGMENT covering it is >= W/2=32 raw points. So a dataset whose
anomaly content lives mostly in segments < 32 produces only diluted anomaly windows -> windowing is an
artifact. Rule: drop if frac of anomaly POINTS in segments >= W/2 is < 0.5 (bulk of anomalies too short)."""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"; S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
D = os.path.join(S, "scratchpad", "streamline"); sys.path.insert(0, os.path.join(S, "scratchpad"))
from hadb_ts_final import W, STRIDE
from hadb_ts_mts import load_mts
HALF = W // 2  # 32
def segments(lab):
    lab = np.asarray(lab, int).ravel(); d = np.diff(np.r_[0, lab, 0])
    starts = np.where(d == 1)[0]; ends = np.where(d == -1)[0]; return ends - starts   # segment lengths
rk = pd.read_csv(os.path.join(D, "STREAM_RANK.csv")); inbench = set(rk[rk.corpus == "tsbad_m"].dataset)
rows = []
for name, src, Xc, lab in load_mts(200):
    nm = str(name)[:40]
    if nm not in inbench: continue
    seglen = segments(lab)
    if len(seglen) == 0: continue
    total_pts = seglen.sum(); frac_long = seglen[seglen >= HALF].sum() / total_pts   # anomaly points in segments >=32
    rows.append({"dataset": nm, "n_seg": len(seglen), "med_seglen": int(np.median(seglen)), "max_seglen": int(seglen.max()),
                 "mean_seglen": round(float(seglen.mean()), 1), "frac_pts_in_long_seg": round(float(frac_long), 3),
                 "point_anom": bool(frac_long < 0.5)})
df = pd.DataFrame(rows).sort_values("frac_pts_in_long_seg"); df.to_csv(os.path.join(D, "STREAM_SEGLEN_MTS.csv"), index=False)
drop = df[df.point_anom]
print(f"=== anomaly-SEGMENT structure across {len(df)} benchmark MTS datasets (W={W}, threshold W/2={HALF}) ===")
print("  frac_pts_in_long_seg = fraction of anomaly POINTS living in segments >= 32 raw pts (windowable)\n")
print(df.head(12)[["dataset", "n_seg", "med_seglen", "max_seglen", "frac_pts_in_long_seg", "point_anom"]].to_string(index=False))
print(f"\n  POINT/SHORT-anomaly datasets (frac_pts_in_long_seg < 0.5): {len(drop)}/{len(df)}")
for _, r in drop.iterrows(): print(f"    {r.dataset:42s} med_seg={r.med_seglen:4d}  max_seg={r.max_seglen:5d}  frac_long={r.frac_pts_in_long_seg:.3f}")
print(f"\n  survivors: {len(df)-len(drop)}   (all have the bulk of anomaly content in >=32-pt segments)")
print("saved streamline/STREAM_SEGLEN_MTS.csv")
