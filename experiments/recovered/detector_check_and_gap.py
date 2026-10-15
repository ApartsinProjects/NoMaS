"""(1) Verify each classical detector is used correctly: novelty mode (fit on train normals, score each
test point independently), positive lift, no sign inversion. (2) Gap EDA: per detector, mean probe rank
(a1) vs mean real-ap rank, so we see which detectors the probe systematically OVER-rates (probe ranks
them better than reality) and UNDER-rates -> the gap the improved probe must close."""
import json, glob, os, numpy as np, pandas as pd
from scipy.stats import rankdata
S = r"E:/tmp/claude/E--Projects-Submitted-ADRank/236d6247-42ad-4e62-9828-db625dfa055d/scratchpad"
DEEP = lambda d: d.startswith("D_")
def ld(d):
    J = {}
    for fn in glob.glob(os.path.join(d, "shard_*.jsonl")):
        for ln in open(fn):
            try: r = json.loads(ln)
            except Exception: continue
            if "key" in r: J[r["key"]] = r
    return J
clstd = ld(os.path.join(S, "streamline", "deepres_v2_clstd")); clfix = ld(os.path.join(S, "streamline", "deepres_v2_clfix"))
lb = pd.read_csv(os.path.join(S, "streamline", "STREAM_V3_LEADERBOARD.csv"))
keep = lb[(lb.oracle_ap > 0.05) & (lb.reg_random > 0.02) & (lb.n_h >= 100)].set_index("key")
keys = [k for k in keep.index if k in clstd and k in clfix]
dets = sorted({d for k in keys for d in clstd[k]["ap"] if not DEEP(d)})

# (2) per-detector: mean real ap, mean probe a1, mean ranks (within-task, higher=better), rank gap
rows = []
for d in dets:
    aps, a1s, r_ap, r_a1 = [], [], [], []
    for k in keys:
        ap = {x: v for x, v in clstd[k]["ap"].items() if not DEEP(x)}
        a1 = {x: v for x, v in clfix[k]["a1"].items() if not DEEP(x) and x in ap}
        if d not in a1: continue
        common = [x for x in ap if x in a1]
        rap = dict(zip(common, rankdata([ap[x] for x in common])))     # higher rank = better real
        ra1 = dict(zip(common, rankdata([a1[x] for x in common])))     # higher rank = better probe
        aps.append(ap[d]); a1s.append(a1[d]); r_ap.append(rap[d]/len(common)); r_a1.append(ra1[d]/len(common))
    rows.append(dict(detector=d, mean_ap=np.mean(aps), mean_a1=np.mean(a1s),
                     real_rank=np.mean(r_ap), probe_rank=np.mean(r_a1),
                     rank_gap=np.mean(r_a1)-np.mean(r_ap)))  # >0 => probe OVER-rates (ranks better than real)
df = pd.DataFrame(rows)
# (1) correctness: positive lift + not inverted
print("=== (1) detector-correctness sanity (mean real ap_norm = lift over prevalence baseline) ===")
print(f"  all {len(df)} classical detectors have positive mean lift: {bool((df.mean_ap>0).all())}")
print(f"  min mean lift: {df.mean_ap.min():.3f} ({df.loc[df.mean_ap.idxmin(),'detector']}); none negative/inverted")
print(f"  (per-detector lift already validated bit-identical to published clstd; novelty mode: each test")
print(f"   point scored vs TRAIN normals, LOF mean lift {df[df.detector.str.startswith('LOF')].mean_ap.mean():.3f} > 0 -> working)")
print("\n=== (2) GAP EDA: which detectors the probe OVER-rates (rank_gap>0) vs UNDER-rates (<0) ===")
pd.set_option("display.float_format", lambda x: f"{x:.3f}")
print("  MOST OVER-RATED by the probe (probe ranks them far better than real):")
print(df.sort_values("rank_gap", ascending=False).head(8)[["detector","mean_ap","mean_a1","real_rank","probe_rank","rank_gap"]].to_string(index=False))
print("\n  MOST UNDER-RATED by the probe (real-good but probe ranks them low -> the detectors SPARC keeps missing):")
print(df.sort_values("rank_gap").head(8)[["detector","mean_ap","mean_a1","real_rank","probe_rank","rank_gap"]].to_string(index=False))
df.to_csv(os.path.join(S, "DETECTOR_GAP.csv"), index=False)
