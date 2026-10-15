"""Per-detector lift over the prevalence/constant baseline. ap_norm = (AP - r)/(1 - r): 0 = no lift
(matches a constant/random-by-prevalence predictor, AP=r), 1 = perfect. Report mean/median ap_norm and
the fraction of tasks each detector beats the baseline (ap_norm>0), all 32 detectors, classical vs deep."""
import json, glob, os, numpy as np, pandas as pd
S = r"E:/tmp/claude/E--Projects-Submitted-ADRank/236d6247-42ad-4e62-9828-db625dfa055d/scratchpad"
CLS = os.path.join(S, "clsstd_valuni"); PF = os.path.join(S, "probefix")
DEEP = lambda d: d.startswith("D_")
clsap = {json.loads(l)["key"]: json.loads(l)["ap"] for l in open(os.path.join(CLS, "ap.jsonl"))}
pf = {}
for f in glob.glob(os.path.join(PF, "*.jsonl")):
    for ln in open(f):
        try: r = json.loads(ln)
        except Exception: continue
        if r.get("ap"): pf[r["key"]] = r["ap"]
lb = pd.read_csv(os.path.join(S, "streamline", "STREAM_V3_LEADERBOARD.csv"))
keep = lb[(lb.oracle_ap > 0.05) & (lb.reg_random > 0.02) & (lb.n_h >= 100)].set_index("key")
keys = [k for k in keep.index if k in clsap and k in pf]
print(f"tasks: {len(keys)}")
def ap_of(k, d): return pf[k].get(d) if DEEP(d) else clsap[k].get(d)
dets = sorted({d for k in keys for d in ({**clsap[k], **pf[k]})})
rows = []
for d in dets:
    v = np.array([ap_of(k, d) for k in keys if ap_of(k, d) is not None])
    rows.append(dict(detector=d, family="deep" if DEEP(d) else "classical", n=len(v),
                     mean_lift=v.mean(), median_lift=np.median(v),
                     frac_beats_baseline=float((v > 0).mean()), frac_lift_gt_0_3=float((v > 0.3).mean())))
df = pd.DataFrame(rows).sort_values("mean_lift", ascending=False).reset_index(drop=True)
df.index += 1
pd.set_option("display.float_format", lambda x: f"{x:.3f}")
print("\n=== per-detector lift over prevalence/constant baseline (ap_norm), ranked ===")
print(df.to_string())
print("\n=== family summary ===")
print(df.groupby("family").agg(mean_lift=("mean_lift","mean"), median_lift=("median_lift","median"),
                               mean_frac_beats=("frac_beats_baseline","mean")).to_string())
# how many detectors have essentially no lift over the baseline on the median task?
print(f"\ndetectors with median lift <= 0 (no better than constant on the median task): "
      f"{int((df.median_lift<=0).sum())} of {len(df)}")
print("  ->", list(df[df.median_lift<=0].detector))
df.to_csv(os.path.join(S, "DETECTOR_LIFT.csv"))
