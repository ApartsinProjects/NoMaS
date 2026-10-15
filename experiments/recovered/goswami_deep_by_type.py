"""Do deep detectors over-separate specific Goswami injection types? Per type: mean deep vs classical
AUC (over-rating gap), the all-32 deep-pick rate, and all-32 regret. Then which subset minimizes
all-32 regret. Uses goswami2 per-type AUCs (all 32) + clstd (classical ap) + probefix (deep ap)."""
import json, glob, os, itertools, numpy as np, pandas as pd
from scipy.stats import rankdata
S = r"E:/tmp/claude/E--Projects-Submitted-ADRank/236d6247-42ad-4e62-9828-db625dfa055d/scratchpad"
G = os.path.join(S, "goswami2"); DEEP = lambda d: d.startswith("D_")
clsap = {json.loads(l)["key"]: json.loads(l)["ap"] for l in open(os.path.join(S, "clsstd_valuni", "ap.jsonl"))}
pf = {}
for f in glob.glob(os.path.join(S, "probefix", "*.jsonl")):
    for ln in open(f):
        try: r = json.loads(ln)
        except Exception: continue
        if r.get("ap"): pf[r["key"]] = {d: v for d, v in r["ap"].items() if DEEP(d)}
lb = pd.read_csv(os.path.join(S, "streamline", "STREAM_V3_LEADERBOARD.csv"))
keep = lb[(lb.oracle_ap > 0.05) & (lb.reg_random > 0.02) & (lb.n_h >= 100)].set_index("key")
oracle = keep.oracle_ap.to_dict()
aucs = {}
for f in glob.glob(os.path.join(G, "*.jsonl")):
    for ln in open(f):
        try: r = json.loads(ln)
        except Exception: continue
        if r.get("aucs"): aucs[r["key"]] = r["aucs"]
TYPES = ["spike", "noise", "scale", "contextual", "cutoff"]
keys = [k for k in keep.index if k in aucs and k in clsap and k in pf]
def ap_of(k, d): return pf[k].get(d) if DEEP(d) else clsap[k].get(d)
def dets32(k): return [d for d in aucs[k] if ap_of(k, d) is not None and all(t in aucs[k][d] for t in TYPES)]

print(f"tasks: {len(keys)}")
print("=== per injection type: mean AUC deep vs classical (over-separation), all-32 deep-pick rate, regret ===")
print(f"{'type':11s} {'deepAUC':>8} {'clsAUC':>8} {'gap':>7} {'deep-pick%':>10} {'regret':>8}")
for t in TYPES:
    da, ca, dpick, reg = [], [], 0, []
    for k in keys:
        ds = dets32(k)
        for d in ds: (da if DEEP(d) else ca).append(aucs[k][d][t])
        pick = max(ds, key=lambda d: aucs[k][d][t])
        if DEEP(pick): dpick += 1
        reg.append(oracle[k] - ap_of(k, pick))
    print(f"{t:11s} {np.mean(da):>8.3f} {np.mean(ca):>8.3f} {np.mean(da)-np.mean(ca):>+7.3f} {100*dpick/len(keys):>9.0f}% {np.mean(reg):>8.4f}")

def borda32(types):
    reg, dpick = [], 0
    for k in keys:
        ds = dets32(k)
        if len(ds) < 5: continue
        tot = sum(rankdata([aucs[k][d][t] for d in ds]) for t in types)
        pick = ds[int(np.argmax(tot))]; reg.append(oracle[k] - ap_of(k, pick))
        if DEEP(pick): dpick += 1
    return np.mean(reg), 100*dpick/len(reg)
full, fp = borda32(TYPES)
print(f"\nall-32 Borda full-5: regret={full:.4f}  deep-pick={fp:.0f}%  (classical-pool Goswami=0.122)")
rows = []
for r in range(1, 6):
    for sub in itertools.combinations(TYPES, r):
        m, dp = borda32(list(sub)); rows.append((",".join(sub), m, dp))
rows.sort(key=lambda x: x[1])
print("=== best all-32 subsets (top 6): regret, deep-pick% ===")
for name, m, dp in rows[:6]: print(f"  {m:.4f}  deep-pick={dp:>3.0f}%  [{name}]")
