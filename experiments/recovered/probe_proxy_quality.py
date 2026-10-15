"""How good a proxy is the probe? Per task, rank-correlate each detector's probe grade (a1) with its
real performance (ap_norm), over classical(25)/deep(7)/all(32). High correlation = probe predicts real
ranking (SPARC works); low/negative = probe misleads. Locates where and for which pool the proxy breaks."""
import json, glob, os, numpy as np, pandas as pd
from scipy.stats import spearmanr
S = r"E:/tmp/claude/E--Projects-Submitted-ADRank/236d6247-42ad-4e62-9828-db625dfa055d/scratchpad"
CLS = os.path.join(S, "clsstd_valuni"); PF = os.path.join(S, "probefix")
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
clsap = {json.loads(l)["key"]: json.loads(l)["ap"] for l in open(os.path.join(CLS, "ap.jsonl"))}
pf = {}
for f in glob.glob(os.path.join(PF, "*.jsonl")):
    for ln in open(f):
        try: r = json.loads(ln)
        except Exception: continue
        if r.get("a1"): pf[r["key"]] = r
lb = pd.read_csv(os.path.join(S, "streamline", "STREAM_V3_LEADERBOARD.csv"))
keep = lb[(lb.oracle_ap > 0.05) & (lb.reg_random > 0.02) & (lb.n_h >= 100)].set_index("key")
corpus = keep.corpus.to_dict()
keys = [k for k in keep.index if k in clstd and k in clfix and k in pf and k in clsap]
def a1_of(k, d): return pf[k]["a1"].get(d) if DEEP(d) else clfix[k]["a1"].get(d)
def ap_of(k, d): return pf[k]["ap"].get(d) if DEEP(d) else clsap[k].get(d)

def corr(keys, sel):
    rows = []
    for k in keys:
        dets = [d for d in ({**clsap[k], **pf[k]['ap']}) if a1_of(k, d) is not None and ap_of(k, d) is not None and sel(d)]
        if len(dets) < 4: continue
        a1 = [a1_of(k, d) for d in dets]; ap = [ap_of(k, d) for d in dets]
        rho = spearmanr(a1, ap).correlation
        rows.append(rho if np.isfinite(rho) else np.nan)
    return np.array(rows)

print(f"tasks: {len(keys)}")
print("=== Spearman(probe a1, real ap_norm) across detectors, per task ===")
for name, sel in [("classical (25)", lambda d: not DEEP(d)), ("deep (7)", DEEP), ("all (32)", lambda d: True)]:
    r = corr(keys, sel); r = r[~np.isnan(r)]
    print(f"  {name:14s}: mean rho={np.nanmean(r):+.3f}  median={np.nanmedian(r):+.3f}  "
          f"frac rho<0 (probe misleads)={100*(r<0).mean():.0f}%  frac rho>0.5={100*(r>0.5).mean():.0f}%")

# does the probe correctly identify the ORACLE detector? rank of the true-best detector by probe grade
print("\n=== does the probe rank the truly-best detector highly? (classical pool) ===")
ranks = []
for k in keys:
    dets = [d for d in clsap[k] if a1_of(k, d) is not None and ap_of(k, d) is not None and not DEEP(d)]
    if len(dets) < 4: continue
    best_real = max(dets, key=lambda d: ap_of(k, d))
    # rank of best_real when detectors sorted by probe a1 (1 = probe's top pick)
    order = sorted(dets, key=lambda d: a1_of(k, d), reverse=True)
    ranks.append(order.index(best_real) + 1)
ranks = np.array(ranks)
print(f"  probe rank of the real-best classical detector: median={int(np.median(ranks))} of {len(dets)}  "
      f"top-1={100*(ranks==1).mean():.0f}%  top-3={100*(ranks<=3).mean():.0f}%")

# where the proxy is worst (classical): list a few tasks
rr = corr(keys, lambda d: not DEEP(d))
tk = [k for k in keys if len([d for d in clsap[k] if a1_of(k,d) is not None and ap_of(k,d) is not None])>=4]
worst = pd.DataFrame({"key": tk, "corpus": [corpus[k] for k in tk], "rho": rr}).sort_values("rho").head(8)
print("\n=== tasks where the probe proxy is worst (classical) ===")
print(worst.to_string(index=False))
