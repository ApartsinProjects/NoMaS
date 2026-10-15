"""Which Goswami injection types help selection? Ablate the 5 types using the existing per-type AUCs
(goswami2). Per task: (a) each type alone (argmax that type's AUC over classical detectors), (b) Borda
over every subset of types. regret vs STREAM_V3 oracle, ap from clstd. Classical pool (Goswami's pool)."""
import json, glob, os, itertools, numpy as np, pandas as pd
from scipy.stats import rankdata, wilcoxon
S = r"E:/tmp/claude/E--Projects-Submitted-ADRank/236d6247-42ad-4e62-9828-db625dfa055d/scratchpad"
G = os.path.join(S, "goswami2")
DEEP = lambda d: d.startswith("D_")
def ld(d):
    J={}
    for fn in glob.glob(os.path.join(d,"shard_*.jsonl")):
        for ln in open(fn):
            try: r=json.loads(ln)
            except: continue
            if "key" in r: J[r["key"]]=r
    return J
clstd = ld(os.path.join(S,"streamline","deepres_v2_clstd"))
clsap = {json.loads(l)["key"]: json.loads(l)["ap"] for l in open(os.path.join(S,"clsstd_valuni","ap.jsonl"))}
lb = pd.read_csv(os.path.join(S,"streamline","STREAM_V3_LEADERBOARD.csv"))
keep = lb[(lb.oracle_ap>0.05)&(lb.reg_random>0.02)&(lb.n_h>=100)].set_index("key")
oracle = keep.oracle_ap.to_dict()
aucs = {}
for f in glob.glob(os.path.join(G,"*.jsonl")):
    for ln in open(f):
        try: r=json.loads(ln)
        except: continue
        if r.get("aucs"): aucs[r["key"]] = r["aucs"]
TYPES = ["spike","noise","scale","contextual","cutoff"]
keys = [k for k in keep.index if k in aucs and k in clsap]
def cls_dets(k): return [d for d in aucs[k] if not DEEP(d) and d in clsap[k] and all(t in aucs[k][d] for t in TYPES)]

# (a) single-type regret
print(f"tasks: {len(keys)}   (classical pool; full-5 Borda baseline = published 0.122)")
print("=== single injection type alone (argmax that type's AUC) ===")
for t in TYPES:
    reg = []
    for k in keys:
        dets = cls_dets(k)
        if len(dets) < 5: continue
        pick = max(dets, key=lambda d: aucs[k][d][t]); reg.append(oracle[k] - clsap[k][pick])
    print(f"  {t:11s}: mean regret = {np.mean(reg):.4f}")

# (b) Borda over subsets
def borda_regret(types):
    reg = []
    for k in keys:
        dets = cls_dets(k)
        if len(dets) < 5: continue
        tot = np.zeros(len(dets))
        for t in types: tot = tot + rankdata([aucs[k][d][t] for d in dets])
        reg.append(oracle[k] - clsap[k][dets[int(np.argmax(tot))]])
    return np.array(reg)
full = borda_regret(TYPES)
print(f"\nfull-5 Borda: mean regret = {full.mean():.4f}  (n={len(full)})")
rows = []
for r in range(1, 6):
    for sub in itertools.combinations(TYPES, r):
        rg = borda_regret(list(sub)); rows.append((",".join(sub), rg.mean(), len(rg)))
rows.sort(key=lambda x: x[1])
print("\n=== best subsets by Borda regret (top 10) ===")
for name, m, n in rows[:10]:
    print(f"  {m:.4f}  [{name}]")
print("\n=== worst subsets (bottom 5) ===")
for name, m, n in rows[-5:]:
    print(f"  {m:.4f}  [{name}]")
# significance of best subset vs full-5
best = rows[0]; bsub = best[0].split(",")
bestreg = borda_regret(bsub); n = min(len(bestreg), len(full))
p = wilcoxon(bestreg[:n], full[:n]).pvalue if (bestreg[:n]!=full[:n]).any() else float("nan")
print(f"\nbest subset [{best[0]}] {best[1]:.4f} vs full-5 {full.mean():.4f}: delta={best[1]-full.mean():+.4f} p={p:.3f}")
