"""(A) SPARC vs Goswami head-to-head: by pool and by data source, paired Wilcoxon + win/tie/loss.
(B) Deep detector quality: are they noise, second-best, or ever the oracle?"""
import json, glob, os, numpy as np, pandas as pd
from scipy.stats import wilcoxon
S = r"E:/tmp/claude/E--Projects-Submitted-ADRank/236d6247-42ad-4e62-9828-db625dfa055d/scratchpad"
CLS = os.path.join(S, "clsstd_valuni"); PF = os.path.join(S, "probefix"); G = os.path.join(S, "goswami2")
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
oracle = keep.oracle_ap.to_dict(); corpus = keep.corpus.to_dict()
aucs = {}
for f in glob.glob(os.path.join(G, "*.jsonl")):
    for ln in open(f):
        try: r = json.loads(ln)
        except Exception: continue
        if r.get("aucs"): aucs[r["key"]] = r["aucs"]
TYPES = ["spike","noise","scale","contextual","cutoff"]
def ap_of(k,d): return pf[k]["ap"].get(d) if DEEP(d) else clsap[k].get(d)
def a1_of(k,d): return pf[k]["a1"].get(d) if DEEP(d) else clfix[k]["a1"].get(d)
def gos(k,dets):
    names=[d for d in dets if d in aucs.get(k,{}) and all(t in aucs[k][d] for t in TYPES)]
    if len(names)<3: return None
    b={d:0.0 for d in names}
    for t in TYPES:
        for rk,d in enumerate(sorted(names,key=lambda x:aucs[k][x][t])): b[d]+=rk
    return max(names,key=lambda d:b[d])
keys=[k for k in keep.index if k in clsap and k in clfix and k in pf and k in aucs]

# (A) SPARC vs Goswami by pool
print("=== (A) SPARC vs Goswami head-to-head ===")
def pool(k,name):
    A=[d for d in ({**clsap[k],**pf[k]['ap']}) if ap_of(k,d) is not None]
    return A if name=="all" else [d for d in A if not DEEP(d)]
for pn in ["classical","all"]:
    sp,go=[],[]
    for k in keys:
        dets=pool(k,pn); o=oracle[k]
        sp.append(o-ap_of(k,max(dets,key=lambda d:a1_of(k,d))))
        gp=gos(k,dets); go.append(o-ap_of(k,gp) if gp else np.nan)
    sp,go=np.array(sp),np.array(go); m=~np.isnan(go)
    d=go[m]-sp[m]  # >0 => SPARC better
    print(f"  [{pn:9s}] SPARC={sp[m].mean():.4f} Goswami={go[m].mean():.4f} diff(Gos-SPARC)={d.mean():+.4f} "
          f"p={wilcoxon(sp[m],go[m]).pvalue:.3f} winS={int((sp[m]<go[m]).sum())} winG={int((go[m]<sp[m]).sum())} tie={int((sp[m]==go[m]).sum())}")
# by source, classical pool
print("  by source (classical pool): SPARC / Goswami")
df=pd.DataFrame([{"key":k,"src":corpus[k],
    "S":oracle[k]-ap_of(k,max(pool(k,'classical'),key=lambda d:a1_of(k,d))),
    "G":(lambda gp: oracle[k]-ap_of(k,gp) if gp else np.nan)(gos(k,pool(k,'classical')))} for k in keys])
CMAP={"ovrbench":"OvrBench","oddbench":"OddBench","tsbad_m":"TSB-AD","adbench":"ADBench","dami":"DAMI"}
df["src"]=df.src.map(CMAP)
for s in ["OvrBench","OddBench","TSB-AD"]:
    sub=df[df.src==s]; print(f"    {s:9s}(n={len(sub)}): SPARC={sub.S.mean():.3f} Goswami={sub.G.mean():.3f}")

# (B) deep quality
print("\n=== (B) deep detector quality ===")
best_cls=[max(ap_of(k,d) for d in clsap[k]) for k in keys]
best_deep=[max(pf[k]['ap'][d] for d in pf[k]['ap'] if DEEP(d)) for k in keys]
best_cls,best_deep=np.array(best_cls),np.array(best_deep)
deep_wins=int((best_deep>best_cls).sum())
print(f"  tasks where a DEEP detector is the oracle (best over all 32): {deep_wins}/{len(keys)} ({100*deep_wins/len(keys):.0f}%)")
print(f"  mean best-classical ap_norm={best_cls.mean():.3f}  mean best-deep ap_norm={best_deep.mean():.3f}  gap={best_cls.mean()-best_deep.mean():+.3f}")
print(f"  on tasks where classical wins: mean (best_cls - best_deep) = {(best_cls[best_cls>best_deep]-best_deep[best_cls>best_deep]).mean():.3f}")
print(f"  on tasks where deep wins:      mean (best_deep - best_cls) = {(best_deep[best_deep>best_cls]-best_cls[best_deep>best_cls]).mean():.3f}")
# are deep 'noise'? distribution of best-deep ap_norm
print(f"  best-deep ap_norm: median={np.median(best_deep):.3f}  frac tasks best-deep<0.05 (near-noise)={100*(best_deep<0.05).mean():.0f}%  frac>0.3={100*(best_deep>0.3).mean():.0f}%")
# per-deep-detector mean ap_norm
dets=sorted({d for k in keys for d in pf[k]['ap'] if DEEP(d)})
print("  per deep detector mean ap_norm:")
for d in dets:
    v=[pf[k]['ap'][d] for k in keys if d in pf[k]['ap']]; print(f"    {d:12s} {np.mean(v):.3f}")
