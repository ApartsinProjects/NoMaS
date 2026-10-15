"""Deep-only column for every method + why deep is bad (overfitting: high probe-AUC, low real-ap).
Pools: classical(25), deep(7), all(32). oracle STREAM_V3; classical ap=clsstd, a1=clfix; deep ap+a1
from probefix; val/uni classical from clsstd, deep from probefix; Goswami from goswami2 aucs."""
import json, glob, os, sys, numpy as np, pandas as pd
sys.path.insert(0, r"E:/Projects/Submitted/zenodo_build/code/lib")
from label_free_criteria import consensus_scores, model_centrality_scores, hits_authority_scores, _em_auc, _mv_auc
S = r"E:/tmp/claude/E--Projects-Submitted-ADRank/236d6247-42ad-4e62-9828-db625dfa055d/scratchpad"
CLS = os.path.join(S, "clsstd_valuni"); PF = os.path.join(S, "probefix"); G = os.path.join(S, "goswami2")
DEEP = lambda d: d.startswith("D_")
def loaddir(d):
    J = {}
    for fn in glob.glob(os.path.join(d, "shard_*.jsonl")):
        for ln in open(fn):
            try: r = json.loads(ln)
            except Exception: continue
            if "key" in r: J[r["key"]] = r
    return J
clsap = {json.loads(l)["key"]: json.loads(l)["ap"] for l in open(os.path.join(CLS, "ap.jsonl"))}
clfix = loaddir(os.path.join(S, "streamline", "deepres_v2_clfix"))
pf = {}
for f in glob.glob(os.path.join(PF, "*.jsonl")):
    for ln in open(f):
        try: r = json.loads(ln)
        except Exception: continue
        if r.get("a1"): pf[r["key"]] = r
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
def ap_of(k, d): return pf[k]["ap"].get(d) if DEEP(d) else clsap[k].get(d)
def a1_of(k, d): return pf[k]["a1"].get(d) if DEEP(d) else clfix[k]["a1"].get(d)
def valuni(k):
    val, uni = {}, {}
    zc = np.load(os.path.join(CLS, f"sc_{k}.npz"))
    for kk in zc.files:
        (val if kk[:5]=="val__" else uni)[kk[5:]] = zc[kk]
    zp = np.load(os.path.join(PF, f"sc_{k}.npz"))
    for kk in zp.files:
        if DEEP(kk[5:]): (val if kk[:5]=="val__" else uni)[kk[5:]] = zp[kk]
    return val, uni
t = np.linspace(0,100,1000); alpha = np.linspace(0.9,0.999,1000)
def emmv(sX,sU): return _em_auc(t,1.0,-np.asarray(sU,float),-np.asarray(sX,float),len(sU)), _mv_auc(alpha,1.0,-np.asarray(sU,float),-np.asarray(sX,float),len(sU))
def gos(k, dets):
    names=[d for d in dets if d in aucs.get(k,{}) and all(tt in aucs[k][d] for tt in TYPES)]
    if len(names)<3: return None
    b={d:0.0 for d in names}
    for tt in TYPES:
        for rk,d in enumerate(sorted(names,key=lambda x:aucs[k][x][tt])): b[d]+=rk
    return max(names,key=lambda d:b[d])
keys=[k for k in keep.index if k in clsap and k in clfix and k in pf and os.path.exists(os.path.join(CLS,f"sc_{k}.npz")) and os.path.exists(os.path.join(PF,f"sc_{k}.npz"))]
print(f"tasks: {len(keys)}")
POOLS={"classical":lambda k:[d for d in ({**clsap[k],**pf[k]['ap']}) if not DEEP(d) and ap_of(k,d) is not None],
       "deep":lambda k:[d for d in pf[k]['ap'] if DEEP(d) and ap_of(k,d) is not None],
       "all":lambda k:[d for d in ({**clsap[k],**pf[k]['ap']}) if ap_of(k,d) is not None]}
res={m:{} for m in ["SPARC","Goswami","Best fixed","Entropy(EM)","Mass-volume","Consensus","HITS","Model-centrality","Random"]}
for pn,poolf in POOLS.items():
    R={m:[] for m in res}
    for k in keys:
        dets=poolf(k); o=oracle[k]
        if len(dets)<3: continue
        val,uni=valuni(k); dv=[d for d in dets if d in val and d in uni]
        L=min(len(val[d]) for d in dv); Smat=np.column_stack([np.asarray(val[d],float)[:L] for d in dv])
        R["SPARC"].append(o-ap_of(k,max(dets,key=lambda d:a1_of(k,d))))
        gp=gos(k,dets); R["Goswami"].append(o-ap_of(k,gp) if gp else np.nan)
        R["Best fixed"].append(min(o-ap_of(k,d) for d in dets))  # per-task best (oracle within pool) -> use fixed below
        R["Consensus"].append(o-ap_of(k,dv[int(np.argmax(consensus_scores(Smat)))]))
        R["HITS"].append(o-ap_of(k,dv[int(np.argmax(hits_authority_scores(Smat)))]))
        R["Model-centrality"].append(o-ap_of(k,dv[int(np.argmax(model_centrality_scores(Smat)))]))
        ems=[emmv(val[d],uni[d]) for d in dv]
        R["Entropy(EM)"].append(o-ap_of(k,dv[int(np.argmax([e for e,_ in ems]))]))
        R["Mass-volume"].append(o-ap_of(k,dv[int(np.argmin([m for _,m in ems]))]))
        R["Random"].append(float(np.mean([o-ap_of(k,d) for d in dets])))
    # best FIXED detector within pool (min mean regret detector, not per-task)
    per={}
    for k in keys:
        for d in poolf(k): per.setdefault(d,[]).append(oracle[k]-ap_of(k,d))
    bf=min(per,key=lambda d:np.mean(per[d]))
    res["Best fixed"][pn]=np.mean(per[bf]); res["Best fixed"][pn+"_det"]=bf
    for m in res:
        if m!="Best fixed": res[m][pn]=np.nanmean(R[m])
T=pd.DataFrame({m:{p:res[m].get(p) for p in ["classical","deep","all"]} for m in res}).T
pd.set_option("display.float_format",lambda x:f"{x:.3f}")
print("\n=== mean regret by method x candidate pool (lower=better) ===")
print(T.to_string())
print(f"\nbest fixed detector per pool: classical={res['Best fixed'].get('classical_det')} deep={res['Best fixed'].get('deep_det')} all={res['Best fixed'].get('all_det')}")

# WHY deep is bad: probe-AUC (a1) vs real-ap, deep vs classical
print("\n=== why deep is bad: probe grade (a1) vs real detection (ap), averaged over tasks/detectors ===")
for grp,pred in [("classical",lambda d:not DEEP(d)),("deep",DEEP)]:
    a1s=[a1_of(k,d) for k in keys for d in (pf[k]['a1'] if grp=='deep' else clfix[k]['a1']) if pred(d) and a1_of(k,d) is not None and ap_of(k,d) is not None]
    aps=[ap_of(k,d) for k in keys for d in (pf[k]['ap'] if grp=='deep' else clsap[k]) if pred(d) and ap_of(k,d) is not None and a1_of(k,d) is not None]
    print(f"  {grp:9s}: mean probe-AUC a1={np.mean(a1s):.3f}  mean real ap_norm={np.mean(aps):.3f}  gap(a1-ap)={np.mean(a1s)-np.mean(aps):+.3f}")
