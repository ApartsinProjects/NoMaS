"""Diagnostic: is the consensus all-32 gap (my 0.25 vs Fable/published 0.229) explained by the
classical val basis (raw-fit probefix vs standardized clsstd)? Recompute consensus all-32 both ways
on the same tasks. If raw-basis ~ 0.229 (published) and std-basis ~ 0.25, the published agreement
criteria used the pre-fix raw-classical val matrix and the std recompute is the correction."""
import json, glob, os, sys, numpy as np, pandas as pd
sys.path.insert(0, r"E:/Projects/Submitted/zenodo_build/code/lib")
from label_free_criteria import consensus_scores
S = r"E:/tmp/claude/E--Projects-Submitted-ADRank/236d6247-42ad-4e62-9828-db625dfa055d/scratchpad"
CLS = os.path.join(S, "clsstd_valuni"); PF = os.path.join(S, "probefix")
DEEP = lambda d: d.startswith("D_")
clsap = {json.loads(l)["key"]: json.loads(l)["ap"] for l in open(os.path.join(CLS, "ap.jsonl"))}
deepap = {}
for f in glob.glob(os.path.join(PF, "*.jsonl")):
    for ln in open(f):
        try: r = json.loads(ln)
        except Exception: continue
        if r.get("ap"): deepap[r["key"]] = r["ap"]      # probefix ap: classical(raw) + deep
lb = pd.read_csv(os.path.join(S, "streamline", "STREAM_V3_LEADERBOARD.csv"))
keep = lb[(lb.oracle_ap > 0.05) & (lb.reg_random > 0.02) & (lb.n_h >= 100)].set_index("key")
oracle = keep.oracle_ap.to_dict()

def valdict(path):
    z = np.load(path); return {k[5:]: z[k] for k in z.files if k.startswith("val__")}

def consensus_regret(keys, get_val, ap_of):
    regs = []
    for k in keys:
        val = get_val(k); dets = [d for d in val if ap_of(k, d) is not None]
        L = min(len(val[d]) for d in dets)
        Smat = np.column_stack([np.asarray(val[d], float)[:L] for d in dets])
        pick = dets[int(np.argmax(consensus_scores(Smat)))]
        regs.append(oracle[k] - ap_of(k, pick))
    return np.mean(regs), len(regs)

keys = [k for k in keep.index if os.path.exists(os.path.join(CLS, f"sc_{k}.npz")) and os.path.exists(os.path.join(PF, f"sc_{k}.npz")) and k in deepap]
print(f"tasks: {len(keys)}")

# (a) RAW basis: probefix sc for all 32 (raw classical + deep), ap from probefix
def ap_pf(k, d): return deepap[k].get(d)
ra, na = consensus_regret(keys, lambda k: valdict(os.path.join(PF, f"sc_{k}.npz")), ap_pf)
print(f"(a) consensus all-32, RAW classical val (probefix), probefix ap : {ra:.4f}  (published/Fable ~0.229)")

# (b) STD basis: clsstd classical val + probefix deep val, ap std-classical + probefix deep
def val_std(k):
    v = valdict(os.path.join(CLS, f"sc_{k}.npz"))
    vd = valdict(os.path.join(PF, f"sc_{k}.npz"))
    v.update({d: vd[d] for d in vd if DEEP(d)}); return v
def ap_std(k, d): return deepap[k].get(d) if DEEP(d) else clsap[k].get(d)
rb, nb = consensus_regret(keys, val_std, ap_std)
print(f"(b) consensus all-32, STD classical val (clsstd)+deep         : {rb:.4f}  (my v2 ~0.25)")

# (c) which detector does consensus pick under each basis? deep vs classical share
def pickshare(get_val, ap_of):
    deep=0
    for k in keys:
        val=get_val(k); dets=[d for d in val if ap_of(k,d) is not None]
        L=min(len(val[d]) for d in dets); Smat=np.column_stack([np.asarray(val[d],float)[:L] for d in dets])
        if DEEP(dets[int(np.argmax(consensus_scores(Smat)))]): deep+=1
    return deep
print(f"deep-picked by consensus: RAW basis {pickshare(lambda k: valdict(os.path.join(PF,f'sc_{k}.npz')), ap_pf)}/{len(keys)}  STD basis {pickshare(val_std, ap_std)}/{len(keys)}")
