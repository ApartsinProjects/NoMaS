"""Reconciled SPARC all-32 vs classical-only, on the PUBLISHED basis (must reproduce SPARC-cls=0.113).
oracle = STREAM_V3 oracle_ap; classical ap=clstd, a1=clfix (fixed probe); deep ap=STREAM_DEEP_MODAL,
deep a1 = probefix (fixed probe). SPARC = argmax a1 over the pool; regret = oracle - ap[selected]."""
import json, glob, os, numpy as np, pandas as pd
from scipy.stats import wilcoxon
S = r"E:/tmp/claude/E--Projects-Submitted-ADRank/236d6247-42ad-4e62-9828-db625dfa055d/scratchpad/streamline"
PF = r"E:/tmp/claude/E--Projects-Submitted-ADRank/236d6247-42ad-4e62-9828-db625dfa055d/scratchpad/probefix"

def load_dir(d):
    J = {}
    for fn in glob.glob(os.path.join(d, "shard_*.jsonl")):
        for ln in open(fn):
            try: r = json.loads(ln)
            except Exception: continue
            if "key" in r: J[r["key"]] = r
    return J

clstd = load_dir(os.path.join(S, "deepres_v2_clstd"))   # classical ap
clfix = load_dir(os.path.join(S, "deepres_v2_clfix"))   # classical a1 (fixed probe)
lb = pd.read_csv(os.path.join(S, "STREAM_V3_LEADERBOARD.csv"))
keep = lb[(lb.oracle_ap > 0.05) & (lb.reg_random > 0.02) & (lb.n_h >= 100)].set_index("key")
oracle = keep.oracle_ap.to_dict()
modal = pd.read_csv(os.path.join(S, "STREAM_DEEP_MODAL.csv")).set_index("key")
deepcols = [c for c in modal.columns if c.startswith("D_")]

pf = {}
for f in glob.glob(os.path.join(PF, "*.jsonl")):
    for ln in open(f):
        try: r = json.loads(ln)
        except Exception: continue
        if r.get("a1"): pf[r["key"]] = r          # FIX (Fable #1): keep full record for deep ap too
print(f"probefix deep-a1 tasks: {len(pf)}")

keys = [k for k in keep.index if k in clstd and k in clfix and k in modal.index]
keys_pf = [k for k in keys if k in pf]
print(f"eligible keys: {len(keys)} ; with probefix deep a1: {len(keys_pf)}")

def ap_of(k, det):
    # FIX (Fable #1): deep ap comes from probefix (= deepres_v2, the oracle's own basis), NOT STREAM_DEEP_MODAL
    if det.startswith("D_"):
        v = pf[k]["ap"].get(det); return float(v) if v is not None else None
    return clstd[k]["ap"].get(det)

reg_cls, reg_all, deeppick = [], [], 0
for k in keys_pf:
    o = oracle[k]
    cls_a1 = clfix[k]["a1"]
    sel_c = max(cls_a1, key=cls_a1.get); reg_cls.append(o - clstd[k]["ap"][sel_c])
    deep_a1 = {d: pf[k]["a1"][d] for d in deepcols if d in pf[k]["a1"]}
    alla1 = {**cls_a1, **deep_a1}
    sel_a = max(alla1, key=alla1.get); apa = ap_of(k, sel_a)
    if apa is None: apa = clstd[k]["ap"][sel_c]        # deep ap missing -> fall back (rare)
    reg_all.append(o - apa)
    if sel_a.startswith("D_"): deeppick += 1

reg_cls, reg_all = np.array(reg_cls), np.array(reg_all)
print(f"\n=== SPARC on published basis ({len(keys_pf)} tasks) ===")
print(f"SPARC classical-only  mean regret = {reg_cls.mean():.4f}   (published = 0.113)")
print(f"SPARC all-32          mean regret = {reg_all.mean():.4f}")
print(f"delta (all - cls)     = {reg_all.mean()-reg_cls.mean():+.4f}")
print(f"deep picked by all-32 : {deeppick}/{len(keys_pf)} ({100*deeppick/len(keys_pf):.0f}%)")
if (reg_all != reg_cls).any():
    print(f"Wilcoxon all vs cls   : p={wilcoxon(reg_all, reg_cls).pvalue:.4f}")
print(f"vs best fixed detector 0.130: all-32 {'beats' if reg_all.mean()<0.130 else 'does NOT beat'} it")
