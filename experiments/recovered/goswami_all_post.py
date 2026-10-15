"""Goswami all-32 vs classical-only: Borda over injection types, on the published basis.
aucs from the goswami2 Modal run (all 32 detectors). ap: clsstd (classical) + probefix (deep).
oracle from STREAM_V3. Classical-only must reproduce published Goswami ~0.122 (cross-check vs GOSWAMI_INJ)."""
import json, glob, os, numpy as np, pandas as pd
from scipy.stats import wilcoxon
S = r"E:/tmp/claude/E--Projects-Submitted-ADRank/236d6247-42ad-4e62-9828-db625dfa055d/scratchpad"
G = os.path.join(S, "goswami2"); CLS = os.path.join(S, "clsstd_valuni"); PF = os.path.join(S, "probefix")
DEEP = lambda d: d.startswith("D_")

aucs = {}
for f in glob.glob(os.path.join(G, "*.jsonl")):
    for ln in open(f):
        try: r = json.loads(ln)
        except Exception: continue
        if r.get("aucs"): aucs[r["key"]] = r["aucs"]
clsap = {json.loads(l)["key"]: json.loads(l)["ap"] for l in open(os.path.join(CLS, "ap.jsonl"))}
deepap = {}
for f in glob.glob(os.path.join(PF, "*.jsonl")):
    for ln in open(f):
        try: r = json.loads(ln)
        except Exception: continue
        if r.get("ap"): deepap[r["key"]] = {d: v for d, v in r["ap"].items() if DEEP(d)}
lb = pd.read_csv(os.path.join(S, "streamline", "STREAM_V3_LEADERBOARD.csv"))
keep = lb[(lb.oracle_ap > 0.05) & (lb.reg_random > 0.02) & (lb.n_h >= 100)].set_index("key")
oracle = keep.oracle_ap.to_dict()
# published classical Goswami picks (for agreement cross-check)
gpub = {}
gp = os.path.join(S, "streamline", "GOSWAMI_INJ.jsonl")
if os.path.exists(gp):
    for ln in open(gp):
        try: r = json.loads(ln)
        except Exception: continue
        if r.get("sel"): gpub[r["key"]] = r["sel"]

def ap_of(k, d): return deepap.get(k, {}).get(d) if DEEP(d) else clsap.get(k, {}).get(d)
TYPES = ["spike", "noise", "scale", "contextual", "cutoff"]

def borda_pick(k, dets):
    names = [d for d in dets if d in aucs[k] and all(t in aucs[k][d] for t in TYPES)]
    if len(names) < 6: return None
    borda = {d: 0.0 for d in names}
    for t in TYPES:
        for rank, d in enumerate(sorted(names, key=lambda x: aucs[k][x][t])): borda[d] += rank
    return max(names, key=lambda d: borda[d])

keys = [k for k in keep.index if k in aucs and k in deepap and k in clsap]
reg_all, reg_cls, agree, deeppick = [], [], 0, 0
for k in keys:
    o = oracle[k]
    alld = [d for d in aucs[k] if ap_of(k, d) is not None]
    clsd = [d for d in alld if not DEEP(d)]
    pa, pc = borda_pick(k, alld), borda_pick(k, clsd)
    if pa is None or pc is None: continue
    reg_all.append(o - ap_of(k, pa)); reg_cls.append(o - ap_of(k, pc))
    if DEEP(pa): deeppick += 1
    if k in gpub and pc == gpub[k]: agree += 1
reg_all, reg_cls = np.array(reg_all), np.array(reg_cls)
print(f"tasks: {len(reg_all)}")
print(f"Goswami classical-only mean regret = {reg_cls.mean():.4f}   (published ~0.122)")
print(f"Goswami all-32         mean regret = {reg_all.mean():.4f}")
print(f"delta (all-cls) = {reg_all.mean()-reg_cls.mean():+.4f}  p={wilcoxon(reg_all,reg_cls).pvalue:.4f}  deep picked {deeppick}/{len(reg_all)}")
print(f"classical-pick agreement vs published GOSWAMI_INJ: {agree}/{sum(1 for k in keys if k in gpub)} ({'n/a' if not gpub else ''})")
print(f"vs best fixed 0.130: classical {'beats' if reg_cls.mean()<0.130 else 'no'}, all-32 {'beats' if reg_all.mean()<0.130 else 'no'}")
