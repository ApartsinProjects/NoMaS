"""Regret by method x data type, classical-only (25-detector) matched pool. Per-task regret for every
method, grouped by source corpus. oracle STREAM_V3; classical ap=clsstd (=clstd); a1=clfix; Goswami
from goswami2 aucs (classical Borda)."""
import json, glob, os, numpy as np, pandas as pd
S = r"E:/tmp/claude/E--Projects-Submitted-ADRank/236d6247-42ad-4e62-9828-db625dfa055d/scratchpad"
CLS = os.path.join(S, "clsstd_valuni"); G = os.path.join(S, "goswami2")
DEEP = lambda d: d.startswith("D_")
clsap = {json.loads(l)["key"]: json.loads(l)["ap"] for l in open(os.path.join(CLS, "ap.jsonl"))}
def loaddir(d):
    J = {}
    for fn in glob.glob(os.path.join(d, "shard_*.jsonl")):
        for ln in open(fn):
            try: r = json.loads(ln)
            except Exception: continue
            if "key" in r: J[r["key"]] = r
    return J
clfix = loaddir(os.path.join(S, "streamline", "deepres_v2_clfix"))
lb = pd.read_csv(os.path.join(S, "streamline", "STREAM_V3_LEADERBOARD.csv"))
keep = lb[(lb.oracle_ap > 0.05) & (lb.reg_random > 0.02) & (lb.n_h >= 100)].set_index("key")
oracle = keep.oracle_ap.to_dict(); corpus = keep.corpus.to_dict()
aucs = {}
for f in glob.glob(os.path.join(G, "*.jsonl")):
    for ln in open(f):
        try: r = json.loads(ln)
        except Exception: continue
        if r.get("aucs"): aucs[r["key"]] = r["aucs"]
uoms = pd.read_csv(os.path.join(S, "RECONCILE_UOMS_V2.csv")).set_index("key")
TYPES = ["spike", "noise", "scale", "contextual", "cutoff"]

def gos_cls(k):
    names = [d for d in aucs.get(k, {}) if not DEEP(d) and all(t in aucs[k][d] for t in TYPES)]
    if len(names) < 6: return None
    b = {d: 0.0 for d in names}
    for t in TYPES:
        for rk, d in enumerate(sorted(names, key=lambda x: aucs[k][x][t])): b[d] += rk
    return max(names, key=lambda d: b[d])

keys = [k for k in keep.index if k in clsap and k in clfix and k in uoms.index]
rows = []
for k in keys:
    o = oracle[k]; ap = clsap[k]
    rec = {"key": k, "corpus": corpus[k]}
    a1 = {d: v for d, v in clfix[k]["a1"].items() if not DEEP(d) and d in ap}
    rec["SPARC"] = o - ap[max(a1, key=a1.get)]
    gp = gos_cls(k); rec["Goswami"] = (o - ap[gp]) if gp and gp in ap else np.nan
    rec["Best fixed (LOF k=10)"] = o - ap.get("LOF_k10", np.nan)
    rec["Mass-volume"] = uoms.loc[k, "MV_cls"]; rec["Entropy (EM)"] = uoms.loc[k, "EM_cls"]
    rec["Consensus"] = uoms.loc[k, "consensus_cls"]; rec["HITS"] = uoms.loc[k, "hits_cls"]
    rec["Model-centrality"] = uoms.loc[k, "centrality_cls"]
    rec["Random"] = float(np.mean([o - ap[d] for d in ap]))
    rows.append(rec)
df = pd.DataFrame(rows)
CMAP = {"ovrbench": "OvrBench", "oddbench": "OddBench", "tsbad_m": "TSB-AD (TS)", "adbench": "ADBench", "dami": "DAMI"}
df["Source"] = df.corpus.map(CMAP)
methods = ["SPARC", "Goswami", "Best fixed (LOF k=10)", "Mass-volume", "Entropy (EM)", "Consensus", "HITS", "Model-centrality", "Random"]
order = ["OvrBench", "OddBench", "TSB-AD (TS)", "ADBench", "DAMI"]
n = df.groupby("Source").size()
out = {}
for m in methods:
    row = {src: df[df.Source == src][m].mean() for src in order}
    row["All"] = df[m].mean(); out[m] = row
T = pd.DataFrame(out).T[order + ["All"]]
print("=== Regret by method x data type (CLASSICAL-ONLY pool, lower=better) ===")
print("tasks per source:", {CMAP[c]: int(n.get(CMAP[c], 0)) for c in CMAP})
pd.set_option("display.float_format", lambda x: f"{x:.3f}")
print(T.to_string())
T.to_csv(os.path.join(S, "BY_TYPE_classical.csv"))
