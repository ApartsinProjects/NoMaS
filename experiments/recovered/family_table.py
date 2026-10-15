"""Family table on the PUBLISHED SPARC basis (reproduces 0.113).
Per task over ALL 32 detectors: which FAMILY does the probe rank #1 (a1 argmax) vs which family
is the real oracle (ap argmax)? Shows the probe's systematic bias: over-ranks deep, blind to marginal,
accurate on neighbor. ap: classical=deepres_v2_clstd, deep=probefix; a1: classical=deepres_v2_clfix,
deep=probefix. Same sources as reconcile_sparc_all.py."""
import json, glob, os, numpy as np, pandas as pd
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
clstd = load_dir(os.path.join(S, "deepres_v2_clstd")); clfix = load_dir(os.path.join(S, "deepres_v2_clfix"))
pf = {}
for f in glob.glob(os.path.join(PF, "*.jsonl")):
    for ln in open(f):
        try: r = json.loads(ln)
        except Exception: continue
        if r.get("a1"): pf[r["key"]] = r
lb = pd.read_csv(os.path.join(S, "STREAM_V3_LEADERBOARD.csv"))
keep = lb[(lb.oracle_ap > 0.05) & (lb.reg_random > 0.02) & (lb.n_h >= 100)].set_index("key")
keys = [k for k in keep.index if k in clstd and k in clfix and k in pf]
print(f"tasks: {len(keys)}")
DEEP = lambda d: d.startswith("D_")
def fam(d):
    # grouping matches paper Table 5 (CBLOF with neighbor/density, LODA with projection)
    if DEEP(d): return "deep"
    if d.startswith(("LOF", "KNN", "CBLOF")): return "neighbor"
    if d.startswith(("HBOS", "ECOD", "COPOD")): return "distribution"
    if d.startswith(("PCA", "LODA")): return "projection"
    if d.startswith("IForest"): return "isolation"
    return "other"
FAMS = ["neighbor", "distribution", "projection", "isolation", "deep"]
def ap_all(k):
    a = {d: v for d, v in clstd[k]["ap"].items() if not DEEP(d)}
    a.update({d: v for d, v in pf[k]["ap"].items() if DEEP(d)})
    return a
def a1_all(k):
    a = {d: v for d, v in clfix[k]["a1"].items() if not DEEP(d)}
    a.update({d: v for d, v in pf[k]["a1"].items() if DEEP(d)})
    return a
probe_top = {f: 0 for f in FAMS}; real_win = {f: 0 for f in FAMS}
# accuracy: of tasks where the probe's #1 is family F, how often is the real oracle also in F?
hit = {f: 0 for f in FAMS}
for k in keys:
    ap = ap_all(k); a1 = a1_all(k)
    dets = [d for d in ap if d in a1]
    pt = max(dets, key=lambda d: a1[d]); rw = max(dets, key=lambda d: ap[d])
    probe_top[fam(pt)] += 1; real_win[fam(rw)] += 1
    if fam(pt) == fam(rw): hit[fam(pt)] += 1
print("\n=== family table: probe's #1 pick vs the real oracle, over all 32 detectors, 151 tasks ===")
print(f"{'family':11s} {'probe ranks #1':>15s} {'real oracle':>12s} {'probe-right':>12s}")
for f in FAMS:
    acc = f"{hit[f]}/{probe_top[f]}" if probe_top[f] else "-"
    print(f"{f:11s} {probe_top[f]:>15d} {real_win[f]:>12d} {acc:>12s}")
print(f"{'TOTAL':11s} {sum(probe_top.values()):>15d} {sum(real_win.values()):>12d}")
