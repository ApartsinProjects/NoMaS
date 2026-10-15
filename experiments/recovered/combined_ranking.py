"""Combined ranking table: mean regret by method, classical-only (25) vs all-32, on the validated basis.
Pulls SPARC, Goswami, EM/MV/consensus/centrality/HITS (already computed) and adds best-fixed + random
per pool, computed consistently (oracle STREAM_V3; classical ap = clsstd; deep ap = probefix)."""
import json, glob, os, numpy as np, pandas as pd
S = r"E:/tmp/claude/E--Projects-Submitted-ADRank/236d6247-42ad-4e62-9828-db625dfa055d/scratchpad"
CLS = os.path.join(S, "clsstd_valuni"); PF = os.path.join(S, "probefix")
DEEP = lambda d: d.startswith("D_")
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
def ap_of(k, d): return deepap.get(k, {}).get(d) if DEEP(d) else clsap.get(k, {}).get(d)
keys = [k for k in keep.index if k in clsap and k in deepap]

# per-detector regret table (over the keys), to derive best-fixed and random per pool
allc = sorted(set.intersection(*[set([d for d in ({**clsap[k], **deepap[k]}) if ap_of(k, d) is not None]) for k in keys]))
per = {d: np.array([oracle[k] - ap_of(k, d) for k in keys]) for d in allc}
clsdets = [d for d in allc if not DEEP(d)]
def bestfixed(dets):
    md = min(dets, key=lambda d: per[d].mean()); return per[md].mean(), md
def rnd(dets): return np.mean([per[d].mean() for d in dets])

# selector means, full 151-task validated (SPARC from reconcile_sparc_all; Goswami from goswami_all_post)
SEL = {
 "SPARC":        {"cls": 0.1131, "all": 0.1617},
 "Goswami":      {"cls": 0.1216, "all": 0.1510},
}
uoms = pd.read_csv(os.path.join(S, "RECONCILE_UOMS_V2.csv"))
for m in ["EM", "MV", "consensus", "centrality", "hits"]:
    SEL[{"EM":"Entropy (EM)","MV":"Mass-volume (MV)","consensus":"Consensus","centrality":"Model-centrality","hits":"HITS"}[m]] = \
        {"cls": uoms[f"{m}_cls"].mean(), "all": uoms[f"{m}_all"].mean()}
bf_c, bfc_d = bestfixed(clsdets); bf_a, bfa_d = bestfixed(allc)
SEL["Best fixed detector"] = {"cls": bf_c, "all": bf_a}
SEL["Random pick"] = {"cls": rnd(clsdets), "all": rnd(allc)}

print(f"tasks: {len(keys)}   best-fixed classical={bfc_d}  all-32={bfa_d}\n")
tab = pd.DataFrame([{"method": m, "classical_only_25": v["cls"], "all_32": v["all"]} for m, v in SEL.items()])
tab = tab.sort_values("classical_only_25").reset_index(drop=True); tab.index += 1
pd.set_option("display.float_format", lambda x: f"{x:.3f}")
print("=== COMBINED RANKING: mean regret by method (lower = better), ranked by classical-only ===")
print(tab.to_string())
tab.to_csv(os.path.join(S, "COMBINED_RANKING.csv"))
