"""Matched-pool evaluation: every selection method over the SAME detector pool.
Reads the fixed-probe artifact (probefix shards: ap,a1 per detector; sc_*.npz: val__/uni__ per detector).
For each method, compute mean regret selecting from (a) all 32 detectors and (b) classical-only.
Oracle = max ap over ALL 32 (fixed zero of regret). Nothing folded into the paper; prints + saves CSV.
"""
import json, glob, sys, numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
sys.path.insert(0, r"E:/Projects/Submitted/zenodo_build/code/lib")
from label_free_criteria import (consensus_scores, model_centrality_scores,
                                 hits_authority_scores, _em_auc, _mv_auc)

ART = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"E:/tmp/claude/E--Projects-Submitted-ADRank/236d6247-42ad-4e62-9828-db625dfa055d/scratchpad/probefix")
FINAL = pd.read_csv(r"E:/Projects/Submitted/zenodo_build/results/model_selection_per_dataset.csv")
final_keys = set(FINAL["dataset"])
DEEP = lambda d: d.startswith("D_")

# --- load ap + a1 from shards ---
meta = {}
for f in glob.glob(str(ART/"*.jsonl")):
    for ln in open(f):
        try: r = json.loads(ln)
        except Exception: continue
        if r.get("ap") and r.get("a1"): meta[r["key"]] = r
print(f"tasks with ap+a1: {len(meta)}")

def em_mv_from_scores(sX_raw, sU_raw, n_gen):
    # match em_mv_scores convention: normality = -anomaly-score
    s_X = -np.asarray(sX_raw, float); s_U = -np.asarray(sU_raw, float)
    t = np.linspace(0, 100, 1000); alpha = np.linspace(0.9, 0.999, 1000)
    return _em_auc(t, 1.0, s_U, s_X, n_gen), _mv_auc(alpha, 1.0, s_U, s_X, n_gen)

def pick_regret(key, dets, ap, oracle, sel_index_fn):
    """sel_index_fn(dets) -> chosen detector name; regret vs fixed oracle."""
    sel = sel_index_fn(dets)
    return oracle - ap[sel], sel

recs = []
skipped = 0
for key in meta:
    if key not in final_keys: continue
    r = meta[key]; ap = r["ap"]; a1 = r["a1"]
    scf = ART/f"sc_{key}.npz"
    if not scf.exists(): skipped += 1; continue
    z = np.load(scf)
    val = {k[5:]: z[k] for k in z.files if k.startswith("val__")}
    uni = {k[5:]: z[k] for k in z.files if k.startswith("uni__")}
    # common detector set present everywhere needed
    allc = [d for d in ap if d in a1 and d in val and d in uni]
    if len(allc) < 5: skipped += 1; continue
    oracle = max(ap[d] for d in allc)             # oracle over the observed pool (all detectors present)
    pools = {"all": allc, "cls": [d for d in allc if not DEEP(d)]}
    row = {"key": key, "corpus": key.split("__")[0], "oracle": oracle, "n_all": len(allc), "n_cls": len(pools['cls'])}
    for pname, dets in pools.items():
        if not dets: continue
        # score matrix on validation normals (rows=points, cols=dets), aligned length
        L = min(len(val[d]) for d in dets)
        S = np.column_stack([np.asarray(val[d], float)[:L] for d in dets])
        # SPARC: argmax probe grade a1
        sel = max(dets, key=lambda d: a1[d]); row[f"reg_SPARC_{pname}"] = oracle - ap[sel]; row[f"sel_SPARC_{pname}"] = sel
        # consensus / centrality / HITS: higher = better detector
        for cname, fn in [("consensus", consensus_scores), ("centrality", model_centrality_scores), ("hits", hits_authority_scores)]:
            sc = fn(S); sel = dets[int(np.argmax(sc))]; row[f"reg_{cname}_{pname}"] = oracle - ap[sel]
        # EM (higher=better) / MV (lower=better)
        ems, mvs = [], []
        for d in dets:
            ng = min(len(uni[d]), len(val[d])) if False else len(uni[d])
            e, m = em_mv_from_scores(val[d], uni[d], len(uni[d])); ems.append(e); mvs.append(m)
        row[f"reg_EM_{pname}"] = oracle - ap[dets[int(np.argmax(ems))]]
        row[f"reg_MV_{pname}"] = oracle - ap[dets[int(np.argmin(mvs))]]
        # random = mean regret over pool
        row[f"reg_random_{pname}"] = float(np.mean([oracle - ap[d] for d in dets]))
    recs.append(row)

df = pd.DataFrame(recs)
print(f"tasks scored: {len(df)}  (skipped {skipped})")
OUT = ART.parent/"MATCHED_POOL.csv"; df.to_csv(OUT, index=False)

# best-fixed detector per pool: min mean regret over tasks
methods = ["SPARC","consensus","centrality","hits","EM","MV","random"]
def bestfixed(pname):
    dets = set.intersection(*[set(json.loads(open(f).readline())) for f in []]) if False else None
    # min mean regret detector, over detectors present in ALL scored tasks for this pool
    common = None
    perdet = {}
    for key in df.key:
        r = meta[key]; ap = r["ap"]
        scf = ART/f"sc_{key}.npz"; z = np.load(scf)
        allc = [d for d in ap if d in r["a1"]]
        pool = allc if pname=="all" else [d for d in allc if not DEEP(d)]
        oracle = max(ap[d] for d in pool)
        for d in pool: perdet.setdefault(d, []).append(oracle - ap[d])
    # only detectors present on all tasks
    ntask = len(df)
    cand = {d:np.mean(v) for d,v in perdet.items() if len(v)>=0.9*ntask}
    bd = min(cand, key=cand.get)
    return bd, cand[bd]

print("\n=== MATCHED-POOL mean regret (lower better); paired Wilcoxon all-vs-cls ===")
print(f"{'method':<12} {'all-32':>8} {'classical':>10} {'delta':>8} {'p(all vs cls)':>14}")
summary = []
for m in methods:
    a = df[f"reg_{m}_all"]; c = df[f"reg_{m}_cls"]
    p = stats.wilcoxon(a, c).pvalue if (a!=c).any() else float("nan")
    print(f"{m:<12} {a.mean():>8.4f} {c.mean():>10.4f} {a.mean()-c.mean():>+8.4f} {p:>14.4f}")
    summary.append(dict(method=m, all32=a.mean(), classical=c.mean(), delta=a.mean()-c.mean(), p_all_vs_cls=p))
for pn in ["all","cls"]:
    bd, br = bestfixed(pn)
    print(f"best-fixed ({pn}): {bd}  mean regret {br:.4f}")
pd.DataFrame(summary).to_csv(ART.parent/"MATCHED_POOL_SUMMARY.csv", index=False)
print(f"\nsaved {OUT} and MATCHED_POOL_SUMMARY.csv")
