"""Recompute the internal/agreement criteria (EM, MV, consensus, centrality, HITS) over all-32 vs
classical-only on the PUBLISHED basis. Pick from val/uni scores (probefix sc_*.npz), regret vs
STREAM_V3 oracle with ap from clstd (classical) / STREAM_DEEP_MODAL (deep). Classical-only recompute
should reproduce published reg_em/reg_mv/etc. Prints per-method all vs cls + published cross-check."""
import json, glob, os, sys, numpy as np, pandas as pd
from scipy.stats import wilcoxon
sys.path.insert(0, r"E:/Projects/Submitted/zenodo_build/code/lib")
from label_free_criteria import (consensus_scores, model_centrality_scores,
                                 hits_authority_scores, _em_auc, _mv_auc)
S = r"E:/tmp/claude/E--Projects-Submitted-ADRank/236d6247-42ad-4e62-9828-db625dfa055d/scratchpad/streamline"
PF = r"E:/tmp/claude/E--Projects-Submitted-ADRank/236d6247-42ad-4e62-9828-db625dfa055d/scratchpad/probefix"
DEEP = lambda d: d.startswith("D_")

def load_dir(d):
    J = {}
    for fn in glob.glob(os.path.join(d, "shard_*.jsonl")):
        for ln in open(fn):
            try: r = json.loads(ln)
            except Exception: continue
            if "key" in r: J[r["key"]] = r
    return J

clstd = load_dir(os.path.join(S, "deepres_v2_clstd"))
lb = pd.read_csv(os.path.join(S, "STREAM_V3_LEADERBOARD.csv"))
keep = lb[(lb.oracle_ap > 0.05) & (lb.reg_random > 0.02) & (lb.n_h >= 100)].set_index("key")
oracle = keep.oracle_ap.to_dict()
pub = keep[["reg_em", "reg_mv", "reg_consensus", "reg_mc", "reg_hits", "reg_random"]]
modal = pd.read_csv(os.path.join(S, "STREAM_DEEP_MODAL.csv")).set_index("key")
deepcols = [c for c in modal.columns if c.startswith("D_")]

def ap_of(k, det):
    if DEEP(det):
        v = modal.loc[k, det] if (k in modal.index and det in modal.columns) else np.nan
        return float(v) if pd.notna(v) else None
    return clstd[k]["ap"].get(det) if k in clstd else None

t = np.linspace(0, 100, 1000); alpha = np.linspace(0.9, 0.999, 1000)
def em_mv(sX_raw, sU_raw):
    s_X = -np.asarray(sX_raw, float); s_U = -np.asarray(sU_raw, float); ng = len(s_U)
    return _em_auc(t, 1.0, s_U, s_X, ng), _mv_auc(alpha, 1.0, s_U, s_X, ng)

rows = []
keys = [k for k in keep.index if os.path.exists(os.path.join(PF, f"sc_{k}.npz")) and k in clstd and k in modal.index]
print(f"tasks with sc_ npz + published basis: {len(keys)} / {len(keep)}")
for k in keys:
    z = np.load(os.path.join(PF, f"sc_{k}.npz"))
    val = {kk[5:]: z[kk] for kk in z.files if kk.startswith("val__")}
    uni = {kk[5:]: z[kk] for kk in z.files if kk.startswith("uni__")}
    o = oracle[k]
    row = {"key": k}
    for pname, isdeep_ok in [("all", True), ("cls", False)]:
        dets = [d for d in val if d in uni and (isdeep_ok or not DEEP(d)) and ap_of(k, d) is not None]
        if len(dets) < 3: continue
        L = min(len(val[d]) for d in dets)
        Smat = np.column_stack([np.asarray(val[d], float)[:L] for d in dets])
        for cname, fn in [("consensus", consensus_scores), ("centrality", model_centrality_scores), ("hits", hits_authority_scores)]:
            sc = fn(Smat); pick = dets[int(np.argmax(sc))]; row[f"{cname}_{pname}"] = o - ap_of(k, pick)
        ems, mvs = [], []
        for d in dets:
            e, m = em_mv(val[d], uni[d]); ems.append(e); mvs.append(m)
        row[f"EM_{pname}"] = o - ap_of(k, dets[int(np.argmax(ems))])
        row[f"MV_{pname}"] = o - ap_of(k, dets[int(np.argmin(mvs))])
    rows.append(row)

df = pd.DataFrame(rows).dropna()
print(f"tasks scored (both pools): {len(df)}")
pubmean = {"EM": pub.reg_em.loc[df.key].mean(), "MV": pub.reg_mv.loc[df.key].mean(),
           "consensus": pub.reg_consensus.loc[df.key].mean(), "centrality": pub.reg_mc.loc[df.key].mean(),
           "hits": pub.reg_hits.loc[df.key].mean()}
print(f"\n{'method':<11} {'all-32':>8} {'classical':>10} {'delta':>8} {'p(all-cls)':>11} {'published':>10}")
for m in ["EM", "MV", "consensus", "centrality", "hits"]:
    a, c = df[f"{m}_all"], df[f"{m}_cls"]
    p = wilcoxon(a, c).pvalue if (a != c).any() else float("nan")
    print(f"{m:<11} {a.mean():>8.4f} {c.mean():>10.4f} {a.mean()-c.mean():>+8.4f} {p:>11.4f} {pubmean[m]:>10.4f}")
df.to_csv(os.path.join(S, "..", "RECONCILE_UOMS.csv"), index=False)
print("\n(classical column should track the published column; that validates the recompute)")
