"""UOMS recompute over all-32 vs classical-only on the VALIDATED standardized basis.
Classical val/uni/ap: clsstd_valuni (bit-identical to published clstd). Deep val/uni/ap: probefix
(standardized, deterministic = deepres_v2). Same uniform sample across both. Oracle from STREAM_V3;
classical ap from clsstd, deep ap from probefix. Classical-only must reproduce published reg_em/reg_mv.
"""
import json, glob, os, sys, numpy as np, pandas as pd
from scipy.stats import wilcoxon
sys.path.insert(0, r"E:/Projects/Submitted/zenodo_build/code/lib")
from label_free_criteria import (consensus_scores, model_centrality_scores,
                                 hits_authority_scores, _em_auc, _mv_auc)
S = r"E:/tmp/claude/E--Projects-Submitted-ADRank/236d6247-42ad-4e62-9828-db625dfa055d/scratchpad"
CLS = os.path.join(S, "clsstd_valuni")           # standardized classical sc_ + ap.jsonl
PF = os.path.join(S, "probefix")                  # standardized deep sc_ + shards (deep ap)
DEEP = lambda d: d.startswith("D_")

# classical ap (= published clstd)
clsap = {}
for ln in open(os.path.join(CLS, "ap.jsonl")):
    r = json.loads(ln); clsap[r["key"]] = r["ap"]
# deep ap from probefix shards
deepap = {}
for f in glob.glob(os.path.join(PF, "*.jsonl")):
    for ln in open(f):
        try: r = json.loads(ln)
        except Exception: continue
        if r.get("ap"): deepap[r["key"]] = {d: v for d, v in r["ap"].items() if DEEP(d)}
lb = pd.read_csv(os.path.join(S, "streamline", "STREAM_V3_LEADERBOARD.csv"))
keep = lb[(lb.oracle_ap > 0.05) & (lb.reg_random > 0.02) & (lb.n_h >= 100)].set_index("key")
oracle = keep.oracle_ap.to_dict()
pub = keep[["reg_em", "reg_mv", "reg_consensus", "reg_mc", "reg_hits"]]

def ap_of(k, det):
    return deepap.get(k, {}).get(det) if DEEP(det) else clsap.get(k, {}).get(det)

t = np.linspace(0, 100, 1000); alpha = np.linspace(0.9, 0.999, 1000)
def em_mv(sX_raw, sU_raw):
    s_X = -np.asarray(sX_raw, float); s_U = -np.asarray(sU_raw, float); ng = len(s_U)
    return _em_auc(t, 1.0, s_U, s_X, ng), _mv_auc(alpha, 1.0, s_U, s_X, ng)

def load_valuni(k):
    # FIX: classical val/uni ONLY from CLS (standardized); DEEP val/uni ONLY from PF.
    # probefix sc also carries RAW-fit classical columns -- must NOT let them overwrite the std ones.
    val, uni = {}, {}
    zc = np.load(os.path.join(CLS, f"sc_{k}.npz"))
    for kk in zc.files:
        if kk.startswith("val__"): val[kk[5:]] = zc[kk]
        elif kk.startswith("uni__"): uni[kk[5:]] = zc[kk]
    zp = np.load(os.path.join(PF, f"sc_{k}.npz"))
    for kk in zp.files:
        det = kk[5:]
        if not DEEP(det): continue          # keep only deep from probefix
        if kk.startswith("val__"): val[det] = zp[kk]
        elif kk.startswith("uni__"): uni[det] = zp[kk]
    return val, uni

rows = []
keys = [k for k in keep.index if os.path.exists(os.path.join(CLS, f"sc_{k}.npz")) and os.path.exists(os.path.join(PF, f"sc_{k}.npz")) and k in deepap]
print(f"tasks (classical+deep sc_ + published oracle): {len(keys)} / {len(keep)}")
for k in keys:
    val, uni = load_valuni(k); o = oracle[k]
    row = {"key": k}
    for pname, deep_ok in [("all", True), ("cls", False)]:
        dets = [d for d in val if d in uni and (deep_ok or not DEEP(d)) and ap_of(k, d) is not None]
        if len(dets) < 3: continue
        L = min(len(val[d]) for d in dets)
        Smat = np.column_stack([np.asarray(val[d], float)[:L] for d in dets])
        for cn, fn in [("consensus", consensus_scores), ("centrality", model_centrality_scores), ("hits", hits_authority_scores)]:
            pick = dets[int(np.argmax(fn(Smat)))]; row[f"{cn}_{pname}"] = o - ap_of(k, pick)
        ems, mvs = [], []
        for d in dets:
            e, m = em_mv(val[d], uni[d]); ems.append(e); mvs.append(m)
        row[f"EM_{pname}"] = o - ap_of(k, dets[int(np.argmax(ems))])
        row[f"MV_{pname}"] = o - ap_of(k, dets[int(np.argmin(mvs))])
    rows.append(row)

df = pd.DataFrame(rows).dropna()
print(f"tasks scored (both pools): {len(df)}")
pm = {"EM": pub.reg_em, "MV": pub.reg_mv, "consensus": pub.reg_consensus, "centrality": pub.reg_mc, "hits": pub.reg_hits}
print(f"\n{'method':<11} {'all-32':>8} {'classical':>10} {'delta':>8} {'p(all-cls)':>11} {'published':>10}")
for m in ["EM", "MV", "consensus", "centrality", "hits"]:
    a, c = df[f"{m}_all"], df[f"{m}_cls"]
    p = wilcoxon(a, c).pvalue if (a != c).any() else float("nan")
    print(f"{m:<11} {a.mean():>8.4f} {c.mean():>10.4f} {a.mean()-c.mean():>+8.4f} {p:>11.4f} {pm[m].loc[df.key].mean():>10.4f}")
df.to_csv(os.path.join(S, "RECONCILE_UOMS_V2.csv"), index=False)
print("\n(the ALL-32 column must track published -- published UOMS select over all 32; that validates the recompute)")

