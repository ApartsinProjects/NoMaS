"""Make-or-break diagnostic for the deep-debias run (modal_deepdebias.py).

Per task, ALL of ap / a1(dep probe) / null(on-manifold interpolants) / gap(train-vs-val overfit)
are co-computed in ONE pass on ONE split for all 32 detectors -> oracle/regret are internally
consistent here (no cross-config joins).

Two questions:
 (Q1 SEPARATION) Among DEEP detectors, do the ones that are the real oracle (genuine deep wins)
   have LOWER null / gap than the deep detectors the probe over-rates? If null/gap separate
   'genuine' from 'fooled' deep, a correction is possible.
 (Q2 REGRET) Does a null/gap-corrected SPARC grade (selecting from ALL 32) beat raw SPARC-all,
   and move toward the perfect-deep-veto ceiling? Baselines: fixed-best-single-detector,
   raw SPARC-all, SPARC-classical, oracle, deep-veto ceiling.
"""
import json, glob, os, numpy as np, pandas as pd
from scipy.stats import wilcoxon, mannwhitneyu
S = r"E:/tmp/claude/E--Projects-Submitted-ADRank/236d6247-42ad-4e62-9828-db625dfa055d/scratchpad"
DEEP = lambda d: d.startswith("D_")
# --- load debias results (self-contained: ap/a1/null/gap for all 32) ---
J = {}
for fn in glob.glob(os.path.join(S, "streamline", "debias_results", "**", "shard_*.jsonl"), recursive=True):
    for ln in open(fn):
        try: r = json.loads(ln)
        except: continue
        if "key" in r and r.get("ap"): J[r["key"]] = r
# --- same 151-task keep filter as the paper ---
lb = pd.read_csv(os.path.join(S, "streamline", "STREAM_V3_LEADERBOARD.csv"))
keep = set(lb[(lb.oracle_ap > 0.05) & (lb.reg_random > 0.02) & (lb.n_h >= 100)].key)
keys = [k for k in J if k in keep]
print(f"debias tasks scored: {len(J)}   in keep-set (of 151): {len(keys)}")
if len(keys) < 5:
    print("too few tasks yet; rerun when the fill completes."); raise SystemExit

# per task helpers on the self-contained basis
def oracle(k):  return max(J[k]["ap"].values())
def dets(k):    return [d for d in J[k]["ap"] if d in J[k]["a1"]]      # gradeable
def cls(k):     return [d for d in dets(k) if not DEEP(d)]
def pick_raw(k, pool):   return max(pool, key=lambda d: J[k]["a1"][d])
def pick_corr(k, pool, alpha, beta):
    def g(d):
        nb = J[k].get("null", {}).get(d, 0.5); gp = J[k].get("gap", {}).get(d, 0.5)
        return J[k]["a1"][d] - alpha * max(0.0, nb - 0.5) - beta * max(0.0, gp - 0.5)
    return max(pool, key=g)

# ============ Q1: does null/gap separate genuine-deep-win from fooled-deep? ============
gen_null, foo_null, gen_gap, foo_gap = [], [], [], []
deep_top_but_lose = 0; deep_genuine = 0
for k in keys:
    ds = dets(k); dp = [d for d in ds if DEEP(d)]
    if not dp: continue
    o = oracle(k); ap = J[k]["ap"]
    real_oracle_is_deep = DEEP(max(ds, key=lambda d: ap[d]))
    probe_top = max(ds, key=lambda d: J[k]["a1"][d]); probe_top_deep = DEEP(probe_top)
    # each deep detector: is it (near-)oracle on real (genuine) or not (fooled), + its null/gap
    for d in dp:
        genuine = ap[d] >= o - 0.02          # within 0.02 of the real oracle
        nb = J[k].get("null", {}).get(d, np.nan); gp = J[k].get("gap", {}).get(d, np.nan)
        (gen_null if genuine else foo_null).append(nb)
        (gen_gap if genuine else foo_gap).append(gp)
    if probe_top_deep and not real_oracle_is_deep: deep_top_but_lose += 1
    if real_oracle_is_deep: deep_genuine += 1
gn, fn_ = np.array(gen_null,float), np.array(foo_null,float)
gg, fg = np.array(gen_gap,float), np.array(foo_gap,float)
def clean(a,b):
    a=a[np.isfinite(a)]; b=b[np.isfinite(b)]; return a,b
print("\n=== Q1  does null / gap separate genuine vs fooled deep detectors? ===")
print(f"  tasks where probe's TOP pick is deep but deep is NOT the real oracle: {deep_top_but_lose}")
print(f"  tasks where a deep detector IS the real oracle: {deep_genuine}")
for nm, a, b in [("null-AUC", *clean(gn, fn_)), ("gap-AUC", *clean(gg, fg))]:
    if len(a) and len(b):
        try: p = mannwhitneyu(a, b, alternative="less").pvalue
        except Exception: p = float("nan")
        print(f"  {nm}: genuine-deep median={np.median(a):.3f} (n={len(a)})   "
              f"fooled-deep median={np.median(b):.3f} (n={len(b)})   MWU(gen<foo) p={p:.3f}")

# ============ Q2: corrected SPARC-all regret vs baselines ============
print("\n=== Q2  regret (lower better) on the self-contained basis ===")
reg_fixed_by_det = {}
for d in set().union(*[set(J[k]["ap"]) for k in keys]):
    rs = [oracle(k) - J[k]["ap"][d] for k in keys if d in J[k]["ap"]]
    if len(rs) >= 0.8 * len(keys): reg_fixed_by_det[d] = np.mean(rs)
best_fixed = min(reg_fixed_by_det, key=reg_fixed_by_det.get)
r_sparc_all = np.mean([oracle(k) - J[k]["ap"][pick_raw(k, dets(k))] for k in keys])
r_sparc_cls = np.mean([oracle(k) - J[k]["ap"][pick_raw(k, cls(k))] for k in keys])
# deep-veto ceiling: SPARC-all but never allowed to pick a deep detector = SPARC-classical here
# perfect-deep-veto: pick classical unless deep is genuinely oracle -> take min(classical-pick, deep-oracle)
r_ceiling = np.mean([min(oracle(k) - J[k]["ap"][pick_raw(k, cls(k))], 0.0) if False else
                     oracle(k) - max(J[k]["ap"][pick_raw(k, cls(k))],
                                     max([J[k]["ap"][d] for d in dets(k) if DEEP(d)], default=-1))
                     for k in keys])
print(f"  oracle regret (def)          : 0.0000")
print(f"  best FIXED single detector   : {reg_fixed_by_det[best_fixed]:.4f}  ({best_fixed})")
print(f"  SPARC classical-only         : {r_sparc_cls:.4f}")
print(f"  SPARC all-32 (raw a1)        : {r_sparc_all:.4f}")
print(f"  perfect deep-veto ceiling    : {r_ceiling:.4f}")
best = None
for alpha in [0.5, 1.0, 1.5, 2.0]:
    for beta in [0.0, 0.5, 1.0]:
        r = np.mean([oracle(k) - J[k]["ap"][pick_corr(k, dets(k), alpha, beta)] for k in keys])
        if best is None or r < best[0]: best = (r, alpha, beta)
r, a, b = best
p = wilcoxon([oracle(k) - J[k]["ap"][pick_corr(k, dets(k), a, b)] for k in keys],
             [oracle(k) - J[k]["ap"][pick_raw(k, dets(k))] for k in keys]).pvalue
print(f"  SPARC all-32 (null/gap corr) : {r:.4f}   (best alpha={a}, beta={b}; vs raw-all p={p:.3f})")
print(f"\n  READ: corrected-all beats fixed={reg_fixed_by_det[best_fixed]:.4f}? "
      f"{'YES' if r < reg_fixed_by_det[best_fixed] else 'no'}   "
      f"beats SPARC-classical={r_sparc_cls:.4f}? {'YES' if r < r_sparc_cls else 'no'}")
