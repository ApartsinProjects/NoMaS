# -*- coding: utf-8 -*-
"""(a) Does our label-free selector pick DEEP detectors when they are best? Reads the ap/a1/a4 shard
format. For each dataset run the matched selector (local_ev threshold on a1/a4) over (i) classical-only
and (ii) classical+deep pools; compare regret vs each pool's oracle, how often deep is picked, and
whether it's picked on the datasets where deep IS the oracle."""
import os, sys, json, glob
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
D = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d\scratchpad\streamline"
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(D, "deepres2")
TAU = -0.14; isdeep = lambda v: v.startswith("D_")
bykey = {}
for fn in glob.glob(os.path.join(SRC, "shard_*.jsonl")):
    for line in open(fn):
        line = line.strip()
        if not line: continue
        r = json.loads(line)
        if "ap" not in r or len(r.get("ap", {})) < 6: continue
        bykey[r["key"]] = r   # dedup by key
rows = list(bykey.values())
def sel(ap, a1, a4, names):
    lev = max(a1[v] for v in names) - max(a4[v] for v in names)
    return (max(names, key=lambda v: a1[v]) if lev > TAU else max(names, key=lambda v: a4[v]))
recs = []
for r in rows:
    ap, a1, a4 = r["ap"], r["a1"], r["a4"]
    alln = [v for v in ap if v in a1 and v in a4]
    cls = [v for v in alln if not isdeep(v)]
    if len(cls) < 6 or len(alln) < 6: continue
    orc_c = max(ap[v] for v in cls); orc_f = max(ap[v] for v in alln)
    pick_c = sel(ap, a1, a4, cls); pick_f = sel(ap, a1, a4, alln)
    oracle_is_deep = isdeep(max(alln, key=lambda v: ap[v]))
    recs.append({"key": r["key"], "orc_cls": orc_c, "orc_full": orc_f, "oracle_is_deep": oracle_is_deep,
                 "reg_cls_vs_clsorc": orc_c - ap[pick_c], "reg_full_vs_fullorc": orc_f - ap[pick_f],
                 "reg_cls_vs_fullorc": orc_f - ap[pick_c], "pick_full_is_deep": isdeep(pick_f), "pick_full": pick_f})
df = pd.DataFrame(recs); df.to_csv(os.path.join(D, "STREAM_EXPANDED_SEL.csv"), index=False)
N = len(df)
print(f"=== (a) expanded-pool selection: classical vs classical+deep ({N} datasets) ===\n")
print(f"  oracle ceiling: classical {df.orc_cls.mean():.3f} -> classical+deep {df.orc_full.mean():.3f}  (+{df.orc_full.mean()-df.orc_cls.mean():.3f})")
print(f"  datasets where a DEEP detector is the oracle: {int(df.oracle_is_deep.sum())}/{N} ({df.oracle_is_deep.mean():.0%})\n")
print(f"  our matched selector regret:")
print(f"    classical pool, vs classical oracle:   {df.reg_cls_vs_clsorc.mean():.3f}")
print(f"    FULL pool,      vs FULL oracle:         {df.reg_full_vs_fullorc.mean():.3f}")
print(f"    classical pool, vs FULL oracle (cost of ignoring deep): {df.reg_cls_vs_fullorc.mean():.3f}")
p = wilcoxon(df.reg_full_vs_fullorc, df.reg_cls_vs_fullorc).pvalue if (df.reg_full_vs_fullorc.values != df.reg_cls_vs_fullorc.values).any() else np.nan
print(f"    => adding deep to the pool changes our regret-vs-full-oracle by {df.reg_full_vs_fullorc.mean()-df.reg_cls_vs_fullorc.mean():+.3f} (p={p:.3f})\n")
print(f"  selector picks a deep detector on {int(df.pick_full_is_deep.sum())}/{N} datasets")
dd = df[df.oracle_is_deep]
print(f"  when DEEP is the oracle ({len(dd)} datasets): selector picks deep on {int(dd.pick_full_is_deep.sum())}/{len(dd)}; mean regret-vs-full-oracle {dd.reg_full_vs_fullorc.mean():.3f} (vs {df[~df.oracle_is_deep].reg_full_vs_fullorc.mean():.3f} when oracle is classical)")
print("saved streamline/STREAM_EXPANDED_SEL.csv")
