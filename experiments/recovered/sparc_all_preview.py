"""Preview: SPARC classical-only vs all-32, from v2 nested shards (buggy probe; directional).
Regret self-consistent within the v2 ap table. Cross-checks against the published 151 keys."""
import json, glob, numpy as np, pandas as pd
from pathlib import Path
DL = Path(r"E:/tmp/claude/E--Projects-Submitted-ADRank/236d6247-42ad-4e62-9828-db625dfa055d/scratchpad/v2shards")
FINAL = pd.read_csv(r"E:/Projects/Submitted/zenodo_build/results/model_selection_per_dataset.csv")
final_keys = set(FINAL["dataset"])

rows = {}
for f in glob.glob(str(DL/"*.jsonl")):
    for ln in open(f):
        try: r = json.loads(ln)
        except Exception: continue
        if "ap" in r and "a1" in r: rows[r["key"]] = r
print(f"shards aggregated: {len(rows)} tasks with ap+a1")

DEEP = lambda d: d.startswith("D_")
keys = [k for k in rows if k in final_keys]
print(f"matched to final-151: {len(keys)}")

def regret_table(keys):
    out = []
    for k in keys:
        ap = rows[k]["ap"]; a1 = rows[k]["a1"]
        cand_all = [d for d in ap if d in a1]
        cand_cls = [d for d in cand_all if not DEEP(d)]
        if not cand_cls or not cand_all: continue
        oracle = max(ap.values())
        sel_cls = max(cand_cls, key=lambda d: a1[d])
        sel_all = max(cand_all, key=lambda d: a1[d])
        out.append(dict(key=k, oracle=oracle,
                        sel_cls=sel_cls, reg_cls=oracle-ap[sel_cls],
                        sel_all=sel_all, reg_all=oracle-ap[sel_all],
                        all_picked_deep=DEEP(sel_all)))
    return pd.DataFrame(out)

t = regret_table(keys)
print(f"\ntasks scored: {len(t)}")
print(f"SPARC classical-only  mean regret = {t.reg_cls.mean():.4f}   (published fixed-probe = 0.113; buggy ~0.122)")
print(f"SPARC all-32          mean regret = {t.reg_all.mean():.4f}")
print(f"delta (all - cls)     = {t.reg_all.mean()-t.reg_cls.mean():+.4f}")
print(f"tasks where all-32 picks a DEEP detector: {t.all_picked_deep.sum()}/{len(t)} ({100*t.all_picked_deep.mean():.0f}%)")
print(f"of those, mean regret all-32 = {t.loc[t.all_picked_deep,'reg_all'].mean():.4f} vs cls {t.loc[t.all_picked_deep,'reg_cls'].mean():.4f}")
print(f"\nvs best fixed detector reference 0.130 (published):")
print(f"  all-32 beats 0.130? {t.reg_all.mean():.4f} {'<' if t.reg_all.mean()<0.130 else '>='} 0.130")
from scipy import stats
# paired all vs cls
w = stats.wilcoxon(t.reg_all, t.reg_cls) if (t.reg_all!=t.reg_cls).any() else None
if w: print(f"  Wilcoxon all vs cls: p={w.pvalue:.4f}")
print("\ntop deep detectors picked by all-32:")
print(t.loc[t.all_picked_deep,'sel_all'].value_counts())
