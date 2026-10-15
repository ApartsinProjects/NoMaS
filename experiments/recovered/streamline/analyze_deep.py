# -*- coding: utf-8 -*-
"""Analyze Modal deep-detector results: deep-vs-classical leaderboard + all-bad gate (does any deep
detector beat the classical best on the all-bad datasets?). Reads deepres/shard_*.jsonl."""
import os, sys, json, glob
import numpy as np, pandas as pd
D = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d\scratchpad\streamline"
rows = []
for fn in glob.glob(os.path.join(D, "deepres", "shard_*.jsonl")):
    for line in open(fn):
        line = line.strip()
        if line: rows.append(json.loads(line))
df = pd.DataFrame(rows)
detcols = [c for c in df.columns if c not in ("key", "err")]
classical = [c for c in detcols if not c.startswith("D_")]
deep = [c for c in detcols if c.startswith("D_")]
print(f"=== Modal deep run: {len(df)} datasets, {len(classical)} classical + {len(deep)} deep detectors ===")
print(f"  deep detectors present: {deep}\n")
df["best_classical"] = df[classical].max(1)
df["best_deep"] = df[deep].max(1) if deep else np.nan
df["best_all"] = df[detcols].max(1)
df["deep_wins"] = df["best_deep"] > df["best_classical"]
# per-detector mean ap_norm (leaderboard of detectors)
print("  --- top detectors by mean ap_norm (deep marked *) ---")
means = df[detcols].mean().sort_values(ascending=False)
for k, v in means.head(15).items(): print(f"    {'*' if k.startswith('D_') else ' '} {k:16s} {v:.3f}")
print(f"\n  best classical mean {df.best_classical.mean():.3f}  |  best deep mean {df.best_deep.mean():.3f}  |  best overall {df.best_all.mean():.3f}")
print(f"  datasets where a DEEP detector beats the best classical: {int(df.deep_wins.sum())}/{len(df)} ({df.deep_wins.mean():.0%})")
print(f"  mean uplift when deep wins: {df[df.deep_wins].apply(lambda r: r.best_deep-r.best_classical,1).mean():.3f}")
# all-bad gate: datasets where classical best <= 0.10
ab = df[df.best_classical <= 0.10]
cracked = ab[ab.best_deep > 0.10]
print(f"\n  === ALL-BAD GATE ===")
print(f"  all-bad by classical (best_classical<=0.10): {len(ab)}")
print(f"  of those, deep lifts above 0.10: {len(cracked)}/{len(ab)} -> {[c[:26] for c in cracked.key.head(10)]}")
print(f"  mean best_classical on all-bad {ab.best_classical.mean():.3f} -> best_deep {ab.best_deep.mean():.3f}")
df.to_csv(os.path.join(D, "STREAM_DEEP_MODAL.csv"), index=False)
print("\nsaved streamline/STREAM_DEEP_MODAL.csv")
