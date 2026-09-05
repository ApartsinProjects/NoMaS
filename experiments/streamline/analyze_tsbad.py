# -*- coding: utf-8 -*-
"""(b) Raw-sequence deep detectors on MTS. Per-model mean ap_norm + best raw-deep per dataset.
CAVEAT: TSB-AD native protocol (temporal split, per-timestep->per-window, >=1pt labels, no feature
hardening) DIFFERS from the tabular hardened pipeline, so the window-classical comparison is indicative,
NOT construct-matched (shown separately, not merged)."""
import os, json, glob
import numpy as np, pandas as pd
D = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d\scratchpad\streamline"
rows = {}
for fn in glob.glob(os.path.join(D, "tsbadres", "shard_*.jsonl")):
    for ln in open(fn):
        ln = ln.strip()
        if not ln: continue
        r = json.loads(ln)
        if "ap" in r and len(r.get("ap", {})): rows[r["key"]] = r   # dedup by key
df = pd.DataFrame([{"key": k, **v["ap"]} for k, v in rows.items()])
models = [c for c in df.columns if c != "key"]
raw_deep = [m for m in models if m != "IForest_raw"]
print(f"=== (b) raw-sequence deep on MTS ({len(df)} datasets) — TSB-AD native protocol ===\n")
print("  per-model mean ap_norm (over datasets where it ran):")
for m in sorted(models, key=lambda c: -df[c].mean()):
    print(f"    {m:22s} {df[m].mean():.3f}   (n={int(df[m].notna().sum())})")
df["best_rawdeep"] = df[raw_deep].max(1); df["best_all_raw"] = df[models].max(1)
print(f"\n  best raw-deep per dataset: mean {df.best_rawdeep.mean():.3f}")
print(f"  IForest_raw (unsup reference): mean {df['IForest_raw'].mean():.3f}")
print(f"  raw-deep beats IForest_raw on {int((df.best_rawdeep > df['IForest_raw']).sum())}/{len(df)} datasets")
# indicative-only cross-protocol comparison to window-feature detectors
dm = os.path.join(D, "STREAM_DEEP_MODAL.csv")
if os.path.exists(dm):
    wf = pd.read_csv(dm); wf = wf[wf.key.str.startswith("tsbad_m")]
    cls = [c for c in wf.columns if c not in ("key", "err") and not c.startswith("D_") and not c.startswith("best") and not c.startswith("deep")]
    wf["win_classical_best"] = wf[cls].max(1)
    m = df.merge(wf[["key", "win_classical_best"]], on="key", how="inner")
    print(f"\n  [INDICATIVE, cross-protocol] on {len(m)} matched MTS datasets:")
    print(f"    raw-deep best (native proto)      mean {m.best_rawdeep.mean():.3f}")
    print(f"    window-classical best (hardened)  mean {m.win_classical_best.mean():.3f}")
    print("    (protocols differ - not a matched claim; a fair test needs both on the same split/labels)")
df.to_csv(os.path.join(D, "STREAM_TSBAD.csv"), index=False)
print("\nsaved streamline/STREAM_TSBAD.csv")
