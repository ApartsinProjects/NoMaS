# -*- coding: utf-8 -*-
"""(b) Bundle RAW MTS series (Xc n x C + per-timestep labels) for the 41 tsbad_m datasets in the
benchmark, so a Modal TSB-AD harness can run raw-sequence deep detectors (USAD/TranAD/etc). Per-dataset
npz (incremental by nature). Capped at MAX_LEN like the construction pipeline."""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"; S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
D = os.path.join(S, "scratchpad", "streamline"); OUTB = os.path.join(D, "rawbundle"); os.makedirs(OUTB, exist_ok=True)
sys.path.insert(0, os.path.join(S, "scratchpad"))
from hadb_ts_final import W, STRIDE, MAX_LEN
from hadb_ts_mts import load_mts
rk = pd.read_csv(os.path.join(D, "STREAM_RANK.csv")); inbench = set(rk[rk.corpus == "tsbad_m"].dataset)
idx = []
for name, src, Xc, lab in load_mts(200):
    nm = str(name)[:40]
    if nm not in inbench: continue
    Xc = np.nan_to_num(np.asarray(Xc, float)); lab = np.asarray(lab, int).ravel()
    if len(Xc) > MAX_LEN:                                    # center on the anomaly region, like the pipeline
        a = np.where(lab == 1)[0]; c = int((a[0] + a[-1]) // 2) if len(a) else MAX_LEN // 2
        lo = max(0, min(c - MAX_LEN // 2, len(Xc) - MAX_LEN)); Xc, lab = Xc[lo:lo + MAX_LEN], lab[lo:lo + MAX_LEN]
    key = nm.replace("/", "_").replace(" ", "_")
    np.savez_compressed(os.path.join(OUTB, key + ".npz"), Xc=Xc.astype(np.float32), lab=lab.astype(np.int8))
    idx.append({"key": key, "dataset": nm, "n": len(Xc), "C": Xc.shape[1], "n_anom": int(lab.sum())})
pd.DataFrame(idx).to_csv(os.path.join(OUTB, "_index.csv"), index=False)
tot = sum(os.path.getsize(os.path.join(OUTB, x)) for x in os.listdir(OUTB)) / 1e6
print(f"bundled {len(idx)} raw MTS series -> {OUTB}  ({tot:.0f} MB)")
