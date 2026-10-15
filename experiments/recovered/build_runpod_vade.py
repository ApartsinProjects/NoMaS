import os
ROOT = r"E:\Projects\Submitted\ADRank"
pipe = open(os.path.join(ROOT, "src", "adrank", "pipeline.py"), encoding="utf-8").read()
vade = open(os.path.join(ROOT, "scripts", "vade.py"), encoding="utf-8").read()
pipe = pipe.replace("from __future__ import annotations\n", "")
vade = vade.replace("from __future__ import annotations\n", "")

header = '''"""RunPod: VaDE-tuning audit for the ADRank ceiling. Fetches DAMI (ARFF from LMU),
runs KMeans vs multiple VaDE configs on the ceiling datasets, reports regret + effK.
"""
from __future__ import annotations
import os
os.environ.setdefault("ADRANK_EXCLUDE_DETECTORS", "OCSVM")
import sys, io, tarfile, urllib.request, time, re
import numpy as np, pandas as pd
import torch
assert torch.cuda.is_available(), "CUDA required."
print("[train] GPU:", torch.cuda.get_device_name(0)); sys.stdout.flush()
'''

agg = '''
def _agg_discriminative(pp):
    ms = pp.dropna(subset=["pseudo_auc"]).groupby(["regime","detector"]).pseudo_auc.mean()
    regimes = pp.regime.unique()
    w = {}
    for r in regimes:
        v = pp[pp.regime==r].dropna(subset=["pseudo_auc"]).groupby("detector").pseudo_auc.mean()
        w[r] = float(v.var())
    sden = sum(w.values()) or 1.0
    w = {r: w[r]/sden for r in regimes}
    dets = ms.index.get_level_values(1).unique()
    out = pd.Series(0.0, index=dets); wsum = 0.0
    for r in ms.index.get_level_values(0).unique():
        rr = ms.loc[r]; rank = rr.rank(ascending=True)
        out = out.add(rank*w.get(r,0.0), fill_value=0.0); wsum += w.get(r,0.0)
    return out/(wsum or 1.0)
'''

# read the fetch+audit tail from a separate file to avoid quote hell
tail = open(os.path.join(os.path.dirname(__file__), "runpod_vade_tail.py"), encoding="utf-8").read()

bundle = header + "\n# ==== pipeline ====\n" + pipe + "\n# ==== vade ====\n" + vade + agg + tail
out = os.path.join(ROOT, "scripts", "runpod_vade.py")
open(out, "w", encoding="utf-8", newline="\n").write(bundle)
print("wrote", out, len(bundle), "bytes")
