# -*- coding: utf-8 -*-
"""PLACEBO TEST for the per-channel triviality criterion on multivariate series.

WHY: the MTS arm reports 86% of TSB-AD-M series "solvable by a single-channel one-liner",
which would independently replicate Pinet et al. 2026. But the criterion tests C channels
x 30 parameterisations, so with C=20 it gets 600 chances for an argmax to land inside the
labelled region, against 30 in the univariate case. At the median 3.7% anomaly rate, chance
alone would flag nearly every series. A suspiciously strong result is a bug until proven
otherwise, so before 86% is reported as anything, it has to survive the same placebo test
that validated the univariate criterion (4.7% on shuffled label positions vs 100% true).

EXPECTED OUTCOME, STATED IN ADVANCE:
  criterion sound   -> placebo fire rate stays low (single digits, as univariate)
  criterion broken  -> placebo fires at a rate near the true rate, and 86% is multiple
                       comparisons rather than triviality
"""
import os, sys, io, zipfile, hashlib, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hadb_ts_final import keogh_trivial  # noqa: E402

ROOT = r"E:\Projects\Submitted\ADRank"
N_SER = int(os.environ.get("PL_SER", "12"))
N_PLACEBO = int(os.environ.get("PL_N", "5"))
MAXL, MAXCH = 10000, 20

z = zipfile.ZipFile(os.path.join(ROOT, "data", "tsbad", "TSB-AD-M.zip"))
fs = sorted(q for q in z.namelist() if q.lower().endswith(".csv"))
fs = sorted(fs, key=lambda s: hashlib.sha1(s.encode()).hexdigest())[:N_SER]

rows, rng = [], np.random.default_rng(0)
for fn in fs:
    try:
        df = pd.read_csv(io.BytesIO(z.read(fn)))
    except Exception:
        continue
    lc = [q for q in df.columns if q.lower() in ("label", "is_anomaly", "anomaly")]
    if not lc:
        continue
    lab = df[lc[0]].to_numpy(int)
    Xc = np.nan_to_num(df.drop(columns=lc).to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)
    if len(Xc) > MAXL:
        a = np.where(lab == 1)[0]
        c = int((a[0] + a[-1]) // 2) if len(a) else MAXL // 2
        lo = max(0, min(c - MAXL // 2, len(Xc) - MAXL))
        Xc, lab = Xc[lo:lo + MAXL], lab[lo:lo + MAXL]
    if Xc.shape[1] > MAXCH:
        Xc = Xc[:, np.argsort(-Xc.std(0))[:MAXCH]]
    L = int(lab.sum())
    if L < 5 or L >= len(lab):
        continue

    def fires(l):
        """True if ANY channel's one-liner argmax lands inside the labelled region."""
        for c in range(Xc.shape[1]):
            if keogh_trivial(Xc[:, c], l)[0]:
                return True
        return False

    t = fires(lab)
    p = 0
    for _ in range(N_PLACEBO):
        s = int(rng.integers(0, len(lab) - L))
        pl = np.zeros(len(lab), int); pl[s:s + L] = 1
        p += int(fires(pl))
    rows.append(dict(dataset=os.path.basename(fn), C=Xc.shape[1], n=len(lab),
                     rate=round(L / len(lab), 5), true_fire=int(t),
                     placebo_fire=p, placebo_n=N_PLACEBO))
    print(f"  {os.path.basename(fn)[:36]:38s} C={Xc.shape[1]:2d} rate={L/len(lab):.4f} "
          f"true={int(t)} placebo={p}/{N_PLACEBO}", flush=True)

R = pd.DataFrame(rows)
R.to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "zz_placebo_mts.csv"), index=False)
tf, pf, pt = R.true_fire.sum(), R.placebo_fire.sum(), R.placebo_n.sum()
print(f"\n  series tested: {len(R)}   channels median {int(R.C.median())}")
print(f"  TRUE    region fires on {tf}/{len(R)} = {100*tf/max(len(R),1):.0f}%")
print(f"  PLACEBO region fires on {pf}/{pt} = {100*pf/max(pt,1):.1f}%")
print(f"  univariate reference placebo rate: 4.7%")
verdict = ("BROKEN - the criterion is measuring MULTIPLE COMPARISONS across channels; "
           "the 86% figure is an artefact and needs a per-channel correction"
           if pf / max(pt, 1) > 0.30 else
           "ACCEPTABLE - placebo stays low, the 86% figure stands")
print(f"  VERDICT: {verdict}")
