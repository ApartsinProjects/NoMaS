# -*- coding: utf-8 -*-
"""Invariant test for the two time-series fixes (audit bugs 2 and 3).

Expected outcomes, STATED BEFORE RUNNING so a silent bug shows up as a violated invariant
rather than a plausible number:

  I1  label alignment   every window labelled anomalous contains >=1 anomalous raw point.
                        OLD: 12.1% of anomaly windows contained ZERO.        NEW: 0%.
  I2  training purity   no training window contains any anomalous raw point.
                        OLD: violated on 96/96 series.                       NEW: 0 series.
  I3  raw disjointness  no test window shares a raw point with a train window.
                        OLD: 100% of test windows shared (mean 1.0000).      NEW: 0%.
  I4  usable split      train and test both survive the purge at workable sizes.

A window at position p covers raw points [p*STRIDE, p*STRIDE + W), so windows i and j share
raw data iff |p_i - p_j| < W/STRIDE.
"""
import os, re, sys, zipfile, hashlib, warnings
import numpy as np

warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"
S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d\scratchpad"
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, S)
from adrank.ts import _window_features, _window_labels          # noqa: E402
from hadb_ts_final import (W, STRIDE, MAX_LEN, MIN_WIN, MIN_NORM_WIN,  # noqa: E402
                           block_split, keogh_trivial)

N_SERIES = int(sys.argv[1]) if len(sys.argv) > 1 else 40
SEEDS = [0, 1, 2]
SHARE = W // STRIDE          # windows closer than this share raw points


def load_ucr(n):
    z = zipfile.ZipFile(os.path.join(ROOT, "data", "ucr",
                                     "UCR_TimeSeriesAnomalyDatasets2021.zip"))
    pat = re.compile(r"_(\d+)_(\d+)_(\d+)\.txt$", re.I)
    fs = [f for f in z.namelist() if pat.search(f)]
    fs = sorted(fs, key=lambda s: hashlib.sha1(s.encode()).hexdigest())[:n]
    for fn in fs:
        m = pat.search(fn)
        try:
            raw = z.read(fn).decode("utf-8", "replace").split()
            x = np.array([float(v) for v in raw if v.strip()], float)
        except Exception:
            continue
        a0, a1 = int(m.group(2)), int(m.group(3))
        lab = np.zeros(len(x), int)
        lab[a0:min(a1 + 1, len(x))] = 1
        if len(x) > MAX_LEN:
            a = np.where(lab == 1)[0]
            if len(a):
                c = int((a[0] + a[-1]) // 2); half = MAX_LEN // 2
                lo = max(0, min(c - half, len(x) - MAX_LEN))
                x, lab = x[lo:lo + MAX_LEN], lab[lo:lo + MAX_LEN]
        yield os.path.basename(fn), x, lab


fake_old = fake_new = anom_old = anom_new = 0
leak_series_old = leak_series_new = 0
ov_old, ov_new, tr_sz, te_sz, purge = [], [], [], [], []
n_used = 0

for name, x, lab in load_ucr(N_SERIES):
    if keogh_trivial(x, lab)[0]:
        continue
    starts = np.arange(0, len(x) - W + 1, STRIDE)
    Xw, centers = _window_features(x, w=W, stride=STRIDE)
    if len(Xw) != len(starts) or len(Xw) < MIN_WIN:
        continue
    y_new = _window_labels(lab, starts, w=W, min_count=1)      # correct
    y_old = _window_labels(lab, centers, w=W, min_count=1)     # the shipped bug
    # true anomalous content of each window, by its REAL raw span
    true_cnt = np.array([lab[s:s + W].sum() for s in starts], int)

    anom_old += int(y_old.sum()); anom_new += int(y_new.sum())
    fake_old += int(((y_old == 1) & (true_cnt == 0)).sum())
    fake_new += int(((y_new == 1) & (true_cnt == 0)).sum())
    if int((y_new == 0).sum()) < MIN_NORM_WIN:
        continue
    n_used += 1

    pos = np.arange(len(Xw))
    for seed in SEEDS:
        # OLD: random split over NORMAL windows, using the buggy labels
        norm_i = np.where(y_old == 0)[0]
        g = np.random.default_rng(seed); idx = np.arange(len(norm_i)); g.shuffle(idx)
        cut = int(0.8 * len(idx))
        tr_o, te_o = norm_i[idx[:cut]], norm_i[idx[cut:]]
        # NEW: contiguous blocks + purge, using correct labels
        tr_n, te_n = block_split(y_new, pos, seed)

        for tag, tr, te, y in (("old", tr_o, te_o, y_old), ("new", tr_n, te_n, y_new)):
            if not len(tr) or not len(te):
                continue
            d = np.abs(te[:, None] - tr[None, :]).min(axis=1)
            frac = float((d < SHARE).mean())
            (ov_old if tag == "old" else ov_new).append(frac)
        if len(tr_o) and true_cnt[tr_o].sum() > 0:
            leak_series_old += 1
        if len(tr_n) and true_cnt[tr_n].sum() > 0:
            leak_series_new += 1
        tr_sz.append(len(tr_n)); te_sz.append(len(te_n))
        purge.append(1 - len(tr_n) / max(1, int((y_new == 0).sum()) * 0.8))

print(f"series used: {n_used}  (seed-splits: {len(tr_sz)})\n")
print("I1 label alignment  anomaly windows containing ZERO anomalous points")
print(f"     OLD {fake_old}/{anom_old} = {100*fake_old/max(anom_old,1):5.1f}%"
      f"   NEW {fake_new}/{anom_new} = {100*fake_new/max(anom_new,1):5.1f}%"
      f"   {'PASS' if fake_new == 0 else 'FAIL'}")
print("\nI2 training purity  splits whose TRAIN set contains anomalous raw points")
print(f"     OLD {leak_series_old}/{len(tr_sz)}   NEW {leak_series_new}/{len(tr_sz)}"
      f"   {'PASS' if leak_series_new == 0 else 'FAIL'}")
print("\nI3 raw disjointness  test windows sharing raw points with a train window")
if ov_old and ov_new:
    print(f"     OLD mean {np.mean(ov_old):.4f} min {np.min(ov_old):.4f}"
          f"   NEW mean {np.mean(ov_new):.4f} max {np.max(ov_new):.4f}"
          f"   {'PASS' if np.max(ov_new) == 0 else 'FAIL'}")
print("\nI4 usable split  sizes after the purge")
if tr_sz:
    print(f"     train median {int(np.median(tr_sz))}  test median {int(np.median(te_sz))}"
          f"  purge cost median {100*np.median(purge):.1f}% of the old train budget")
    print(f"     splits with train<50: {sum(1 for t in tr_sz if t < 50)}/{len(tr_sz)}"
          f"   {'PASS' if np.median(tr_sz) >= 50 else 'FAIL'}")
