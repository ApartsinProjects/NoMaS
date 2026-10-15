# -*- coding: utf-8 -*-
"""Inspect specific MTS datasets: where mixed synthetic loses, what detector it picks vs EM vs
the true best, and HOW the anomalies look (single-channel extreme vs cross-channel rupture).
"""
import os, sys, io, contextlib, zipfile, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"; S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
sys.path.insert(0, os.path.join(S, "scratchpad"))
from dev_common import TS_POOL, true_apnorm
from hadb_ts_final import W, STRIDE, MAX_LEN, block_split3
from hadb_ts_mts import mts_window_features, window_labels as mts_wlab, MAX_CH
LOCAL = ("LOF", "KNN", "CBLOF"); GLOBAL = ("HBOS", "COPOD", "ECOD", "PCA")
fam = lambda v: "local" if str(v).startswith(LOCAL) else ("global" if str(v).startswith(GLOBAL) else "other")
FPC = 8


def mts_load(name):
    z = zipfile.ZipFile(os.path.join(ROOT, "data", "tsbad", "TSB-AD-M.zip"))
    cand = [n for n in z.namelist() if os.path.basename(n) == name] or [n for n in z.namelist() if n.lower().endswith(".csv") and name.split("_")[0] in n]
    df = pd.read_csv(io.BytesIO(z.read(cand[0]))); lc = [q for q in df.columns if q.lower() in ("label", "is_anomaly", "anomaly")]
    lab = df[lc[0]].to_numpy(int); Xc = np.nan_to_num(df.drop(columns=lc).to_numpy(float))
    if len(Xc) > MAX_LEN:
        a = np.where(lab == 1)[0]; c = int((a[0] + a[-1]) // 2) if len(a) else MAX_LEN // 2; lo = max(0, min(c - MAX_LEN // 2, len(Xc) - MAX_LEN)); Xc, lab = Xc[lo:lo + MAX_LEN], lab[lo:lo + MAX_LEN]
    if Xc.shape[1] > MAX_CH:
        Xc = Xc[:, np.argsort(-Xc.std(0))[:MAX_CH]]
    Xw, starts = mts_window_features(Xc); yw = mts_wlab(lab, starts); Xw = np.nan_to_num(Xw)
    return Xw, yw, Xc, lab, Xc.shape[1]


def _hists(Xn, B=30, cap=98):
    H = []
    for j in range(Xn.shape[1]):
        col = Xn[:, j]; cnt, edges = np.histogram(col, bins=B); dens = cnt / max(cnt.sum(), 1)
        lo, hi = np.percentile(col, 100 - cap), np.percentile(col, cap); centers = (edges[:-1] + edges[1:]) / 2
        H.append((edges, dens, (cnt > 0) & (centers >= lo) & (centers <= hi)))
    return H


def gen_mts(Xn, ns, nch, frac=0.4, seed=0):
    rng = np.random.default_rng(seed); n, d = Xn.shape; H = _hists(Xn); out = np.empty((ns, d))
    for r in range(ns):
        base = Xn[rng.integers(n)].copy(); k = max(1, int(frac * nch)); chans = rng.choice(nch, k, replace=False)
        for i, ch in enumerate(chans):
            sl = slice(ch * FPC, ch * FPC + FPC)
            if i % 2 == 0:
                base[sl] = Xn[rng.integers(n), sl]
            else:
                for j in range(ch * FPC, ch * FPC + FPC):
                    edges, dens, allowed = H[j]; w = np.where(allowed, np.maximum(dens, 1e-6) ** -2.0, 0.0)
                    if w.sum() > 0:
                        b = rng.choice(len(w), p=w / w.sum()); base[j] = rng.uniform(edges[b], edges[b + 1])
        out[r] = base
    return out


def sauc(Xtr, Xval, syn):
    Xe = np.vstack([Xval, syn]); ye = np.r_[np.zeros(len(Xval)), np.ones(len(syn))]; o = {}
    for vn, ct in TS_POOL:
        try:
            m = ct()
            with contextlib.redirect_stdout(io.StringIO()): m.fit(Xtr)
            s = np.asarray(m.decision_function(Xe), float)
            if np.all(np.isfinite(s)) and np.nanstd(s) > 1e-12: o[vn] = float(roc_auc_score(ye, s))
        except Exception: pass
    return o


def anom_channels(Xc, lab, nch):
    """How many channels are individually anomalous in the anomaly region (raw-signal per-channel z)."""
    nm = lab == 0; a = np.where(lab == 1)[0]
    if len(a) < 3:
        return np.nan, np.nan
    mu, sd = Xc[nm].mean(0), Xc[nm].std(0) + 1e-9
    za = np.abs((Xc[a] - mu) / sd)                 # [n_anom_pts, nch]
    ch_extreme = (za.max(0) > 3.0)                 # channels with an extreme value somewhere in the anomaly
    return int(ch_extreme.sum()), float(za.max(0).mean())


M = pd.read_csv(os.path.join(S, "HADB_MANIFEST.csv")); inc = M[M.include & M.corpus.eq("tsbad_m")]
rows = []
for _, r in inc.iterrows():
    try:
        Xw, yw, Xc, lab, nch = mts_load(r.dataset)
        pos = np.arange(len(Xw)); tr, va, te = block_split3(yw, pos, 0)
        if len(tr) < 40 or len(va) < 30:
            continue
        ap, emv = true_apnorm("hadb_ts_mts.csv", r.dataset)
    except Exception:
        continue
    common = [v for v, _ in TS_POOL if v in ap.index and v in emv.index and emv[v] == emv[v]]
    if len(common) < 6:
        continue
    av = pd.Series({v: ap[v] for v in common}); ev = pd.Series({v: emv[v] for v in common}); best = av.max()
    sa = sauc(Xw[tr], Xw[va], gen_mts(Xw[va], 150, nch)); sc = {v: sa[v] for v in common if v in sa}
    if len(sc) < 6:
        continue
    sv = pd.Series(sc)
    mix_pick = sv.idxmax(); em_pick = ev.idxmax(); tb = av.idxmax()
    nchx, meanz = anom_channels(Xc, lab, nch)
    rows.append(dict(dataset=r.dataset[:34], nch=nch, n_anom_chan=nchx,
                     mix_pick=mix_pick, mix_fam=fam(mix_pick), mix_reg=round(best - av[mix_pick], 3),
                     em_pick=em_pick, em_fam=fam(em_pick), em_reg=round(best - av[em_pick], 3),
                     true_best=tb, tb_fam=fam(tb)))
R = pd.DataFrame(rows)
R["mix_loses"] = R.mix_reg > R.em_reg + 1e-9
print(f"=== MTS inspection ({len(R)} datasets; mixed loses to EM on {int(R.mix_loses.sum())}) ===\n")
cols = ["dataset", "nch", "n_anom_chan", "true_best", "tb_fam", "mix_pick", "mix_reg", "em_pick", "em_reg"]
print("--- datasets where MIXED LOSES ---")
print(R[R.mix_loses][cols].to_string(index=False))
print("\n--- datasets where MIXED WINS ---")
print(R[~R.mix_loses][cols].to_string(index=False))
print("\n  anomaly nature: n_anom_chan = # channels with an extreme raw value in the anomaly "
      "(1 = single-channel/global-type; many/0 = cross-channel/local-type)")
for tag, sub in [("mixed LOSES", R[R.mix_loses]), ("mixed WINS", R[~R.mix_loses])]:
    if len(sub):
        print(f"  {tag}: median anomalous-channels {sub.n_anom_chan.median():.1f}, "
              f"true-best global {int((sub.tb_fam=='global').sum())}/{len(sub)}")
