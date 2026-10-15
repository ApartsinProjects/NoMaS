# -*- coding: utf-8 -*-
"""STAGE 2: leaderboard under PURITY-THRESHOLD MTS labeling. Recompute ONLY the tsbad_m (MTS) rows with
window labels = (anomaly iff purity>=PURITY; normal iff purity==0; 0<purity<PURITY EXCLUDED), enforcing
MIN_HARD anomaly windows (so CreditCard/GECCO drop), then merge onto the UNCHANGED tabular rows of
STREAM_RANK.csv and recompute the leaderboard. Tabular datasets have no windowing -> untouched."""
import os, sys, io, contextlib, warnings
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"; S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
D = os.path.join(S, "scratchpad", "streamline")
sys.path.insert(0, os.path.join(S, "scratchpad")); sys.path.insert(0, os.path.join(ROOT, "src"))
import adrank.pipeline as PP
from dev_common import TAB_POOL
from hadb_ts_final import W, STRIDE, block_split3
from hadb_ts_mts import mts_window_features, load_mts
from hadb_round2_common import consensus_scores, model_centrality_scores, hits_authority_scores, _em_auc, _mv_auc
LOCAL = ("LOF", "KNN", "CBLOF"); GLOBAL = ("HBOS", "COPOD", "ECOD", "PCA")
fam = lambda v: "local" if str(v).startswith(LOCAL) else ("global" if str(v).startswith(GLOBAL) else "other")
Q = 0.05; NGEN = 1000; TAU = -0.14; PURITY = float(os.environ.get("PURITY", "0.5")); MIN_HARD = 100
MTS = {}
for name, src, Xc, lab in load_mts(200): MTS[str(name)[:40]] = (Xc, np.asarray(lab, int).ravel())
def severity(Xtr, X, bins=30):
    n, d = X.shape; sev = np.zeros(n)
    for j in range(d):
        tr = np.sort(Xtr[:, j])
        if tr[-1] - tr[0] < 1e-12: continue
        m = len(tr); fb = np.searchsorted(tr, X[:, j], side="right") / m
        tail = -np.log(np.maximum(2 * np.minimum(fb, 1 - fb), 1.0 / (2 * m)))
        cnt, edges = np.histogram(Xtr[:, j], bins=bins); dens = cnt / max(cnt.sum(), 1)
        b = np.clip(np.digitize(X[:, j], edges[1:-1]), 0, bins - 1); sev = np.maximum(sev, np.maximum(tail, -np.log(dens[b] + 1e-9)))
    return sev
def harden(Xtr, Xhold, Xa):
    so_h, so_a = severity(Xtr, Xhold), severity(Xtr, Xa); tho = np.quantile(so_h, 1 - Q)
    try:
        sc = StandardScaler().fit(Xtr); pca = PCA(n_components=0.95, whiten=True, random_state=0).fit(sc.transform(Xtr))
        Ptr, Ph, Pa = pca.transform(sc.transform(Xtr)), pca.transform(sc.transform(Xhold)), pca.transform(sc.transform(Xa))
        sp_h, sp_a = severity(Ptr, Ph), severity(Ptr, Pa); thp = np.quantile(sp_h, 1 - Q)
    except Exception:
        sp_a = np.zeros(len(Xa)); thp = np.inf
    return Xa[~((so_a > tho) | (sp_a > thp))]
def _H(Xn, cap=98):
    H = []
    for j in range(Xn.shape[1]):
        col = Xn[:, j]; cnt, edges = np.histogram(col, bins=30); dens = cnt / max(cnt.sum(), 1)
        lo, hi = np.percentile(col, 100 - cap), np.percentile(col, cap); ctr = (edges[:-1] + edges[1:]) / 2
        H.append((edges, dens, (cnt > 0) & (ctr >= lo) & (ctr <= hi)))
    return H
def gen_beta(Xn, ns, beta, frac=0.4, seed=0):
    rng = np.random.default_rng(seed); n, d = Xn.shape; H = _H(Xn); out = np.empty((ns, d))
    for r in range(ns):
        base = Xn[rng.integers(n)].copy()
        for j in rng.choice(d, max(1, int(frac * d)), replace=False):
            edges, dens, allowed = H[j]; w = np.where(allowed, np.maximum(dens, 1e-6) ** beta, 0.0)
            base[j] = Xn[rng.integers(n), j] if w.sum() == 0 else rng.uniform(*edges[[(b := rng.choice(len(w), p=w / w.sum())), b + 1]])
        out[r] = base
    return out
def get3_mts(name):
    Xc, lab = MTS[name]; Xw, st = mts_window_features(Xc); Xw = np.nan_to_num(Xw)
    pur = np.array([lab[s:s + W].sum() / W for s in st]); allpos = np.arange(len(Xw))
    keep = (pur == 0) | (pur >= PURITY)                          # drop ambiguous 0<pur<PURITY
    Xw2, yw, pos = Xw[keep], (pur[keep] >= PURITY).astype(int), allpos[keep]
    if int(yw.sum()) < MIN_HARD: return None                     # dataset drops (CreditCard/GECCO)
    tr, va, te = block_split3(yw, pos, 0)
    return Xw2[tr][yw[tr] == 0], Xw2[va][yw[va] == 0], Xw2[te][yw[te] == 0], Xw2[yw == 1]
rk = pd.read_csv(os.path.join(D, "STREAM_RANK.csv")); mts_names = list(rk[rk.corpus == "tsbad_m"].dataset)
t = np.linspace(0, 100, 1000); alpha = np.linspace(0.9, 0.999, 1000); rows = []; dropped = []
for i, nm in enumerate(mts_names):
    r3 = None
    try: r3 = get3_mts(nm)
    except Exception: r3 = None
    if r3 is None: dropped.append(nm); continue
    Xtr, Xval, Xtn, Xa = r3
    if len(Xtr) < 40 or len(Xval) < 20 or len(Xtn) < 10 or len(Xa) < 5: dropped.append(nm); continue
    Xh = harden(Xtr, np.vstack([Xval, Xtn]), Xa)
    if len(Xh) < 20: dropped.append(nm); continue
    Xte = np.vstack([Xtn, Xh]); yte = np.r_[np.zeros(len(Xtn)), np.ones(len(Xh))]; base = yte.mean()
    rng = np.random.default_rng(0); lo, hi = Xval.min(0), Xval.max(0); U = lo + rng.random((NGEN, Xval.shape[1])) * np.where(hi > lo, hi - lo, 1.0)
    syn1, syn4 = gen_beta(Xval, 200, 1.0), gen_beta(Xval, 200, -4.0)
    names, V, ap, a1, a4, emv, mvv = [], [], {}, {}, {}, {}, {}
    ye1 = np.r_[np.zeros(len(Xval)), np.ones(len(syn1))]
    for vn, ct in TAB_POOL:
        try:
            m = ct()
            with contextlib.redirect_stdout(io.StringIO()): m.fit(Xtr)
            sv = np.asarray(m.decision_function(Xval), float); st = np.asarray(m.decision_function(Xte), float); su = np.asarray(m.decision_function(U), float)
            s1 = np.asarray(m.decision_function(syn1), float); s4 = np.asarray(m.decision_function(syn4), float)
            if not (np.all(np.isfinite(sv)) and np.all(np.isfinite(st)) and np.nanstd(st) > 1e-12): continue
            names.append(vn); V.append(sv)
            ap[vn] = (average_precision_score(yte, st) - base) / (1 - base + 1e-12)
            a1[vn] = roc_auc_score(ye1, np.r_[sv, s1]); a4[vn] = roc_auc_score(np.r_[np.zeros(len(Xval)), np.ones(len(syn4))], np.r_[sv, s4])
            su = np.nan_to_num(su, nan=float(np.nanmedian(su[np.isfinite(su)])) if np.isfinite(su).any() else 0.0)
            emv[vn] = _em_auc(t, 1.0, -su, -sv, NGEN); mvv[vn] = _mv_auc(alpha, 1.0, -su, -sv, NGEN)
        except Exception: pass
    if len(names) < 6: dropped.append(nm); continue
    Vm = np.column_stack(V); cons = dict(zip(names, consensus_scores(Vm))); mc = dict(zip(names, model_centrality_scores(Vm))); ht = dict(zip(names, hits_authority_scores(Vm)))
    av = pd.Series(ap); best = av.max(); loc = [av[v] for v in names if fam(v) == "local"]; glo = [av[v] for v in names if fam(v) == "global"]
    p1 = max(names, key=lambda v: a1[v]); pe = max(names, key=lambda v: a4[v]); lev = max(a1[v] for v in names) - max(a4[v] for v in names)
    pk = {"beta1": p1, "matched": p1 if lev > TAU else pe, "em": max(names, key=lambda v: emv[v]), "mv": min(names, key=lambda v: mvv[v]),
          "consensus": max(names, key=lambda v: cons[v]), "model_centrality": max(names, key=lambda v: mc[v]), "hits": max(names, key=lambda v: ht[v])}
    rec = {"corpus": "tsbad_m", "dataset": nm, "truefam": "local" if (loc and (not glo or max(loc) > max(glo))) else "global", "reg_random": best - av.mean()}
    for k, v in pk.items(): rec["reg_" + k] = best - av[v]
    rows.append(rec)
    if i % 10 == 0: print(f"  ..{i}/{len(mts_names)} {len(rows)} kept", flush=True)
new_mts = pd.DataFrame(rows)
df = pd.concat([rk[rk.corpus != "tsbad_m"], new_mts], ignore_index=True)
df.to_csv(os.path.join(D, "STREAM_RANK_PURITY.csv"), index=False)
def macro(frame, c): return np.mean([frame[frame.truefam == fa][c].mean() for fa in ["local", "global"] if (frame.truefam == fa).any()])
def board(frame, tag):
    e, mv = frame.reg_em.values, frame.reg_mv.values
    print(f"\n=== {tag} ({len(frame)}: {int((frame.truefam=='local').sum())} local / {int((frame.truefam=='global').sum())} global) ===")
    print(f"  {'method':22s} {'micro':>6s} {'macro':>6s} {'vsEM':>7s} {'vsMV':>7s}")
    for nm, c in [("NoMaS matched (ours)", "reg_matched"), ("NoMaS beta=1 (ours)", "reg_beta1"), ("EM", "reg_em"), ("MV", "reg_mv"),
                  ("consensus/UDR", "reg_consensus"), ("ModelCentrality", "reg_model_centrality"), ("HITS", "reg_hits"), ("random", "reg_random")]:
        a = frame[c].values
        pe = wilcoxon(a, e).pvalue if c != "reg_em" and (a != e).any() else np.nan
        pm = wilcoxon(a, mv).pvalue if c != "reg_mv" and (a != mv).any() else np.nan
        print(f"  {nm:22s} {a.mean():6.3f} {macro(frame,c):6.3f} {pe:7.3f} {pm:7.3f}")
print(f"PURITY={PURITY}  MTS recomputed: {len(new_mts)} kept, {len(dropped)} dropped -> {dropped}")
board(rk, "BEFORE (current >=1pt labeling)")
board(df, f"AFTER (purity>={PURITY} labeling)")
print("\nsaved streamline/STREAM_RANK_PURITY.csv")
