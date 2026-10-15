# -*- coding: utf-8 -*-
"""Contamination-impact experiment (local, parallel): how much of the agreement selectors' apparent
success is the TRANSDUCTIVE evaluation used by the UOMS literature (criterion computed on the test set,
with anomalies visible) versus our leak-free protocol (criterion on validation normals only).

Per task, fit the 25 classical detectors on Ztr; build the [n x m] anomaly-score matrix on (a) Zval
(validation normals -> INDUCTIVE / leak-free) and (b) Zte = test normals + hard anomalies (TRANSDUCTIVE).
For each agreement selector (consensus, model-centrality, HITS), pick the argmax detector under each
score matrix; regret = oracle_ap - clsap[pick]. Aggregate mean regret per selector x {inductive,
transductive}, against the random-pick and fixed-detector references. Reports the fraction of each
selector's transductive advantage over random that vanishes under the leak-free protocol."""
import json, glob, os, warnings, sys, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
S = r"E:/tmp/claude/E--Projects-Submitted-ADRank/236d6247-42ad-4e62-9828-db625dfa055d/scratchpad"
ST = os.path.join(S, "streamline"); BUN = os.path.join(ST, "bundle_v2")
sys.path.insert(0, S)
from label_free_criteria import consensus_scores, model_centrality_scores, hits_authority_scores
SEL = {"consensus": consensus_scores, "model_centrality": model_centrality_scores, "hits": hits_authority_scores}


def classical_pool():
    from pyod.models.iforest import IForest; from pyod.models.lof import LOF; from pyod.models.knn import KNN
    from pyod.models.hbos import HBOS; from pyod.models.ecod import ECOD; from pyod.models.copod import COPOD
    from pyod.models.pca import PCA as PPCA; from pyod.models.cblof import CBLOF; from pyod.models.loda import LODA
    p = [("IForest", lambda: IForest(n_estimators=100, random_state=0)), ("LOF", lambda: LOF(n_neighbors=20)),
         ("KNN", lambda: KNN(n_neighbors=5)), ("ECOD", lambda: ECOD()), ("COPOD", lambda: COPOD()),
         ("HBOS", lambda: HBOS(n_bins=10)), ("PCA", lambda: PPCA(n_components=0.5, random_state=0)),
         ("CBLOF", lambda: CBLOF(n_clusters=8, random_state=0)), ("LODA", lambda: LODA(n_bins=10))]
    for k in [3, 10, 35, 100, 200]:
        p.append((f"LOF_k{k}", lambda k=k: LOF(n_neighbors=k))); p.append((f"KNN_k{k}", lambda k=k: KNN(n_neighbors=k)))
    for nb in [5, 20, 50]: p.append((f"HBOS_b{nb}", lambda nb=nb: HBOS(n_bins=nb)))
    for nc in [0.3, 0.7, 0.9]: p.append((f"PCA_c{nc}", lambda nc=nc: PPCA(n_components=nc, random_state=0)))
    return p


def work(arg):
    from sklearn.preprocessing import StandardScaler
    key, f, ap, o = arg
    try:
        z = np.load(f)
        Xtr, Xval, Xtn, Xh = z["Xtr"].astype("float64"), z["Xval"].astype("float64"), z["Xtn"].astype("float64"), z["Xh"].astype("float64")
        ss = StandardScaler().fit(Xtr); Ztr = ss.transform(Xtr); Zval = ss.transform(Xval)
        Zte = ss.transform(np.vstack([Xtn, Xh]))
        if len(Zval) < 20 or len(Zte) < 20: return None
        names, Sval_cols, Ste_cols = [], [], []
        for nm, ctor in classical_pool():
            if nm not in ap: continue
            try:
                m = ctor(); m.fit(Ztr)
                sv = np.asarray(m.decision_function(Zval), float); st = np.asarray(m.decision_function(Zte), float)
                if np.all(np.isfinite(sv)) and np.all(np.isfinite(st)) and np.nanstd(sv) > 1e-12 and np.nanstd(st) > 1e-12:
                    names.append(nm); Sval_cols.append(sv); Ste_cols.append(st)
            except Exception: pass
        if len(names) < 5 or "LOF_k10" not in ap: return None
        Sval = np.column_stack(Sval_cols); Ste = np.column_stack(Ste_cols)
        out = {"key": key, "fixed": o - ap["LOF_k10"], "rand": float(np.mean([o - ap[n] for n in names]))}
        for sn, fn in SEL.items():
            pi = names[int(np.argmax(fn(Sval)))]; pt = names[int(np.argmax(fn(Ste)))]
            out[f"{sn}_ind"] = o - ap[pi]; out[f"{sn}_trans"] = o - ap[pt]
        return out
    except Exception as e:
        return ("ERR:" + key, str(e))


def main():
    from multiprocessing import Pool
    clsap = {json.loads(l)["key"]: json.loads(l)["ap"] for l in open(os.path.join(S, "clsstd_valuni", "ap.jsonl"))}
    lb = pd.read_csv(os.path.join(ST, "STREAM_V3_LEADERBOARD.csv"))
    keep = lb[(lb.oracle_ap > 0.05) & (lb.reg_random > 0.02) & (lb.n_h >= 100)].set_index("key")
    oracle = keep.oracle_ap.to_dict()
    files = [f for f in sorted(glob.glob(os.path.join(BUN, "*.npz")))
             if os.path.basename(f)[:-4] in keep.index and os.path.basename(f)[:-4] in clsap]
    args = [(os.path.basename(f)[:-4], f, clsap[os.path.basename(f)[:-4]], oracle[os.path.basename(f)[:-4]]) for f in files]
    print(f"datasets: {len(args)}  workers: 6", flush=True)
    rows = []; done = 0
    with Pool(6) as pool:
        for r in pool.imap_unordered(work, args, chunksize=1):
            done += 1
            if r is None: continue
            if isinstance(r, tuple): print("  " + r[0], r[1], flush=True); continue
            rows.append(r)
            if done % 25 == 0: print(f"  {done}/{len(args)}", flush=True)
    df = pd.DataFrame(rows); df.to_pickle(os.path.join(S, "CONTAM_TRANSDUCTIVE.pkl"))
    rand = df["rand"].mean(); fixed = df["fixed"].mean()
    print(f"\n=== {len(df)} tasks. random pick {rand:.4f}   fixed detector {fixed:.4f} ===")
    print(f"{'selector':18s} {'inductive':>10s} {'transduct.':>10s}  adv_ind  adv_trans   %vanished")
    from scipy.stats import wilcoxon
    for sn in SEL:
        ri = df[f"{sn}_ind"].mean(); rt = df[f"{sn}_trans"].mean()
        adv_i = rand - ri; adv_t = rand - rt
        vanished = 100 * (adv_t - adv_i) / adv_t if adv_t > 1e-9 else float("nan")
        pt = wilcoxon(df[f"{sn}_trans"], df["rand"]).pvalue
        pi = wilcoxon(df[f"{sn}_ind"], df["rand"]).pvalue
        print(f"{sn:18s} {ri:10.4f} {rt:10.4f}  {adv_i:+.4f}  {adv_t:+.4f}   {vanished:5.1f}%   "
              f"(trans vs rand p={pt:.3g}, ind vs rand p={pi:.3g})")
    # pooled across the three agreement selectors
    ai = np.mean([rand - df[f"{s}_ind"].mean() for s in SEL]); at = np.mean([rand - df[f"{s}_trans"].mean() for s in SEL])
    print(f"\n  agreement selectors pooled: transductive advantage over random {at:+.4f}, leak-free {ai:+.4f} "
          f"-> {100*(at-ai)/at:.0f}% of the apparent advantage vanishes under the leak-free protocol")
    print("saved CONTAM_TRANSDUCTIVE.pkl")


if __name__ == "__main__":
    main()
