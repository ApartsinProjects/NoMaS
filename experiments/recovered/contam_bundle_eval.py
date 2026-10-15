# -*- coding: utf-8 -*-
"""Contamination-impact experiment (local, parallel): evaluate the SAME source datasets as a CONTAMINATED
benchmark (bundle: naive leaky split + trivially-separable anomalies + no base-rate cap) versus the CLEANED
ADReal benchmark (bundle_v2). Per task: fit the 25 classical detectors on Ztr, score the test set, compute
each detector's base-rate-normalized average precision (ap_norm), the oracle (max), the random-pick regret
(oracle - mean), and the leak-free agreement-selector regret (criterion on validation normals). Aggregate
the two versions to show how contamination inflates apparent achievability and hides the selection problem."""
import glob, os, warnings, sys, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
S = r"E:/tmp/claude/E--Projects-Submitted-ADRank/236d6247-42ad-4e62-9828-db625dfa055d/scratchpad"
ST = os.path.join(S, "streamline")
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


def work(f):
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import average_precision_score
    key = os.path.basename(f)[:-4]
    try:
        z = np.load(f)
        Xtr, Xval, Xtn, Xh = z["Xtr"].astype("float64"), z["Xval"].astype("float64"), z["Xtn"].astype("float64"), z["Xh"].astype("float64")
        if len(Xh) < 5 or len(Xtn) < 5 or len(Xval) < 20: return None
        ss = StandardScaler().fit(Xtr); Ztr = ss.transform(Xtr); Zval = ss.transform(Xval)
        Xte = np.vstack([Xtn, Xh]); Zte = ss.transform(Xte); yte = np.r_[np.zeros(len(Xtn)), np.ones(len(Xh))]
        r = float(yte.mean())
        names, ap, Sval = [], {}, []
        for nm, ctor in classical_pool():
            try:
                m = ctor(); m.fit(Ztr)
                st = np.asarray(m.decision_function(Zte), float); sv = np.asarray(m.decision_function(Zval), float)
                if not (np.all(np.isfinite(st)) and np.nanstd(st) > 1e-12 and np.nanstd(sv) > 1e-12): continue
                a = average_precision_score(yte, st); an = (a - r) / (1 - r + 1e-12)
                names.append(nm); ap[nm] = float(an); Sval.append(sv)
            except Exception: pass
        if len(names) < 5: return None
        o = max(ap.values()); reg_rand = o - float(np.mean(list(ap.values())))
        out = {"key": key, "base_rate": r, "oracle": o, "reg_random": reg_rand, "n_anom": int(len(Xh))}
        Sv = np.column_stack(Sval)
        for sn, fn in SEL.items():
            out[f"reg_{sn}"] = o - ap[names[int(np.argmax(fn(Sv)))]]
        return out
    except Exception as e:
        return ("ERR:" + key, str(e))


def run_bundle(name):
    from multiprocessing import Pool
    files = sorted(glob.glob(os.path.join(ST, name, "*.npz")))
    print(f"[{name}] {len(files)} tasks", flush=True)
    rows = []
    with Pool(6) as pool:
        for i, r in enumerate(pool.imap_unordered(work, files, chunksize=1)):
            if isinstance(r, tuple): print("  " + r[0], r[1], flush=True); continue
            if r: rows.append(r)
            if (i + 1) % 40 == 0: print(f"  {i+1}/{len(files)}", flush=True)
    return pd.DataFrame(rows)


def summarize(df, label):
    n = len(df)
    triv = (df.reg_random <= 0.02).mean() * 100
    unsolv = (df.oracle <= 0.05).mean() * 100
    hi_br = (df.base_rate > 0.5).mean() * 100
    print(f"\n[{label}] {n} tasks")
    print(f"  median base rate {df.base_rate.median():.3f}   tasks with base rate > 0.5: {hi_br:.0f}%")
    print(f"  median oracle ap_norm {df.oracle.median():.3f}   mean {df.oracle.mean():.3f}")
    print(f"  selection-trivial (random-pick regret <= 0.02): {triv:.0f}%   unsolvable (oracle <= 0.05): {unsolv:.0f}%")
    print(f"  random-pick regret {df.reg_random.mean():.4f}")
    for sn in SEL:
        adv = df.reg_random.mean() - df[f"reg_{sn}"].mean()
        print(f"    {sn:16s} regret {df[f'reg_{sn}'].mean():.4f}   advantage over random {adv:+.4f}")


def main():
    dc = run_bundle("bundle"); dc.to_pickle(os.path.join(S, "CONTAM_bundle.pkl"))
    dk = run_bundle("bundle_v2"); dk.to_pickle(os.path.join(S, "CONTAM_bundle_v2.pkl"))
    summarize(dc, "CONTAMINATED (raw: leaky split + trivial anomalies + no cap)")
    summarize(dk, "CLEANED (ADReal: leak-free + hardened + capped)")
    print("\nsaved CONTAM_bundle.pkl, CONTAM_bundle_v2.pkl")


if __name__ == "__main__":
    main()
