"""Local standardized CLASSICAL scorer that also saves val/uni scores for the UOMS criteria.
Fits each classical detector on standardized Ztr (matches the published clstd/clfix basis) and scores
Zval (validation normals) + Zu (uniform), where Zu is built EXACTLY as modal_probefix builds it (raw
Xval box, seed 0, 800 pts, then standardized) so classical val/uni combine bit-consistently with the
probefix DEEP val/uni. Writes clsstd_valuni/sc_<key>.npz (val__/uni__ per detector) + a jsonl with ap.
Deep scores are reused from the probefix run (deep there was already standardized/correct)."""
import os, glob, json, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score
BUN = r"E:/tmp/claude/E--Projects-Submitted-ADRank/236d6247-42ad-4e62-9828-db625dfa055d/scratchpad/streamline/bundle_v2"
OUT = r"E:/tmp/claude/E--Projects-Submitted-ADRank/236d6247-42ad-4e62-9828-db625dfa055d/scratchpad/clsstd_valuni"
os.makedirs(OUT, exist_ok=True)

def _classical_pool():
    from pyod.models.iforest import IForest; from pyod.models.lof import LOF; from pyod.models.knn import KNN
    from pyod.models.hbos import HBOS; from pyod.models.ecod import ECOD; from pyod.models.copod import COPOD
    from pyod.models.pca import PCA as PPCA; from pyod.models.cblof import CBLOF; from pyod.models.loda import LODA
    p = [("IForest", lambda: IForest(n_estimators=100, random_state=0)), ("LOF", lambda: LOF(n_neighbors=20)),
         ("KNN", lambda: KNN(n_neighbors=5)), ("ECOD", lambda: ECOD()), ("COPOD", lambda: COPOD()),
         ("HBOS", lambda: HBOS(n_bins=10)), ("PCA", lambda: PPCA(n_components=0.5, random_state=0)),
         ("CBLOF", lambda: CBLOF(n_clusters=8, random_state=0)), ("LODA", lambda: LODA(n_bins=10, random_state=0))]
    for k in [3, 10, 35, 100, 200]:
        p.append((f"LOF_k{k}", lambda k=k: LOF(n_neighbors=k))); p.append((f"KNN_k{k}", lambda k=k: KNN(n_neighbors=k)))
    for nb in [5, 20, 50]: p.append((f"HBOS_b{nb}", lambda nb=nb: HBOS(n_bins=nb)))
    for nc in [0.3, 0.7, 0.9]: p.append((f"PCA_c{nc}", lambda nc=nc: PPCA(n_components=nc, random_state=0)))
    return p

def apn(y, s):
    bb = y.mean(); return float((average_precision_score(y, s) - bb) / (1 - bb + 1e-12))

outj = open(os.path.join(OUT, "ap.jsonl"), "w")
files = sorted(glob.glob(os.path.join(BUN, "*.npz")))
pool = _classical_pool()
for i, f in enumerate(files):
    key = os.path.basename(f)[:-4]
    z = np.load(f)
    Xtr, Xval, Xtn, Xh = [z[k].astype("float64") for k in ("Xtr", "Xval", "Xtn", "Xh")]
    ss = StandardScaler().fit(Xtr)
    Ztr, Zval, Zte = ss.transform(Xtr), ss.transform(Xval), ss.transform(np.vstack([Xtn, Xh]))
    yte = np.r_[np.zeros(len(Xtn)), np.ones(len(Xh))]
    # uniform U built EXACTLY like modal_probefix: raw Xval box, seed 0, 800 pts, then standardized
    rng = np.random.default_rng(0); lo, hi = Xval.min(0), Xval.max(0)
    U = lo + rng.random((800, Xval.shape[1])) * np.where(hi > lo, hi - lo, 1.0); Zu = ss.transform(U)
    VAL, UNI, ap = {}, {}, {}
    for nm, ctor in pool:
        try:
            m = ctor(); m.fit(Ztr)
            sv = np.asarray(m.decision_function(Zval), float)
            su = np.asarray(m.decision_function(Zu), float)
            st = np.asarray(m.decision_function(Zte), float)
            if np.all(np.isfinite(sv)) and np.all(np.isfinite(su)):
                VAL[nm] = sv.astype("float32"); UNI[nm] = su.astype("float32")
            if np.all(np.isfinite(st)) and np.nanstd(st) > 1e-12: ap[nm] = apn(yte, st)
        except Exception: pass
    np.savez_compressed(os.path.join(OUT, f"sc_{key}.npz"),
                        **{f"val__{k}": v for k, v in VAL.items()}, **{f"uni__{k}": v for k, v in UNI.items()})
    outj.write(json.dumps({"key": key, "ap": ap}) + "\n"); outj.flush()
    if i % 20 == 0: print(f"  {i}/{len(files)} {key}: {len(VAL)} classical detectors", flush=True)
outj.close()
print(f"done -> {OUT}")
