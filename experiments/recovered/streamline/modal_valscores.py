# -*- coding: utf-8 -*-
"""Modal: fit CLASSICAL(25)+DEEP(7) on standardized Ztr and SAVE each detector's raw validation-normal
decision_function vector (+ ap), so IREOS / IFOREST-R can be recomputed on classical/deep/full pools on
the current basis. Same fit as modal_probe1000. Val capped at 800 points, scores rounded to save space."""
import modal
APP_NAME = "adrank-valscores"
app = modal.App(APP_NAME)
image = (modal.Image.from_registry("pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel")
         .pip_install("numpy", "pandas", "scikit-learn", "pyod", "deepod"))
data_vol = modal.Volume.from_name("adrank-bundle", create_if_missing=True)
res_vol = modal.Volume.from_name("adrank-valscores-results", create_if_missing=True)


def _classical_pool():
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


def _deep_pool(dev):
    M = []
    for nm, mk in [("D_DeepSVDD", lambda: __import__("deepod.models", fromlist=["DeepSVDD"]).DeepSVDD(epochs=40, device=dev, random_state=0)),
                   ("D_ICL", lambda: __import__("deepod.models", fromlist=["ICL"]).ICL(epochs=40, device=dev, random_state=0)),
                   ("D_RCA", lambda: __import__("deepod.models", fromlist=["RCA"]).RCA(epochs=40, device=dev, random_state=0)),
                   ("D_SLAD", lambda: __import__("deepod.models", fromlist=["SLAD"]).SLAD(epochs=40, device=dev, random_state=0)),
                   ("D_DIF", lambda: __import__("deepod.models", fromlist=["DeepIsolationForest"]).DeepIsolationForest(epochs=1, device=dev, random_state=0)),
                   ("D_AE", lambda: __import__("pyod.models.auto_encoder", fromlist=["AutoEncoder"]).AutoEncoder(epoch_num=50, verbose=0)),
                   ("D_VAE", lambda: __import__("pyod.models.vae", fromlist=["VAE"]).VAE(epoch_num=50, verbose=0))]:
        try: M.append((nm, mk))
        except Exception: pass
    return M


@app.function(image=image, gpu="A10G", timeout=3600 * 3, volumes={"/data": data_vol, "/results": res_vol})
def run_shard(keys):
    import os, io, contextlib, json, warnings
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import average_precision_score
    import torch
    warnings.filterwarnings("ignore")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    def apn(y, s):
        bb = y.mean(); return float((average_precision_score(y, s) - bb) / (1 - bb + 1e-12))
    import hashlib
    # DETERMINISTIC shard name (builtin hash() is per-process randomized -> re-runs make duplicate shard
    # files, breaking resume and leaving partial downloads mid-fill; hashlib is stable across processes)
    outpath = f"/results/shard_{hashlib.md5(','.join(sorted(keys)).encode()).hexdigest()[:10]}.jsonl"
    done = set()
    if os.path.exists(outpath):
        for ln in open(outpath):
            try:
                r = json.loads(ln)
                if "vs" in r or "err" in r: done.add(r["key"])
            except Exception: pass
    fo = open(outpath, "a")
    for key in keys:
        if key in done: continue
        try:
            z = np.load(f"/data/bv2/{key}.npz")
            Xtr, Xval, Xtn, Xh = z["Xtr"].astype("float64"), z["Xval"].astype("float64"), z["Xtn"].astype("float64"), z["Xh"].astype("float64")
        except Exception as e:
            fo.write(json.dumps({"key": key, "err": f"load:{e}"}) + "\n"); fo.flush(); continue
        Xte = np.vstack([Xtn, Xh]); yte = np.r_[np.zeros(len(Xtn)), np.ones(len(Xh))]
        ss = StandardScaler().fit(Xtr); Ztr, Zte, Zval = ss.transform(Xtr), ss.transform(Xte), ss.transform(Xval)
        if len(Zval) > 800:
            rng = np.random.default_rng(0); sel = np.sort(rng.choice(len(Zval), 800, replace=False)); Zval = Zval[sel]
        res = {"key": key, "ap": {}, "vs": {}, "nval": int(len(Zval))}
        def grade(nm, m):
            st = np.asarray(m.decision_function(Zte), float)
            if np.all(np.isfinite(st)) and np.nanstd(st) > 1e-12: res["ap"][nm] = apn(yte, st)
            sv = np.asarray(m.decision_function(Zval), float)
            if np.all(np.isfinite(sv)): res["vs"][nm] = [round(float(x), 5) for x in sv]
        for nm, ctor in _classical_pool():
            try: m = ctor(); m.fit(Ztr); grade(nm, m)
            except Exception: pass
        for nm, ctor in _deep_pool(dev):
            try:
                m = ctor()
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    m.fit(Ztr); grade(nm, m)
            except Exception: pass
        fo.write(json.dumps(res) + "\n"); fo.flush(); res_vol.commit()
        print(f"  {key}: {len(res['vs'])} detectors, nval={res['nval']}", flush=True)
    fo.close(); res_vol.commit()
    return len(keys)


@app.local_entrypoint()
def main(shards: int = 12):
    import pandas as pd, os
    D = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d\scratchpad\streamline"
    idx = pd.read_csv(os.path.join(D, "bundle_v2", "_index.csv"))
    keys = list(idx.key)
    parts = [keys[i::shards] for i in range(shards)]; parts = [p for p in parts if p]
    total = 0
    for n in run_shard.map(parts): total += n
    print(f"DONE: {total} datasets; val scores in adrank-valscores-results")
