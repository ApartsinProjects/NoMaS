# -*- coding: utf-8 -*-
"""Modal harness: for each bundled dataset, fit CLASSICAL pool + DEEP detectors on the SAME hardened
split and co-compute ap_norm. Fan out over shards. Deep: pyod AE/VAE + deepod DeepSVDD/DIF/SLAD/ICL."""
import modal

APP_NAME = "adrank-deep"
app = modal.App(APP_NAME)
image = (
    modal.Image.from_registry("pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel")
    .pip_install("numpy", "pandas", "scikit-learn", "pyod", "deepod")
)
data_vol = modal.Volume.from_name("adrank-bundle", create_if_missing=True)
res_vol = modal.Volume.from_name("adrank-deep2-results", create_if_missing=True)


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
    try:
        from deepod.models import DeepSVDD; M.append(("D_DeepSVDD", lambda: DeepSVDD(epochs=40, device=dev, random_state=0)))
    except Exception: pass
    try:
        from deepod.models import ICL; M.append(("D_ICL", lambda: ICL(epochs=40, device=dev, random_state=0)))
    except Exception: pass
    try:
        from deepod.models import RCA; M.append(("D_RCA", lambda: RCA(epochs=40, device=dev, random_state=0)))
    except Exception: pass
    try:
        from deepod.models import SLAD; M.append(("D_SLAD", lambda: SLAD(epochs=40, device=dev, random_state=0)))
    except Exception: pass
    try:
        from deepod.models import DeepIsolationForest as DIF; M.append(("D_DIF", lambda: DIF(epochs=1, device=dev, random_state=0)))
    except Exception: pass
    try:
        from pyod.models.auto_encoder import AutoEncoder; M.append(("D_AE", lambda: AutoEncoder(epoch_num=50, verbose=0)))
    except Exception: pass
    try:
        from pyod.models.vae import VAE; M.append(("D_VAE", lambda: VAE(epoch_num=50, verbose=0)))
    except Exception: pass
    return M


def _gen_beta(Xn, ns, beta, frac=0.4, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed); n, d = Xn.shape; out = np.empty((ns, d)); Hs = []
    for j in range(d):
        col = Xn[:, j]; cnt, edges = np.histogram(col, bins=30); dens = cnt / max(cnt.sum(), 1)
        lo, hi = np.percentile(col, 2), np.percentile(col, 98); ctr = (edges[:-1] + edges[1:]) / 2
        allowed = (cnt > 0) & (ctr >= lo) & (ctr <= hi); w = np.where(allowed, np.maximum(dens, 1e-6) ** beta, 0.0)
        Hs.append((edges, w / w.sum() if w.sum() > 0 else None))
    for r in range(ns):
        base = Xn[rng.integers(n)].copy()
        for j in rng.choice(d, max(1, int(frac * d)), replace=False):
            edges, w = Hs[j]; base[j] = Xn[rng.integers(n), j] if w is None else rng.uniform(*edges[[(b := rng.choice(len(w), p=w)), b + 1]])
        out[r] = base
    return out


@app.function(image=image, gpu="A10G", timeout=3600 * 3, volumes={"/data": data_vol, "/results": res_vol})
def run_shard(keys):
    import os, io, contextlib, json, warnings
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import average_precision_score, roc_auc_score
    import torch
    warnings.filterwarnings("ignore")
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def apn(y, s):
        bb = y.mean(); return float((average_precision_score(y, s) - bb) / (1 - bb + 1e-12))
    outpath = f"/results/shard_{abs(hash(tuple(keys))) % 10**8}.jsonl"
    done = set()
    if os.path.exists(outpath):
        for ln in open(outpath):
            try:
                r = json.loads(ln)
                if "ap" in r or "err" in r: done.add(r["key"])   # only skip rows in THIS (nested) format
            except Exception: pass
    n_written = len(done)
    fo = open(outpath, "a")
    for key in keys:
        if key in done: continue
        try:
            z = np.load(f"/data/b/{key}.npz")
            Xtr, Xval, Xtn, Xh = z["Xtr"].astype("float64"), z["Xval"].astype("float64"), z["Xtn"].astype("float64"), z["Xh"].astype("float64")
        except Exception as e:
            fo.write(json.dumps({"key": key, "err": f"load:{e}"}) + "\n"); fo.flush(); continue
        Xte = np.vstack([Xtn, Xh]); yte = np.r_[np.zeros(len(Xtn)), np.ones(len(Xh))]
        ss = StandardScaler().fit(Xtr); Ztr, Zte, Zval = ss.transform(Xtr), ss.transform(Xte), ss.transform(Xval)
        syn1, syn4 = _gen_beta(Xval, 200, 1.0), _gen_beta(Xval, 200, -4.0)
        Zs1, Zs4 = ss.transform(syn1), ss.transform(syn4)
        y1 = np.r_[np.zeros(len(Xval)), np.ones(200)]
        res = {"key": key, "ap": {}, "a1": {}, "a4": {}}

        def probe(nm, sv, s1, s4, st):
            if np.all(np.isfinite(st)) and np.nanstd(st) > 1e-12: res["ap"][nm] = apn(yte, st)
            if np.nanstd(np.r_[sv, s1]) > 1e-12: res["a1"][nm] = float(roc_auc_score(y1, np.r_[sv, s1]))
            if np.nanstd(np.r_[sv, s4]) > 1e-12: res["a4"][nm] = float(roc_auc_score(y1, np.r_[sv, s4]))
        for nm, ctor in _classical_pool():
            try:
                m = ctor(); m.fit(Xtr)
                probe(nm, np.asarray(m.decision_function(Xval), float), np.asarray(m.decision_function(syn1), float), np.asarray(m.decision_function(syn4), float), np.asarray(m.decision_function(Xte), float))
            except Exception: pass
        for nm, ctor in _deep_pool(dev):
            try:
                m = ctor()
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    m.fit(Ztr)
                    probe(nm, np.asarray(m.decision_function(Zval), float), np.asarray(m.decision_function(Zs1), float), np.asarray(m.decision_function(Zs4), float), np.asarray(m.decision_function(Zte), float))
            except Exception: pass
        fo.write(json.dumps(res) + "\n"); fo.flush(); res_vol.commit()   # INCREMENTAL: one row per dataset, committed
        n_written += 1
        print(f"  {key}: {len(res['ap'])} detectors (ap+probe)", flush=True)
    fo.close(); res_vol.commit()
    return n_written


@app.local_entrypoint()
def main(shards: int = 12):
    import pandas as pd, os
    D = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d\scratchpad\streamline"
    idx = pd.read_csv(os.path.join(D, "bundle", "_index.csv"))
    keys = list(idx.key)
    parts = [keys[i::shards] for i in range(shards)]
    parts = [p for p in parts if p]
    total = 0
    for n in run_shard.map(parts):
        total += n
    print(f"DONE: scored {total} datasets across {len(parts)} shards; results in volume adrank-deep2-results")


@app.local_entrypoint()
def fill():
    D = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d\scratchpad\streamline"
    import os
    keys = [k.strip() for k in open(os.path.join(D, "_missing.txt")) if k.strip()]
    print(f"filling {len(keys)} missing datasets")
    n = run_shard.remote(keys)
    print(f"DONE: scored {n} datasets; results in volume adrank-deep2-results")
