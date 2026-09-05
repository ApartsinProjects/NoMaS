# -*- coding: utf-8 -*-
"""(b) Raw-sequence TS deep detectors on the 41 MTS datasets via TSB-AD. Per dataset: train on the
clean-normal prefix, run each semisupervised model on the test series, aggregate per-timestep scores to
our windows (W=64, stride=16, max-pool), compute ap_norm vs window labels. IForest (unsupervised, raw)
as the reference. Incremental per-dataset writes + resume. Compare to window-feature classical later."""
import modal
APP_NAME = "adrank-tsbad"
app = modal.App(APP_NAME)
image = (modal.Image.from_registry("pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel")
         .pip_install("numpy==1.26.4", "pandas", "scikit-learn", "TSB-AD"))
data_vol = modal.Volume.from_name("adrank-rawmts", create_if_missing=True)
res_vol = modal.Volume.from_name("adrank-tsbad-results", create_if_missing=True)
W, STRIDE = 64, 16
SEMI = ["USAD", "TranAD", "AnomalyTransformer", "OmniAnomaly", "TimesNet", "LSTMAD"]


@app.function(image=image, gpu="A10G", timeout=3600 * 4, volumes={"/data": data_vol, "/results": res_vol})
def run_shard(keys):
    import os, io, contextlib, json, warnings
    import numpy as np
    from sklearn.metrics import average_precision_score
    import TSB_AD.model_wrapper as MW
    warnings.filterwarnings("ignore")

    def win_labels_scores(lab, ts_score):
        starts = np.arange(0, len(lab) - W + 1, STRIDE)
        yl = np.array([int(lab[s:s + W].sum() >= 1) for s in starts])
        sc = np.array([np.nanmax(ts_score[s:s + W]) for s in starts])
        return yl, sc

    def apn(y, s):
        m = np.isfinite(s)
        if m.sum() < 10 or y[m].sum() < 3 or y[m].sum() == m.sum(): return None
        bb = y[m].mean(); return float((average_precision_score(y[m], s[m]) - bb) / (1 - bb + 1e-12))
    outpath = f"/results/shard_{abs(hash(tuple(keys))) % 10**8}.jsonl"
    done = set()
    if os.path.exists(outpath):
        for ln in open(outpath):
            try: done.add(json.loads(ln)["key"])
            except Exception: pass
    fo = open(outpath, "a"); n = len(done)
    for key in keys:
        if key in done: continue
        try:
            z = np.load(f"/data/r/{key}.npz"); Xc = z["Xc"].astype("float64"); lab = z["lab"].astype(int)
        except Exception as e:
            fo.write(json.dumps({"key": key, "err": f"load:{e}"}) + "\n"); fo.flush(); continue
        a = np.where(lab == 1)[0]
        tr_end = int(a[0]) if len(a) and a[0] >= 800 else max(800, int(0.4 * len(Xc)))   # clean-normal prefix
        tr_end = min(tr_end, len(Xc) - 200)
        Xtr = Xc[:tr_end][lab[:tr_end] == 0]; Xte = Xc[tr_end:]; lab_te = lab[tr_end:]
        if len(Xtr) < 300 or len(Xte) < W + STRIDE or lab_te.sum() < 5:
            fo.write(json.dumps({"key": key, "err": "split"}) + "\n"); fo.flush(); continue
        res = {"key": key, "ap": {}, "n_test": int(len(Xte)), "n_anom_test": int(lab_te.sum())}
        # reference: unsupervised IForest on raw
        try:
            s = np.asarray(MW.run_Unsupervise_AD("IForest", Xte), float)
            yl, sc = win_labels_scores(lab_te, s); v = apn(yl, sc)
            if v is not None: res["ap"]["IForest_raw"] = v
        except Exception: pass
        for mdl in SEMI:
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    s = np.asarray(MW.run_Semisupervise_AD(mdl, Xtr, Xte), float)
                yl, sc = win_labels_scores(lab_te, s); v = apn(yl, sc)
                if v is not None: res["ap"][mdl] = v
            except Exception as e:
                res.setdefault("err_models", {})[mdl] = repr(e)[:100]
        fo.write(json.dumps(res) + "\n"); fo.flush(); res_vol.commit()
        n += 1; print(f"  {key}: {len(res['ap'])} models", flush=True)
    fo.close(); res_vol.commit()
    return n


@app.local_entrypoint()
def main(shards: int = 8):
    import pandas as pd, os
    D = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d\scratchpad\streamline"
    keys = list(pd.read_csv(os.path.join(D, "rawbundle", "_index.csv")).key)
    parts = [keys[i::shards] for i in range(shards)]; parts = [p for p in parts if p]
    total = 0
    for x in run_shard.map(parts): total += x
    print(f"DONE: scored {total} MTS datasets with raw-sequence deep detectors")


@app.local_entrypoint()
def smoke():
    import pandas as pd, os
    D = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d\scratchpad\streamline"
    keys = list(pd.read_csv(os.path.join(D, "rawbundle", "_index.csv")).key)[:2]
    print("SMOKE:", run_shard.remote(keys), "datasets scored")
