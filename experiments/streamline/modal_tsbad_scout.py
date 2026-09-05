# -*- coding: utf-8 -*-
"""Scout the TSB-AD API on Modal: verify install, list available multivariate models, run one on a toy
series to confirm the (data -> per-timestep score) interface before building the full raw-TS harness."""
import modal
APP_NAME = "tsbad-scout"
app = modal.App(APP_NAME)
image = (modal.Image.from_registry("pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel")
         .pip_install("numpy==1.26.4", "pandas", "scikit-learn", "TSB-AD"))


@app.function(image=image, gpu="A10G", timeout=1200)
def scout():
    import numpy as np, traceback
    out = {}
    try:
        import TSB_AD; out["TSB_AD_ver"] = getattr(TSB_AD, "__version__", "?")
    except Exception as e:
        return {"import_err": repr(e)}
    import TSB_AD.model_wrapper as MW
    out["wrapper_attrs"] = [a for a in dir(MW) if "Pool" in a or a.startswith("run_")]
    for pn in ("Semisupervise_AD_Pool", "Unsupervise_AD_Pool"):
        try:
            p = getattr(MW, pn); out[pn] = list(p.keys()) if isinstance(p, dict) else list(p)
        except Exception as e:
            out[pn] = repr(e)[:120]
    # toy multivariate: train on clean normals, test has an injected anomaly window
    rng = np.random.default_rng(0); Xtr = rng.standard_normal((800, 5)).astype("float32")
    Xte = rng.standard_normal((400, 5)).astype("float32"); Xte[180:210] += 6.0
    yte = np.zeros(400); yte[180:210] = 1
    for mdl in ["USAD", "TranAD", "AnomalyTransformer", "TimesNet", "OmniAnomaly", "CNN", "AutoEncoder", "LSTMAD", "Donut"]:
        try:
            s = np.asarray(MW.run_Semisupervise_AD(mdl, Xtr, Xte), float)
            out[f"ok_{mdl}"] = {"shape": list(s.shape), "finite": bool(np.all(np.isfinite(s))), "sep": float(np.nanmean(s[180:210]) - np.nanmean(s[yte == 0]))}
        except Exception as e:
            out[f"err_{mdl}"] = repr(e)[:150]
    return out


@app.local_entrypoint()
def main():
    import json
    r = scout.remote()
    print(json.dumps(r, indent=2, default=str))
