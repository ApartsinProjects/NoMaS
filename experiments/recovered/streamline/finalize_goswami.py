# -*- coding: utf-8 -*-
"""Fold the Goswami injection-selector choices into regret, on the same 151 in-benchmark tasks.
For each task, the selected classical detector's real-test ap_norm comes from the cached classical
results (deepres_v2_clstd), and regret = oracle_ap (full pool, from the leaderboard) - ap[selected].
Reports overall + per modality, with paired Wilcoxon vs SPARC. Writes GOSWAMI_REGRET.csv."""
import os, json, glob
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
D = os.path.dirname(os.path.abspath(__file__))
CLS = os.path.join(D, "deepres_v2_clstd")


def load_cls_ap():
    J = {}
    for fn in glob.glob(os.path.join(CLS, "shard_*.jsonl")):
        for ln in open(fn):
            try:
                r = json.loads(ln)
                if "ap" in r:
                    J[r["key"]] = r["ap"]
            except Exception:
                pass
    return J


def main():
    lb = pd.read_csv(os.path.join(D, "STREAM_V3_LEADERBOARD.csv"))
    keep = lb[(lb.oracle_ap > 0.05) & (lb.reg_random > 0.02) & (lb.n_h >= 100)].copy()
    oracle = dict(zip(keep.key, keep.oracle_ap))
    apc = load_cls_ap()
    sel = {}
    for ln in open(os.path.join(D, "GOSWAMI_INJ.jsonl")):
        try:
            r = json.loads(ln)
            if r.get("sel"):
                sel[r["key"]] = (r["sel"], r.get("sel_mean"))
        except Exception:
            pass
    rows = []
    for k in keep.key:
        if k not in sel or k not in apc:
            continue
        s_borda, s_mean = sel[k]
        if s_borda not in apc[k]:
            continue
        reg_b = oracle[k] - apc[k][s_borda]
        reg_m = oracle[k] - apc[k].get(s_mean, apc[k][s_borda])
        rows.append(dict(key=k, corpus=k.split("__")[0], reg_goswami=reg_b, reg_goswami_mean=reg_m))
    g = pd.DataFrame(rows).merge(keep[["key", "reg_combined_cls", "reg_em", "reg_mv", "reg_random"]], on="key")
    g.to_csv(os.path.join(D, "GOSWAMI_REGRET.csv"), index=False)
    print(f"matched {len(g)}/{len(keep)} in-benchmark tasks")

    def rep(x, lab):
        a = x.reg_combined_cls.values; gg = x.reg_goswami.values
        p = wilcoxon(a, gg).pvalue if not np.allclose(a, gg) else np.nan
        print(f"[{lab:11s} n={len(x):3d}] Goswami-inj {gg.mean():.3f} (Borda)  mean-agg {x.reg_goswami_mean.mean():.3f}  "
              f"| SPARC {a.mean():.3f} (p={p:.4f})  EM {x.reg_em.mean():.3f}  MV {x.reg_mv.mean():.3f}  rand {x.reg_random.mean():.3f}")
    rep(g, "ALL")
    for cp in ["tsbad_m", "oddbench", "ovrbench"]:
        x = g[g.corpus == cp]
        if len(x) >= 5:
            rep(x, cp)


if __name__ == "__main__":
    main()
