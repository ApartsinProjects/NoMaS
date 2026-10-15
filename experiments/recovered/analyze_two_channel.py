# -*- coding: utf-8 -*-
"""Analyse the two-channel scores: do the channels separate detector families?

Reads HADB_TWO_CHANNEL.csv (per dataset x model: edge_auc, embedded_auc, true_ap_norm).
Aggregates per model and per family and asks whether GLOBAL detectors score high on the EDGE
channel (peripheral pseudo-anomalies) and LOCAL detectors high on the EMBEDDED channel.
"""
import os
import numpy as np
import pandas as pd

S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
D = pd.read_csv(os.path.join(S, "HADB_TWO_CHANNEL.csv"))

LOCAL = ("LOF", "KNN", "CBLOF")
GLOBAL = ("HBOS", "COPOD", "ECOD", "PCA")


def fam(v):
    v = str(v)
    if v.startswith(LOCAL):
        return "local"
    if v.startswith(GLOBAL):
        return "global"
    return "other"   # IForest / IF_ / LODA


D["family"] = D.variant.map(fam)
# per-model means across datasets
per = D.groupby(["variant", "family"], as_index=False).agg(
    edge=("edge_auc", "mean"), emb=("embedded_auc", "mean"),
    n=("edge_auc", "size"), true=("true_ap_norm", "mean"))
per["delta"] = per.emb - per.edge     # >0 = better at embedded (local-like); <0 = better at edge

print("=== per-family channel means (does edge->global, embedded->local?) ===")
fg = per.groupby("family").agg(edge=("edge", "mean"), emb=("emb", "mean"),
                               delta=("delta", "mean"), n_models=("variant", "size"))
print(fg.round(3).to_string())
print("\n  delta = embedded_AUC - edge_AUC. Positive => the family is better at separating")
print("  EMBEDDED (inside/between) pseudo-anomalies, i.e. it behaves like a LOCAL detector.")

print("\n=== the detectors most EDGE-leaning (global-like) ===")
print(per.sort_values("delta").head(6)[["variant", "family", "edge", "emb", "delta"]].round(3).to_string(index=False))
print("\n=== the detectors most EMBEDDED-leaning (local-like) ===")
print(per.sort_values("delta").tail(6)[["variant", "family", "edge", "emb", "delta"]].round(3).to_string(index=False))

# does channel-leaning predict which family actually wins on real anomalies?
print("\n=== correlation of channel scores with true test ap_norm (per model, across datasets) ===")
for ch in ["edge_auc", "embedded_auc"]:
    r = D[[ch, "true_ap_norm"]].dropna().corr().iloc[0, 1]
    print(f"  corr({ch}, true_ap_norm) = {r:.3f}")
print(f"  models: {per.variant.nunique()}  |  rows: {len(D)}  |  datasets: {D.dataset.nunique()}")
