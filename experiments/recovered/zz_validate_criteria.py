# -*- coding: utf-8 -*-
"""Validate the label-free criteria on synthetic data, expectations stated in advance.

I1 EM/MV directionality: a detector that genuinely separates a planted anomaly must score
   BETTER (higher EM, lower MV) than a random-noise detector.                    [decisive]
I2 consensus/centrality/HITS reward the majority: in a pool of correlated good detectors plus
   a few noise detectors, the noise detectors must get the LOWEST scores.        [decisive]
I3 permutation invariance: shuffling detector order permutes scores identically. [decisive]
I4 a perfect scorer beats a noisy scorer on EM and MV.                           [decisive]
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from label_free_criteria import (consensus_scores, model_centrality_scores,
                                  hits_authority_scores, em_mv_scores)

rng = np.random.default_rng(0)

# ---- synthetic data: normal blob + a compact anomaly cluster ----
d = 6
Xn = rng.normal(0, 1, (600, d))
Xa = rng.normal(0, 1, (30, d)) + 6.0          # clearly displaced anomalies
X = np.vstack([Xn, Xa])
y = np.r_[np.zeros(600), np.ones(30)]
center = Xn.mean(0)

def good(A):   # distance to normal centre - a real detector
    return np.linalg.norm(A - center, axis=1)
def perfect(A):  # near-oracle: distance, sharpened
    return np.linalg.norm(A - center, axis=1) ** 3
def noise(A):
    return rng.random(len(A))

print("=== I1 / I4  EM & MV directionality ===")
em, mv = em_mv_scores([perfect, good, noise], X, n_generated=20000, seed=1)
for nm, e, m in zip(["perfect", "good", "noise"], em, mv):
    print(f"  {nm:8s} EM={e:.4f} (higher better)   MV={m:.4f} (lower better)")
i1 = em[1] > em[2] and mv[1] < mv[2]
i4 = em[0] >= em[1] and mv[0] <= mv[1] + 1e-9
print(f"  I1 good beats noise (EM up, MV down): {'PASS' if i1 else 'FAIL'}")
print(f"  I4 perfect >= good:                   {'PASS' if i4 else 'FAIL'}")

print("\n=== I2  consensus / centrality / HITS down-weight noise ===")
# pool: 5 correlated good detectors (distance-based, jittered) + 2 pure-noise detectors
S = []
base = good(X)
for k in range(5):
    S.append(base + rng.normal(0, 0.05 * base.std(), len(X)))   # good, correlated
for k in range(2):
    S.append(rng.random(len(X)))                                # noise
S = np.column_stack(S)   # [n, 7]; cols 0-4 good, 5-6 noise
for nm, fn in [("consensus", consensus_scores),
               ("centrality", model_centrality_scores),
               ("HITS", hits_authority_scores)]:
    sc = fn(S)
    good_min = sc[:5].min(); noise_max = sc[5:].max()
    ok = good_min > noise_max
    print(f"  {nm:11s} good=[{sc[:5].min():.3f},{sc[:5].max():.3f}]  "
          f"noise=[{sc[5:].min():.3f},{sc[5:].max():.3f}]  "
          f"{'PASS' if ok else 'FAIL'}")

print("\n=== I3  permutation invariance ===")
perm = rng.permutation(S.shape[1])
base_sc = model_centrality_scores(S)
perm_sc = model_centrality_scores(S[:, perm])
i3 = np.allclose(base_sc[perm], perm_sc)
print(f"  centrality permutation-invariant: {'PASS' if i3 else 'FAIL'}")

print("\n=== picks (what each selector would choose in the mixed pool) ===")
for nm, fn in [("consensus", consensus_scores), ("centrality", model_centrality_scores),
               ("HITS", hits_authority_scores)]:
    pick = int(np.argmax(fn(S)))
    print(f"  {nm:11s} picks detector #{pick}  ({'GOOD' if pick < 5 else 'NOISE'})")
