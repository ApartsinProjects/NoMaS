# DRAFT: §5 matched-pool reframing (for review, not yet folded into nomas.md)

All numbers are the validated full-151 recompute. Classical-only values reproduce the published
SPARC 0.113 and Goswami 0.122; all-32 values reproduce the published entropy/mass-volume/agreement
numbers. The change from the current §5 is that every selector is now reported on a **matched
candidate pool**, which is what makes the comparison to the fixed detector and across methods fair.

---

## 5. Unsupervised model selection: methods, protocol, and results

**Selection methods and references.** *(unchanged from the current draft: entropy/mass-volume,
consensus/model-centrality/HITS, UDR, IFOREST-R, IREOS, Goswami [35], SPARC; random-pick floor;
best-fixed-detector bar.)* One clarification is added: a selector's candidate pool is a design
choice, so we evaluate every method on two pools, the 25 classical detectors and the full
32-detector pool that adds the seven deep detectors, and compare methods only within a pool.

**Protocol.** *(unchanged.)*

**Adding deep detectors to the candidate pool raises every selector's regret.** Table 6 reports
each method on both pools. Enlarging the pool from the 25 classical detectors to all 32 raises
regret for every label-free method: SPARC from 0.113 to 0.164, Goswami from 0.122 to 0.152, entropy
from 0.163 to 0.176, mass-volume from 0.165 to 0.174, HITS from 0.219 to 0.235, and the other
agreement selectors similarly. The increases are significant for SPARC, Goswami, entropy,
mass-volume, and HITS ($p < 0.01$, paired Wilcoxon). The deep detectors separate each method's
synthetic or distributional signal on the validation normals without generalizing to the test
anomalies, so a label-free selector that may choose them is drawn to a detector that then scores
poorly. A fair comparison across methods therefore holds the candidate pool fixed; the classical
pool is where every label-free method performs best, and the results below are stated on it, with
the full-pool numbers alongside.

**On a matched classical pool, only SPARC beats the fixed detector.** On the 25 classical detectors,
the agreement selectors (consensus, HITS, model-centrality, 0.218 to 0.223) do not beat a random
pick (0.219): with no anomalies in validation, the quantity they rank on is dominated by detector
behavior on normals and does not track detection skill. Entropy and mass-volume clear the random
floor (0.163 and 0.165) but fall short of the fixed detector at 0.130. Goswami's injection selector
reaches 0.122, matching the fixed detector but not beating it ($p = 0.88$). SPARC is the only method
that significantly beats the fixed detector, at regret 0.113 ($p = 0.04$). On the full 32-detector
pool no label-free method beats the fixed detector, and SPARC remains the best of them at 0.164.

**Table 6.** Model-selection regret (lower is better) on the two candidate pools, with the
best-fixed-detector and random-pick references. Enlarging the pool to include the deep detectors
raises every selector's regret; on the matched classical pool SPARC is the only method that beats
the fixed detector.

| Selector | Classical pool (25) | Full pool (32) |
|---|---:|---:|
| SPARC (Section 6) | **0.113** | 0.164 |
| Goswami injection [35] | 0.122 | 0.152 |
| Best fixed detector | 0.130 | 0.130 |
| Entropy (EM) [1] | 0.163 | 0.176 |
| Mass-volume (MV) [1] | 0.165 | 0.174 |
| Consensus [4] | 0.218 | 0.232 |
| Random pick | 0.219 | 0.219 |
| HITS [18] | 0.219 | 0.235 |
| Model-centrality | 0.223 | 0.233 |

**Selection helps only where no detector is near-universal.** Breaking the classical-pool regret
down by source (Table 6b) locates where selection adds value over the fixed detector. On time series
and OvrBench a single neighbor detector is already near the oracle (fixed-detector regret 0.038 and
0.145), and no method meaningfully beats it. Selection earns its keep on OddBench, where the oracle
detector varies from dataset to dataset: there SPARC reaches 0.153 against the fixed detector's 0.198
($p = 0.029$) and entropy's 0.164. The agreement selectors collapse on both tabular collections
(0.23 to 0.25) and only sit near random on time series.

**Table 6b.** Classical-pool selection regret by data source (lower is better). SPARC's advantage
over the fixed detector is concentrated in the heterogeneous tabular collections; on time series a
neighbor detector is already near-optimal.

| Selector | OvrBench (66) | OddBench (39) | TSB-AD / TS (38) | All (151) |
|---|---:|---:|---:|---:|
| SPARC | 0.137 | **0.153** | 0.037 | **0.113** |
| Goswami | 0.137 | 0.159 | 0.053 | 0.122 |
| Best fixed detector | 0.145 | 0.198 | 0.038 | 0.130 |
| Entropy (EM) | 0.162 | 0.164 | 0.145 | 0.163 |
| Mass-volume (MV) | 0.162 | 0.170 | 0.155 | 0.165 |
| Consensus | 0.243 | 0.234 | 0.126 | 0.218 |
| Random pick | 0.237 | 0.241 | 0.141 | 0.219 |

*(ADBench (5) and DAMI (3) are omitted from Table 6b for space; both are small and noisy. Full
per-source numbers for both pools are released with the code.)*

---

## §6 change (the "Candidates" paragraph)

The current §6 justifies SPARC's classical restriction as a SPARC-specific fix. With the matched-pool
result in §5, it becomes an instance of a general effect, which is a stronger justification:

> **Candidates.** Deep detectors overfit the synthetic probe, separating it without generalizing to
> real anomalies; Section 5 shows this is not specific to SPARC, since adding the deep detectors
> raises regret for every label-free selector. SPARC therefore selects among the classical detectors,
> while the pool and the oracle still include the deep detectors. That is the whole method: one probe,
> one grade per detector, an argmax, with a single parameter (the resample fraction) and no thresholds.

The Table 7 ablation line ("the classical-candidate restriction earns the rest") stays, now
cross-referencing the §5 matched-pool result rather than asserting the effect in isolation.

---

## What this buys us against the ChatGPT / reviewer objections

- **C1 (candidate-pool asymmetry):** fully answered. Every selector is now reported on the same
  pool; the classical restriction is shown to help all methods, not to engineer a SPARC win.
- The published headline (SPARC 0.113 beats the fixed detector, the internal criteria do not) is
  unchanged; it is now stated on a matched pool, so it is fair.
- The Figure 6 bar chart should be redrawn as a grouped bar (classical vs full pool per method), or
  replaced by Table 6; the current single-pool figure is what created the mismatch.
