# Streamlined HADB construction (ADOPTED as the primary benchmark, 2026-09-04)

**Date:** 2026-09-04. A single, principled, detector-free pipeline that rebuilds the benchmark FROM
SOURCE, replacing the accumulated ad-hoc filters (max|z|, Wu-Keogh, per-dataset rule, low-spread) of
the earlier build. **DECISION (2026-09-04): this streamlined 176-dataset benchmark is ADOPTED as the
paper's PRIMARY benchmark.** The earlier 199-dataset build is retained as the prior/robustness version
(the headline - NoMaS beats the UOMS field incl. EM/MV/IFOREST-R - reproduces on both, so the result is
not a filtering artifact). All code + result CSVs live in `experiments/streamline/`.

## Naming & terminology (STANDARDIZED 2026-09-05 - use these in the paper)

**Method: SPARC** = Synthetic-Probe Anomaly-detector Routing by Correlation-spectrum. Selects a detector
label-free by (a) probing detectors with synthetic anomalies (local-type beta=+1 / global-type beta=-4,
graded by separation **ap_norm** vs val normals - the AP-probe beats the AUC-probe on the routed selectors,
matched 0.137->0.128 / combined 0.129->0.126 macro on the classical pool; use AP), (b) measuring the normal
data's correlation spectrum, and (c)
routing to the local-best detector if the spectrum is anisotropic OR local_ev is high, else the global-best.
Configs: SPARC-beta1 (fixed local probe), SPARC-matched (route by local_ev), SPARC-corr (route by corr_str),
SPARC-combined (route by corr_str OR local_ev, the headline).

**The routing axis is a DATASET property, NOT an anomaly-type label (we never observe anomalies at selection
time).** Name it by the measurable normal-data geometry: **ANISOTROPIC <-> ISOTROPIC** (variance concentrated
in a few correlated directions / thin manifold  <->  spread evenly / round ball). Measured label-free by
corr_str (mean|correlation|, AUC 0.73 for the family), eff_dim (participation dim: local 4.5 vs global 9.1),
top1_var. This is a CAUSAL CHAIN, only the first link observable: (1) dataset ANISOTROPIC vs ISOTROPIC ->
(2) which DETECTOR FAMILY wins: LOCAL (neighbor: LOF/KNN/CBLOF) vs GLOBAL (marginal: HBOS/COPOD/ECOD/PCA)
[keep local/global - standard detector terms] -> (3) the likely ANOMALY TYPE: DEPENDENCY/joint vs MARGINAL
[a consequence, never observed; use only when explaining the mechanism]. So: axis = anisotropic/isotropic
(dataset), families = local/global (detectors), anomaly type = dependency/marginal (mechanism only).

## Update 2026-09-05: Stage 0 point-anomaly filter, richer metrics, deep detectors, alpha spectrum

**Stage 0 (NEW) - drop point/short-anomaly TS series.** A series whose anomaly content is <50% in
segments >= W/2 (=32) raw steps cannot form a majority-anomaly window at W=64, so W-windowing yields
only diluted near-normal "anomaly" labels (verified: CreditCard median window-purity 0.016, 83% single
raw point; GECCO 0.33). Detector-free, computed from raw label runs (`is_point_anom` in
`pipeline_final.py`). Drops exactly 2 datasets (CreditCard, GECCO), 176 -> **174** (tsbad_m 43 -> 41),
zero collateral. Chosen over a per-window purity relabel, which collaterally shrank n_eff on 3 healthy
datasets (+12 lost >30% n_eff) for only a mild ceiling gain. Root-caused via a labeling-artifact check
(the user's instinct): only 2/43 MTS datasets are affected; the other 41 are genuine segment anomalies.

**Filtered leaderboard (173 kept, 101 local / 72 global).** NoMaS matched **beats MV on BOTH micro
(0.143 vs 0.153) and macro (0.146 vs 0.150), p=0.010**, and beats EM (0.165, p=0.003); crushes
consensus/MC/HITS/random (~0.23). The prior tie with MV on macro flips to a win once the 2 mislabeled
point-anomaly datasets are removed.

**Report distribution, not just mean (regret is right-skewed).** median regret: matched **0.071** vs MV
0.089 vs EM 0.091 (mean 0.14x for all - a shared hard tail inflates the mean equally). %-oracle (picked
the exactly-best detector): matched 19%, beta=1 **25%**, MV/EM 13%. %-badly-wrong (>0.3): matched 13%,
MV 17%. Put median + p90 + %oracle + %>.3 + the ECDF in the leaderboard, not mean alone. The Wilcoxon is
already rank-based so it is unaffected. Max regret 0.939 is shared by all methods (datasets no label-free
heuristic solves). Tail inspection (co-computed): the high-regret datasets are genuine hard SELECTION
(a real cluster of good detectors we missed), NOT mislabeled - two failure modes, wrong-family and
right-family-wrong-member; the global half is MV-fixable, the local half fails for MV too.

**Deep detectors on all 173 (Modal, A10G, 25 classical + 7 deep co-computed on the same hardened test;
`modal_deep.py`, `STREAM_DEEP_MODAL.csv`).** DeepSVDD is the **best single detector** (mean ap_norm
0.161, edging LOF_k10 0.151). But no single deep model matches classical diversity: best-classical
oracle 0.290 > best-deep 0.237; combined 0.328 (deep complementary, +0.038 ceiling, beats best classical
on 32%). All-bad gate: of 50 datasets all-bad by classical (best<=0.10), deep rescues **12** above 0.10
-> "all-bad" is pool-relative, keep those. Extends ADBench/TSB-AD "simple is competitive" into the
hard-anomaly regime. Reusable pipeline (bundle on modal volume + harness).

**Full-spectrum synthetic generation + alpha (`gen_alpha_selector.py`).** Anomaly type = single-value
weird (marginal, global detectors) <-> combination weird (voids, local detectors), set by how the
deviation is spread across features at fixed total weirdness. Generate-at-alpha: one generator, alpha
in [0,1] (alpha=0 one feature at a tail value; alpha=1 all features at real permuted values). Result:
the **best-alpha ORACLE cuts regret 0.141 -> 0.077 (nearly half)** - the correctly-typed probe exists on
the spectrum for most datasets (framework validated). BUT label-free alpha selectors (max-disagreement,
alpha-from-local_ev) TIE binary matched (0.141), not reaching it. **Open problem = estimating alpha
(anomaly type) from normal data**: structural void measures (silhouette AUC 0.62, joint-multimodality
0.61) confirm the user's "unimodal->marginal, multimodal-voids->combination" intuition in DIRECTION but
only ~0.6; local_ev ~0.79 but as a continuous alpha does not beat binary. Marginal generator is
global-biased, permutation generator local-biased; an OOD drop-in-cluster filter does not fix the tail
(the issue is type-bias, not in-distribution contamination - synthetics are 1-7% normal-like). All
selection stays label-free (normals + synthesized-from-normals); labels only score results + mark oracles.

## Update 2026-09-05b: deep in the selection pool (a), raw-TS deep (b), synthetic-source + band tests

**(a) Deep detectors in the SELECTION pool (Modal, ap+a1+a4 co-computed per detector; `modal_deep.py`,
`STREAM_EXPANDED_SEL.csv`, 173 datasets).** Our label-free matched selector EXTENDS cleanly to deep:
when a deep detector is the oracle (56/173, 32%) it picks deep 43/56 = **77%** of the time (the a1/a4
probe generalizes to DeepSVDD/AE), and does slightly better on deep-oracle datasets (regret 0.157) than
classical-oracle ones (0.174). But adding deep to the pool does NOT change net selection: regret-vs-full-
oracle 0.168 with deep vs 0.167 ignoring deep (p=0.565) - the oracle rises +0.038 and we capture the same
fraction. So deep is worth adding for the DETECTOR benchmark (higher ceiling), neutral for SELECTION.

**(b) Raw-sequence deep on MTS via TSB-AD (`modal_tsbad.py`, `STREAM_TSBAD.csv`, 34 datasets, TSB-AD
native temporal-split protocol - NOT construct-matched to the tabular hardened pipeline).** Best raw
model **OmniAnomaly 0.354**, then LSTMAD 0.261, IForest_raw 0.255, USAD/TranAD ~0.21; **transformers
underperform** (TimesNet 0.095, AnomalyTransformer 0.073 - below raw IForest), reproducing TSB-AD's
headline. A matched raw-vs-window comparison (same split/labels) is the clean follow-up.

**Synthetic SOURCE (`STREAM_SRC_COMPARE.csv`): val is correct, train is worse.** matched regret: val-
synth/val-ref **0.144** < train-synth/train-ref 0.153 < train-synth/val-ref 0.159; val wins even on the
low-n_val half (0.149 vs 0.171). Reason: detectors are fit on train, so train-synthetics probe memorized
points (artifact), while val-synthetics probe UNSEEN normal structure (what predicts test). Not a coverage
issue -> keep synth from val. The band edges are detector-free (severity quantile + NN scale) so they can
be calibrated on train without leak, but this is moot given val-source wins.

**Band-filtered alpha (`STREAM_GENALPHA_BAND.csv`): does not close the estimator gap.** Keeping synthetics
in the difficulty band [OOD floor, hardening ceiling] (both label-free from normals) leaves the deployable
selectors at binary-matched level (max-disagreement 0.144) while the best-alpha ORACLE stays at ~0.074.
So difficulty-normalization was not the missing piece. STATE OF THE ALPHA LINE: framework proven (oracle
~0.074-0.077, ~half the binary regret, robust across band/no-band), but NO label-free alpha estimator
(max-disagreement, local_ev-alpha, structural void measures, band-normalized) reaches it - the anomaly-
type estimator is the genuine open problem; binary matched (0.141-0.146, beats MV) is the current best.

## Update 2026-09-05c: alpha-estimator attempts, few-shot upper bound, void structure (alpha line CLOSED)

**Alpha estimator (`alpha_estimator.py`, 173 datasets).** Five estimators vs the best-alpha oracle
(0.075/0.077): binary matched 0.138/0.145, max-disagreement 0.142/0.149, MARGIN 0.144/0.151, CONSENSUS
0.144/0.146, LEARNED-alpha LOO random-forest on label-free features (corr-void, silhouette, PC1-kurtosis,
d, local_ev) 0.138/**0.138** (best deployable MACRO, +0.007 - a faint learnable signal, tiny). None nears
the oracle; anomaly-TYPE is under-determined by the normals.

**Few-shot upper bound (`alpha_semisup.py`): 10% labeled anomalies calibrate, regret on held-out 90%.**
binary matched (unsup) 0.135; alpha-via-10% 0.088; **direct-10% (measure each detector's ap on the 10%
anomalies, pick best - no synthetics) 0.018** (median 70 calib anomalies). A sliver of supervision nearly
solves selection, and routing THROUGH the synthetic probe HURTS once labels exist (0.088 vs 0.018).
Positioning: NoMaS/synthetic-alpha is the best tool for the ZERO-label regime (beats UOMS); with >=10%
labels use them directly on detectors. The zero-label regime is where the hard problem lives.

**Void structure explains the estimator failure (`void_analysis.py`).** best-alpha UNCORRELATED with every
void/placement measure: void_mass (single-Gaussian mass in mixture-void regions, user's measure) rho -0.05,
log-lik gap -0.06, n_modes -0.03, void_fill -0.03, hull_ext +0.02, embedded +0.02 (all p>0.4). Structure
EXISTS - normals very multimodal (void_mass 0.82, 7.7 modes), anomalies mostly void-filling (51% void-fill,
27% embedded, 22% hull-exterior) - but the winning anomaly TYPE is structurally decoupled from it. Coarse
family weakly predictable (local_ev AUC 0.79); finer best-alpha carries no geometric signal. ALPHA LINE
(complete arc): framework proven -> estimator fails -> WHY (best-alpha decoupled from void geometry) ->
few-shot blows past it (0.018). Publishable: positive method + characterized/explained open problem + few-
shot upper bound.

## Update 2026-09-05d: FINAL unified leaderboard (full classical+deep pool) + spectrum router

**Local-vs-global is a SPECTRUM-SHAPE property (`mine_localglobal.py`, `STREAM_MINE.csv`, 173).** Mined ~20
label-free normal-data stats. Strongest family discriminators: **corr_str (mean|correlation| AUC 0.73),
top1_var (PC1 variance 0.72), eff_dim (participation dim, 0.71 for global: local 4.5 vs global 9.1),
eigengap 0.69, silhouette 0.69**. MULTIMODALITY (void_mass/n_modes/ll_gap) all ~0.50 = NOT discriminative.
Mechanism: LOCAL = low-dim, correlated, anisotropic (steep spectrum, thin manifold) -> off-manifold
combination anomalies -> neighbor detectors; GLOBAL = high-dim, isotropic (flat spectrum, round ball) ->
marginal anomalies -> histogram detectors. corr_str (0.73) is a pure normal-data stat ~matching the
synthetic probe local_ev (0.79). Effective-dimension (participation ratio of the covariance spectrum) is
the discriminating "dimensionality", NOT ambient d (AUC 0.46) or two-NN intrinsic manifold dim (0.48).

**SPECTRUM ROUTER upgrades the method (`rank_spectrum.py` classical; `finalize_leaderboard.py` full pool).**
Route local-vs-global by corr_str instead of / combined-with local_ev. Classical pool (construct-matched):
**combined (corr_str OR local_ev) 0.132/0.137 macro beats MV (p=0.002) and EM (p=0.001)** and the old
matched (0.148); corr_str-alone 0.143 also beats matched.

**FINAL UNIFIED LEADERBOARD (`finalize_leaderboard.py`, `STREAM_FINAL_LEADERBOARD.csv`,
`FIG_final_leaderboard.png`): full CLASSICAL+DEEP pool, all selectors construct-matched, 173 datasets
(104L/69G).** Macro regret: **NoMaS combined 0.156 (micro 0.149) - beats MV p=0.021, EM p=0.036, the only
significant winner**; NoMaS corr_str 0.168; NoMaS beta=1 0.170; EM 0.172; NoMaS matched 0.173 (old matched
now ~ties EM on the expanded pool - the spectrum router is what restores the win); MV 0.179; UDR 0.213;
IFOREST-R 0.218; random 0.248; HITS/consensus/MC 0.26-0.27 (below random, degenerate on normals-only val).
Deep raises the oracle +0.038 (classical 0.290 -> combined 0.328) so all regrets grow vs the classical-pool
table but the ranking holds. Detectors: DeepSVDD best single (0.161), classical diversity wins the oracle;
raw-TS deep (OmniAnomaly 0.354, transformers underperform) reported as a SEPARATE MTS-modality table (TSB-AD
native protocol, not construct-matched to the hardened tabular test). OPEN baseline: IREOS (only UOMS method
not yet computed). Score-saving Modal harness on volume adrank-deep3-results (val+uniform per detector).

## Migration checklist (to finish promoting streamlined -> primary)
- [x] Selection pipeline + final set (`STREAM_FINAL2_SET.csv`, 176 datasets)
- [x] Full EDA (`STREAM_EDA2_ALL.csv`): sizes, diversity, marginal+joint multimodality, solvability, family
- [x] Selector leaderboard + all UOMS baselines incl. faithful IFOREST-R and UDR (`STREAM_RANK/IFR/UDR.csv`)
- [x] Pseudo-anomaly control (fails -> the win is OOD synthesis)
- [ ] Save per-detector ground-truth ap_norm on the hardened test as a canonical results CSV (rank.py
      currently keeps regrets only; regenerate a per-variant table for full reproducibility)
- [ ] Regenerate paper tables/figures from the streamlined numbers (the canonical FIG_leaderboard etc.
      become the prior-version appendix); primary leaderboard fig = `experiments/figs/FIG_stream_leaderboard.png`
- [ ] Zenodo/data-availability: package `STREAM_FINAL2_SET` + `STREAM_EDA2_ALL` as the released benchmark

## Pipeline (5 stages, all feature-space, no detector fitting)

Philosophy shift: instead of DROPPING whole datasets that look easy, HARDEN each dataset in place
(remove trivially-caught anomalies) and keep it if enough genuinely-hard, distinct anomalies remain.

1. **Stage 1 - drop OR-solvable datasets** (hardening rule catches >= 90% of anomalies).
2. **Stage 2a - marginal hardening**: per-original-feature OR, severity = max(ECDF two-sided tail,
   histogram -log-density) => edges + INTERIOR multimodal gaps; threshold at 5% normal FP.
3. **Stage 2b - joint hardening**: same OR in PCA-whitened space (95% var) => oblique/principal-direction
   separations the original axes miss; 5% normal FP. Applied as a SEPARATE stage, UNION with 2a
   (an anomaly is trivial if either flags it). NB merging 2a+2b into one rule and recalibrating to 5%
   FP BACKFIRES (multiple-comparison threshold inflation: catch 0.22, fixes 1/21 suspects); the
   sequential union is correct (catch 0.31, fixes 9/21).
4. **Stage 3 - keep >= 100 DISTINCT hard anomalies** (n_eff = greedy radius-cover at the normal NN
   scale). Distinct-count, not raw count: e.g. internet_firewall has 3028 hard anomalies but only 9
   distinct (near-duplicates) -> correctly dropped.
5. **Stage 4 - dedup** by data fingerprint (dim + sorted normal feature-moments); catches same-source
   datasets across corpora (mostly overlapping MTS series). adbench/dami vs oddbench/ovrbench are
   anomaly-MODIFIED variants, not identical, so correctly NOT merged.
6. **Stage 5 - keep >= 800 normals** (held-out) to model the normal distribution.
- **No base-rate cap**: anomalies need not be rare; a 1000-normal/1000-anomaly dataset is valid as long
  as normals model well and anomalies are enough+distinct for a robust ap_norm estimate.

## Result: 176 datasets (`STREAM_FINAL2_SET.csv`)

ovrbench 76, oddbench 46, MTS 43, adbench 7, dami 4 (tabular + multivariate-TS). Univariate TS (UCR,
tsbad_u) excluded: their single contiguous anomaly region yields <100 DISTINCT (mostly overlapping)
anomaly windows, so they cannot support robust evaluation.

EDA profile (`STREAM_EDA2_ALL.csv`, 176 datasets, all stats co-computed):
- **Well-modeled**: median 2400 normals (all >= 800).
- **Robustly evaluable**: median 371 distinct hard anomalies, eff_frac 0.54.
- **Solvable (validated by scoring the pool on the hardened sets)**: oracle ap_norm median 0.22, 71%
  > 0.1, only 6 unsolvable. Two-stage is HARDER than single-stage (was 0.33 / 82%) by design - it
  removes the oblique multivariate-trivial anomalies.
- **Family**: 101 local / 74 global (58/42), median base_rate 0.37.
- **Multimodal**: 0.71 of features marginally multimodal (Hartigan dip); 0.51 jointly multimodal
  (discriminant-axis dip); median 12 GMM BIC components. (Random-projection dip is a CLT artifact -
  do not use it; the discriminant-direction dip is the valid joint test.)

## Validation findings (figures in `experiments/figs/`)

- **Dropped-for-easy-separation** (OR_solvable, `FIG_stream_orsolvable.png`): anomalies form clearly
  separated clusters/regions - correctly removed.
- **Kept datasets with separable clusters** (`stream_suspects.py`): the marginal Stage-2a is blind to
  JOINT separation, so ~21/182 kept datasets have anomalies a simple kNN aces (multivariate-trivial;
  every feature in-band but the combination is far from normal). Stage-2b (PC-OR) catches ~half of
  these (survivor kNN 0.85 -> 0.55, 9-10/21 fixed). Residual ~12 hide in low-variance PCs or are
  nonlinearly separable; catching them would require a detector-defined rule, deliberately not done.
- **Dropped-but-embedded** anomalies were dropped for FEW-DISTINCT (n_eff<100), a count issue, not
  easiness (56/82 few-distinct drops are embedded/hard with frac_triv<0.3).

## Selector leaderboard + faithful UOMS baselines (streamlined benchmark, 175 datasets)

Construct-matched on the two-stage hardened test (`rank.py`, `ifr.py`, `udr.py`; regret on ap_norm,
macro = family-balanced):
- **NoMaS matched (ours) 0.149**, MV 0.148, **NoMaS beta=1 (ours) 0.151**, EM 0.159 - all beat random.
- **UDR 0.216** (seed-stability on the iForest sub-pool) - NOT significantly different from random
  (p=0.27), exactly Ma et al's finding. **IFOREST-R 0.218** (faithful 81-config average, n_estimators x
  max_features). consensus/ModelCentrality/HITS 0.222-0.225. random 0.228.
- **Our methods beat IFOREST-R and UDR at p<0.001** (matched beats IFOREST-R on 113/175 datasets).
This COUNTERS Ma et al ("none significantly different from random model selection ... all significantly
worse than random-config iForest"): on a hard-anomaly, local-vs-global benchmark with a diverse pool,
selection genuinely beats the iForest bar. UDR is ill-defined on the full mixed pool (deterministic
detectors LOF/KNN/HBOS/COPOD/ECOD/PCA have perfect seed-stability), so it is run faithfully on the
stochastic iForest sub-pool only.

**Pseudo-anomaly selection retried (`pseudo_cluster.py`) - fails again, confirming the mechanism.**
On the streamlined (strongly cluster-structured: 12 GMM modes) benchmark: smallest-cluster pseudo-
anomalies within-dataset rho -0.06 (no signal), EDGE pseudo-anomalies rho -0.20 (ANTI-correlated,
regret 0.233 > random), vs beta=1 synthetic rho +0.284 (regret 0.143). Holding out IN-distribution
normal points measures normal-structure separation, not anomaly detection - even on cluster-structured
data. The win is specifically OUT-OF-distribution synthesis, not any anomaly proxy. Result CSVs:
STREAM_RANK/IFR/UDR/PSEUDO.csv.

## Reproduce

`experiments/streamline/pipeline_final.py` (definitive selection) -> `STREAM_FINAL2_{ALL,SET}.csv`;
`score2.py` (solvability+family); `stream_multimodality.py` (marginal+joint modality); `rank.py`
(selector leaderboard). Analysis: `stream_suspects.py`, `stream_pca_or.py`, `stream_hist_bins.py`,
`stream_combined.py`/`stream_sequential.py`. Viz: `viz_grid.py`, `viz_dropped.py`, `make_eda.py`.

## Update 06 (2026-09-05): EDA visuals corrected - anomaly-geometry claims refuted, type signal is in the NORMAL data

Building anomaly-type EDA figures surfaced two claims that FAIL a direct data check; both are dropped.

1. **"Anisotropic anomalies sit OFF the correlation line" - REFUTED.** `probe_offline.py`: for the
   top-25 anisotropic (local-family, high corr_str) datasets, the anomalies' median perpendicular
   residual off the normals' correlation axis, over the normals' 95th percentile, is <1.0 for EVERY
   dataset (max 0.78). Hard anomalies RESPECT the correlations; there is no dependency-violation
   signature. The retired `FIG_mechanism.png` asserted the opposite and was wrong.

2. **"Anisotropic anomalies need a joint view (multivariate >> marginal)" - REFUTED.**
   `probe_margjoint.py` (STREAM_MARGJOINT.csv, n=52 with clean family + enough hardened anoms):
   best-single-feature (oracle) AUC vs best unsupervised multivariate detector AUC gives NEGATIVE
   joint gain for both families (local -0.036, global -0.089), and the family difference is NS
   (MWU p=0.11). An oracle single feature separates as well or better than the multivariate detectors
   for both types.

3. **Unsupervised 2D embeddings do not separate hard anomalies.** `eda_embed_explore.py`: PCA, t-SNE,
   and UMAP (joint-embedded normals+anoms) all leave the hardened anomalies scattered through the
   normals. These anomalies are near-normal by construction; any 2D view that made them "pop" would be
   an artifact. Retired the PCA anomaly-scatter column accordingly.

**What IS supported (kept, `eda_type.py` -> `FIG_type_separation.png`).** The dataset-type signal lives
in the NORMAL-data structure, not in an anomaly-geometry picture. Label-free anisotropy stats separate
the winning detector family at AUC ~0.65-0.67 on the 173-set (top1_var 0.666, corr_str 0.665, eigengap
0.659, eff_dim 0.650; all measure the same anisotropy). Correction to earlier notes: the raw-stat
corr_str family-AUC is **0.665**, not 0.73 (the 0.73 was a probe/routing signal on a different subset).
The figure: (left) normal-data eigenvalue spectra of illustrative tabular+TS exemplars, steep
(anisotropic->local) vs flat (isotropic->global); (right) all 173 datasets in top1_var x eff_dim,
colored by winning family, exemplars starred, AUC annotated with an honest "families overlap" caveat.
The anomaly-TYPE consequence remains a downstream leaderboard result (SPARC routing), not a scatter.

Retired: `eda_mechanism.py`/`FIG_mechanism.png` (refuted claim), `eda_spectrum.py`/`FIG_spectrum_eda.png`
(flagged PCA anomaly scatter; spectra now in FIG_type_separation). Diagnostics kept: `probe_offline.py`,
`probe_margvsjoint.py`, `STREAM_MARGJOINT.csv`, `eda_embed_explore.py`.
