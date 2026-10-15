# ResearchPivot state brief: ADReal project (cycle 0, 2026-10-15)

## 1. Problem and current claim

Current paper, "No Easy Wins: A Contamination-Controlled Benchmark for Evaluating Anomaly Detection and
Model Selection". It contributes (i) ADReal, 151 tabular + multivariate-time-series anomaly-detection
tasks built by a detector-free pipeline that removes trivially separable anomalies, anomalies identical
to normals, train/test duplicate leakage, degenerate base rates; (ii) an evaluation of 32 detectors
(25 classical PyOD + 7 deep); (iii) an evaluation of unsupervised (label-free) model-selection methods
against a fixed "always deploy the single best-on-average detector" reference. Headline: no popular
label-free selector beats the fixed detector. It is correct but reads as "a benchmark plus negative
results", which risks "incremental over ADBench / just negative" at D&B venues. A proposed method
(SPARC) was dropped because it was not significantly better than the fixed detector.

## 2. Evidence (151 tasks; regret = oracle ap_norm minus picked detector ap_norm; ap_norm = (AP-r)/(1-r))

- Construction: 1,192 source datasets -> 174 candidates -> 173 constructed -> 151 kept (floors: oracle
  ap_norm > 0.05, random-pick regret > 0.02, >= 100 test anomalies). Sources: OvrBench 66, OddBench 39,
  TSB-AD time series 38, ADBench 5, DAMI 3.
- Contamination prevalence in source data: 49% of datasets have duplicate normal rows, 35% duplicate
  anomalies, 24% anomalies that are exact copies of a normal (4,869 removed in 32 tasks); a naive random
  split leaks normals train->test in 62 of 173 tasks (8,285 points); 31% of pre-cap tasks have anomaly
  base rate > 0.5 (metric degenerate).
- Detectors: classical detectors are the per-task oracle on 118/151 (78%); adding all 7 deep detectors
  raises the mean oracle by only 0.021; best deep trails best classical by 0.071 per task. Top 6 by mean
  ap_norm are LOF/KNN variants (LOF k=10 best). Family oracle shares: neighbor 45%, distribution 24%,
  deep 22%, projection 9%, isolation 0%.
- Selection, classical pool (25 detectors): fixed detector LOF k=10 0.130; random pick 0.219; consensus
  0.218, HITS 0.219, model-centrality 0.223 (= random, p >= 0.56); excess-mass 0.163, mass-volume 0.165
  (significantly WORSE than fixed, p=0.019 / 0.007); IREOS 0.233; IFOREST-R 0.241; UDR 0.238; Goswami
  injection selector 0.122 (not significant vs fixed, p=0.88). Adding deep detectors to the candidate pool
  raises regret for 7 of 8 selectors.
- Regret is heavy-tailed: fixed detector median 0.076 but 90th percentile 0.300 -> selection matters on a
  minority of tasks. Fixed detector regret by source: OddBench 0.198, OvrBench 0.145, time series 0.038.
- Dropped method SPARC (grade detectors by ROC-AUC separating a dependency-breaking synthetic probe from
  validation normals): 0.114 over 10 seeds vs fixed 0.130, p=0.076 (not significant), not significant on
  any source; its edge comes almost entirely from choosing a smaller neighborhood size k for LOF/KNN per
  task (55 of 63 wins).

## 3. Key insights (non-obvious)

1. Hardening leaves anomalies that are marginally typical but sit in locally sparse joint regions, so
   neighbor/local-density detectors dominate and marginal/distribution criteria fail.
2. Deep detectors get INFLATED label-free scores: they separate synthetic/normal-data signals better than
   classical (probe separation ROC-AUC 0.72 vs 0.63) while being no better on real anomalies (ap_norm
   0.12 vs 0.13). Every label-free criterion that may pick a deep detector is drawn to them and pays regret.
3. Agreement selectors fail even transductively (criterion computed with the test anomalies visible):
   advantage over random +0.006, not significant. The failure is the hardness, not only the protocol.
4. A single good default (LOF k=10) is a very strong, rarely-beaten label-free choice; most remaining
   regret sits in a minority of tasks where the oracle is a different family.
5. The only per-task signal any selector exploited was the neighborhood-size choice.

## 4. Tried and rejected (do NOT propose again)

- SPARC dependency-breaking synthetic probe as a selector (not significant vs fixed; seed-sensitive).
- Probe variants: bin-interpolation vs observed-value draws (no difference at 10-seed mean); single-
  feature marginal probe (0.155 alone; combined with SPARC worse, p=0.017); kNN density-valley probe;
  hold-out-cluster "void" probe; fixed-radius void probe; null-probe subtraction / generalization-gap
  correction to de-bias deep detectors (real signal, but selection not improved).
- Combined routers by dataset-type index (anisotropy index AUC 0.66), marginal-tail regime router.
- Deep clustering (VaDE) as a preprocessing for selection: not better than PCA+k-means (p=0.23).
- Contamination-impact on SELECTORS: contaminated vs cleaned benchmark -> agreement selectors at random
  on both; transductive vs leak-free -> no significant difference.
- Agreement selectors, internal criteria (EM, MV, IREOS), IFOREST-R, UDR, injection selector (Goswami):
  all evaluated, none beats the fixed detector.

### Cycle 1 (2026-10-15): strategist proposed (1) few-label selection, (2) benchmark rank reversals,
(3) calibrated detector portfolios. Triage on all 150 tasks, 25 classical detectors, per-point scores:
- (3) PORTFOLIO, validated DEAD: LOF_k10+COPOD max-of-tail-p regret 0.148 vs LOF_k10 0.108 (worse);
  recall at 5% FPR -15%; not a tie artifact (only 4% of anomalies at the p floor; tie-break 0.146; mean of
  standardized scores 0.153). Every LOF+specialist combination tested is worse than LOF alone.
- (1) FEW-LABEL SELECTION (k anomalies used only to choose among the 25 detectors): gap closed vs fixed
  G1 0.13, G2 0.20, G3 0.26, G5 0.40, G10 0.49 overall; draws beating fixed 87-100%. TABULAR ONLY (112
  tasks) G5 = 0.55 (OddBench 0.69, OvrBench 0.47). TIME SERIES harmful (regret 0.073 vs fixed 0.019):
  all 38 TS tasks contain a SINGLE anomaly event (overlapping windows of one segment), and the default is
  already near-optimal there. Abstain-unless-margin variant: G5 0.42.
- (2) RANK REVERSALS: not run (needs raw reconstruction); prior art is crowded (Kluttermann et al. 2026
  ranking instability on 690 OddBench datasets; Deng 2026 overlap).
- Decisive incumbent check DONE (112 tabular tasks, 10 draws): TRAINING on the same k labels beats
  SELECTING with them from k=2. Gap closed vs fixed (G; >1 = beats the per-task oracle of the 25
  unsupervised detectors): k=1 SEL 0.28 / LR 0.23; k=2 0.34 / 0.66; k=3 0.44 / 0.86; k=5 0.55 / 1.07;
  k=10 SEL 0.63, LR 1.34, RF 1.44. Per source at k=5: OddBench LR 1.38 (beats oracle on 23/38),
  OvrBench 0.86 (33/66), ADBench 0.20 (n=5), DAMI 1.74 (n=3). Nearest-labeled-anomaly distance only
  0.07 (so not a near-duplicate effect). NEGATIVE CONTROL (5 normals mislabeled as anomalies): regret
  0.34 vs fixed 0.14 -> pipeline sound.
- Stress tests (k=5): UNSEEN TYPE (labels from one k-means anomaly cluster, test on the others): LR -0.25
  (worse than default, loses on 66/111), SEL 0.05, HYBRID rank-average(LR, LOF_k10) +0.22. Same draws on
  all types: LR 0.61, HYB 0.88. REALISTIC 2% PREVALENCE: LR 1.30, HYB 1.44. HYB was pre-specified.
- Current lead: "label-free selection cannot beat a strong default, but 2-5 labeled anomalies with a
  linear model hedged by the default beat every unsupervised choice; pure few-label supervision fails on
  unseen anomaly types, the hedge does not." Main risk: novelty vs ADBench semi-supervised results,
  OneShot (arXiv 2610.16219, TS label-efficient detector weighting), open-set supervised AD (DRA etc.).

### Cycle 2 (2026-10-16): strategist (gpt-6.1-sol-pro) + second opinion (claude-sonnet-5.5) via
OpenRouter both converge: thesis "spend a few labels on TRAINING, not selection; hedge with a strong
unsupervised default"; target TMLR (45-65% if confirmed), DMKD fallback. Prior art: WSADBench (KDD 2026
D&B, 1/5/10 labels, known/unknown types) is the main overlap; it has no per-dataset oracle and no hedge.
Smoke of the most likely sinking objection (label-acquisition realism), fixed evaluation set, 112 tabular
tasks x 10 reps, INDUCTIVE hedge (empirical CDFs on validation normals):
  random 5 labels: SEL 0.50, LR 1.04, HYBi 1.09; incumbent-CAUGHT (LOF top-20, ~11 labels): SEL 0.05,
  LR 0.70, HYBi 0.85; CAUGHT top-50 (~27): SEL 0.04, LR 1.17, HYBi 1.12; incumbent-MISSED 5: SEL 0.38,
  LR 0.95, HYBi 1.02. HYBi worse than the default on 10-17 of 112 tasks.
  -> selection collapses under realistic acquisition; training + hedge does not. VALIDATED LEAD; loop
  stopped after cycle 2. Semantic anomaly types recoverable for ADBench CV/NLP families (CIFAR10,
  FashionMNIST 100%, 20news 99%, agnews 30%) by exact embedding match; OvrBench/OddBench raw data has no
  per-anomaly class.

## 5. Assets available today

- 173 task bundles (train normals / validation normals / test normals / hard test anomalies),
  standardized-feature basis, local (168 MB).
- Per-task test ap_norm for all 32 detectors; per-task validation-normal scores for all 32 detectors.
- Raw source datasets on disk (ADBench, DAMI, OddBench, OvrBench, TSB-AD, UCR, CV and NLP embedding sets),
  so contaminated "before" versions of tasks can be rebuilt; recovered construction code.
- Compute: local 6-core CPU workers (classical detectors refit in minutes), local RTX 2060 6 GB,
  RunPod GPU account (budget fine for a few-hour job). No new data collection.

## 6. Target and impact bar

Venue: TMLR, NeurIPS Datasets & Benchmarks, KDD / ICLR / AAAI / Data Mining and Knowledge Discovery. The
paper should be novel AND clear a decisive bar, e.g. any of:
- a selection or deployment method that cuts regret vs the fixed detector by >= 30% (0.130 -> <= 0.09)
  and significantly, or closes >= 50% of the fixed-to-oracle gap;
- a label-efficiency result of the form "k labeled anomalies (k <= 5) recover >= 50% of the oracle gap",
  if not already known;
- a contamination finding that flips a large fraction (>= 25%) of published pairwise detector-ranking
  conclusions or a headline claim of a widely cited benchmark;
- a genuinely new capability or protocol others would adopt.
A marginal (few-point) improvement is not worth a cycle. It is OK to change almost everything,
including the title, framing, and which contribution is the headline.
