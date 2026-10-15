````markdown
CONFIRMED: The three attached files are identical copies of the ADReal state brief and current paper source; their topic matches the requested research pivot.

# ADReal: Three High-Impact Research Pivots
## Ranked research strategy — Cycle 1

### Executive assessment

The most important finding in the existing project is not that
unsupervised model selection fails. It is that detector choice has
substantial value on a minority of difficult tasks, but normal-only
validation provides insufficient information to recover that value.

This suggests changing the research question.

Instead of asking:

"Can we select the best anomaly detector using only normal data?"

ask one of three substantially different questions:

1. How little anomaly supervision is needed to recover most of the
   value of selecting the right detector?
2. How much of the established detector leaderboard is an artifact
   of benchmark construction, rather than genuine detector quality?
3. Can we avoid selecting one detector entirely, and deploy a small
   calibrated portfolio that captures complementary anomalies?

These are ranked by estimated combination of quantitative ceiling,
scientific novelty, practical feasibility, and publication potential.

| Rank | Pivot | Primary outcome | Proposed high-impact target |
|---|---|---|---|
| 1 | Five Anomalies Are Enough? | Few-shot selection regret | 0.130 -> <=0.065 |
| 2 | Benchmark-Induced Rank Reversals | Pairwise ranking inversions | >=25–35% |
| 3 | Replace Selection with Portfolios | Deployment regret | 0.130 -> <=0.080 |

All numbers above are research targets, not observed results.

My preliminary ranking is:

- Highest probability of a decisive positive result: #1
- Highest potential Datasets & Benchmarks contribution: #2
- Highest algorithmic upside, but less predictable: #3

A central qualification: the supplied files contain aggregate
experimental results, not the underlying score matrices or task
bundles. I have not run experiments. The ceilings below are reasoned
hypotheses; the proposed tests are designed to reject them quickly.

All three directions require substantial checks against prior work.
The closest papers named below come from the attached manuscript
and established knowledge, not a new literature search.


============================================================
PIVOT 1 — FIVE ANOMALIES ARE ENOUGH?
============================================================

Proposed title:

"Five Anomalies Are Enough? The Value of Minimal Supervision
for Anomaly Detector Selection"

Alternative, if results are particularly strong:

"Breaking the Unsupervised Selection Barrier with Five
Anomaly Examples"


(a) NEW HYPOTHESIS / REFRAMING

Normal-only detector selection is fundamentally underinformed
because normal behavior does not identify which departures from
normality matter in the application. However, a tiny number of
representative real anomalies may identify the relevant detector
family and recover much of the regret suffered by a fixed default.

The contribution changes from finding a better unsupervised
criterion to measuring and exploiting the value of extremely
limited anomaly supervision.


(b) NOVELTY AND WHO WOULD CITE IT

The key research question is:

How many labeled anomalies are required before per-task detector
selection becomes economically worthwhile?

This is different from conventional few-shot anomaly detection.

In ordinary few-shot detection, labeled anomalies are used to
train or adapt a detector.

Here:

- All 32 detectors are already trained without anomalies.
- No detector parameters are changed.
- The only learned decision is which detector to deploy.
- Supervision budgets are extremely small: 1, 2, 3, or 5 anomalies.
- Performance is assessed against a strong fixed detector and
  a per-task oracle.

This isolates the informational value of anomaly labels
independently of detector training.

Closest work:

1. MetaOD:
   Zhao, Rossi and Akoglu (2021),
   "Automatic Unsupervised Outlier Model Selection."

   MetaOD uses experience across datasets to predict detector
   performance.

   The proposed work studies the complementary question:
   how much direct, task-specific anomaly information is worth,
   especially with only a handful of labels?

2. ELECT:
   Zhao, Zhang and Akoglu (2022),
   "Toward Unsupervised Outlier Model Selection."

   ELECT studies information transfer and budget-aware selection.

   Our distinction must be explicit: the budget is for real
   labeled anomalies, with controlled acquisition and a
   quantitative label-value curve.

3. ADBench:
   Han et al. (2022).

   ADBench includes supervised and unsupervised detection
   settings, but does not establish the proposed marginal
   value-of-label result for selecting among a fixed detector
   portfolio on a hardened benchmark.

Potential audiences:

- Anomaly detection researchers.
- Active learning and experimental design researchers.
- AutoML and model selection researchers.
- Industrial practitioners deciding whether to invest in
  labeling a handful of incidents.

The strongest novel contribution would not merely be
"five examples improve model selection."

It would be a reproducible label-efficiency frontier showing:

- when zero-label selection fails;
- how quickly real labels close the oracle gap;
- whether anomaly diversity matters more than label count;
- whether detector selection requires fewer labels than
  training or adapting a detector;
- when further labeling produces diminishing returns.

IMPORTANT NOVELTY RISK:

Few-shot detector selection, weakly supervised anomaly detection,
and active anomaly labeling are established research areas.

A large numerical improvement alone may not be sufficiently novel.

The potential contribution is a sharply defined decision problem,
a principled low-label selection method, and reproducible evidence
of an unexpectedly small supervision requirement.

This is a medium-novelty, potentially high-impact direction,
not an assured first-of-its-kind method.


(c) PLAUSIBLE QUANTITATIVE CEILING

Current reference regret:

R_fixed = 0.130

The strongest result worth pursuing:

R_5 <= 0.065

That would recover at least 50% of the fixed-to-oracle gap.

It is a 50% relative reduction in regret.

A more ambitious outcome would be:

R_5 <= 0.050

This corresponds to approximately 62% gap recovery.

Why this might be reachable:

1. The oracle is highly heterogeneous.

   Its winning families are distributed approximately as:

   - Neighbor: 45%
   - Distribution: 24%
   - Deep: 22%
   - Projection: 9%

   Thus, different anomaly mechanisms reward substantially
   different scoring functions.

2. The benchmark's strong default fails disproportionately
   on a minority of tasks.

   Median regret: 0.076
   90th percentile: 0.300

   There is a large pool of potentially recoverable regret.

3. A single real anomaly carries information that normal-only
   validation cannot supply.

   For example, if LOF ranks the observed anomaly near the
   bottom of its anomaly distribution while another detector
   ranks it near the top, this directly discriminates
   between detector hypotheses.

4. Five positive examples combined with thousands of
   unlabeled validation normals can provide useful estimates
   of anomaly-vs-normal ranking.

Nevertheless, five samples can be catastrophically misleading
when anomalies are heterogeneous.

This may be the main obstacle to reaching 0.065.

The ceiling is therefore plausible, not established.

I would not claim 0.050 is likely without first measuring the
variance across sampled anomaly examples.


(d) SINGLE CHEAPEST CONFIRM-OR-KILL EXPERIMENT

Experiment:

"Five positive examples as a detector-selection signal."

Use a stratified sample of 30 ADReal tasks:

- 10 OddBench
- 10 OvrBench
- 10 TSB-AD

Choose tasks by a fixed hash of task identifier, not by
their detector regrets or expected method performance.

Candidate pool:

Use all 25 classical detectors initially, conditional on
recovering their test-point scores. Do not train deep models.

For each task:

1. Fit or load all 25 detectors.

2. Retrieve detector scores for validation normals.

3. Retrieve detector scores for every test point.

4. Randomly choose five distinct positive examples from
   the existing test anomalies.

5. Remove those five from the evaluation set.

6. Score each candidate detector using the pairwise
   probability that the selected anomalies outrank
   validation normals.

For detector j:

Q_j = (1/5) sum over positive examples a [
          (1/|V|) sum over validation normals n
          I(score_j(a) > score_j(n))
      ]

Use 0.5 credit for tied scores.

This is a positive-versus-normal ranking statistic.

It does not require choosing a decision threshold.

7. Select argmax_j Q_j.

8. Compute ap_norm on the remaining test examples only.

9. Repeat with 100 deterministic random draws of five
   positive examples per task.

CRITICAL:

The detector oracle and fixed-detector performance must
also be recomputed after removing the sampled positives.

Otherwise, selection regret and baseline regret are evaluated
on different populations.

Primary metric:

Gap recovery:

G_5 = 1 - mean(R_5) / mean(R_fixed)

where both regrets use the same reduced evaluation sets.

Also measure:

- Median gap recovery across repetitions.
- Fraction of repetitions improving over the fixed detector.
- Paired per-task gains.
- Regret on the highest-regret quartile.
- Oracle-family identification accuracy.

Pre-stated positive lead:

G_5 >= 0.40 on the smoke sample,

and at least 70% of sampling repetitions improve on the
fixed detector.

This would justify a full 151-task study targeting G_5 >= 0.50.

Pre-stated kill:

G_5 < 0.20,

or fewer than 55% of repetitions improve on the fixed detector.

The latter indicates that even five examples produce unstable,
unreliable decisions.

A result between these thresholds is inconclusive.

Why choose a 40% smoke threshold rather than demand 50%
immediately?

Because 30 tasks produce noisy estimates, particularly when
the source composition differs from the full benchmark.

A 40% signal is strong enough to justify expansion but does
not count as publication-level success.

No parameter tuning on these 30 tasks is allowed.


(e) MAIN RISK

The greatest risk is representativeness.

Five observed anomalies may belong to one narrow subtype.

The detector that ranks those five well may be poor on other
anomalies within the same task.

Other risks:

- This is a known form of empirical model validation.
- A simple ranking-AUC selector may be too obvious.
- The data may contain near-duplicate anomaly examples.
- Sampling examples from the same test population can
  produce optimistic results relative to temporal deployment.
- Strong results may reflect favorable anomaly sampling rather
  than genuine label efficiency.

Scientific safeguards:

Sample anomaly exemplars by distinct feature vectors.

Retain the full distribution of results over draws.

Ultimately introduce a separate labeled-calibration-anomaly
split and an untouched future test split.

The final paper must distinguish randomly supplied positive
examples from actively queried examples.

An active-query protocol would require a separate labeling
budget and a carefully defined unlabeled candidate pool.

Do not treat anomaly labels drawn freely from ground truth
as if an operational active learner could obtain them without
search or screening costs.

Publication potential:

TMLR: HIGH if the result is robust.
KDD: MODERATE-HIGH with a genuine active-selection method.
ICLR: MODERATE, requiring stronger methodological innovation.
NeurIPS D&B: HIGH as a reusable label-efficiency protocol.

Overall recommendation:

BEST FIRST PIVOT TO TEST.


============================================================
PIVOT 2 — BENCHMARK-INDUCED RANK REVERSALS
============================================================

Proposed title:

"Are Anomaly Detection Leaderboards Stable?
A Controlled Study of Benchmark-Induced Rank Reversals"

Stronger result-dependent title:

"Benchmark Construction Reverses Anomaly Detector Rankings"


(a) NEW HYPOTHESIS / REFRAMING

Anomaly detector rankings may depend more strongly on how a
benchmark constructs its evaluation population than on the
underlying detector algorithms.

By treating duplicate handling, leakage prevention, anomaly
hardening, and prevalence as controlled interventions, we can
measure how frequently the same methods exchange statistical
rank ordering under defensible evaluation protocols.

This pivots from introducing another cleaned benchmark to
studying benchmark validity and ranking stability itself.


(b) NOVELTY AND WHO WOULD CITE IT

The new object of study is not the detector.

It is the benchmark intervention.

For each source dataset, define several versions representing
different evaluation decisions.

Examples:

B0: Source-style benchmark construction.

B1: Duplicate- and leakage-controlled construction.

B2: B1 with ambiguous anomaly labels removed.

B3: B2 with trivial anomaly cases removed.

B4: B3 with controlled prevalence.

These should be constructed as reproducible, matched
experimental conditions wherever possible.

The scientific question becomes:

What fraction of conclusions about detector superiority is
stable under reasonable benchmark construction choices?

Closest prior work:

1. Campos et al. (2016),
   "On the Evaluation of Unsupervised Outlier Detection:
   Measures, Datasets, and an Empirical Study."

   Established that evaluation choices, datasets and metrics
   substantially influence detector comparisons.

2. Steinbuss and Böhm (2021),
   "Benchmarking Unsupervised Outlier Detection with
   Realistic Synthetic Data."

   Studied realism and anomaly generation.

3. Wu and Keogh (2023),
   "Current Time Series Anomaly Detection Benchmarks
   are Flawed and are Creating the Illusion of Progress."

   Demonstrated severe time-series benchmarking problems.

4. Röchner et al. (2025),
   "We Need to Rethink Benchmarking in Anomaly Detection."

   Challenges existing tabular benchmarking practices.

5. MacrOData (2026).

   Provides a much larger tabular outlier detection benchmark
   and an important competing reference for broad claims.

The differentiation cannot simply be:

"Contamination affects benchmark results."

That is already established.

The potential new claim is:

"Standard benchmark construction decisions systematically
reverse a substantial fraction of statistically supported
detector comparisons, even when the detector implementation,
data source, and evaluation metric are held fixed."

This is closer to an empirical measurement science paper.

Possible contributions:

- Paired counterfactual benchmark construction.
- Statistical ranking stability measures.
- Quantification of intervention-specific ranking reversals.
- Identification of which interventions cause the reversals.
- A reusable detector-ranking robustness audit.
- A criterion for when benchmark leaderboards should be
  considered unstable.

Who would cite it:

- Authors of anomaly-detection benchmarks.
- Authors comparing new detectors against baselines.
- Evaluation-methodology researchers.
- NeurIPS D&B reviewers and benchmark designers.
- Researchers studying reproducibility and dataset artifacts.

This direction potentially changes what future authors need
to report, not merely which detector they select.


(c) PLAUSIBLE QUANTITATIVE CEILING

Primary target:

At least 25% of statistically supported pairwise detector
ranking conclusions reverse between source-style and
controlled benchmark construction.

Ambitious ceiling:

35–40% of supported pairwise conclusions reverse.

A second possible high-impact outcome:

The published aggregate ranking of detector families reverses.

For example, neighbor-based detectors may dominate after
hardening while another family dominates before hardening.

That would be a much stronger narrative than a modest
change in mean AP.

Why this might be reachable:

The brief documents substantial artifacts:

- 49% of source datasets contain duplicate normal rows.
- 35% contain duplicate anomalies.
- 24% contain anomalies identical to normals.
- 62 of 173 tasks leak normals across naive splits.
- 31% of pre-cap tasks exceed 50% anomaly prevalence.

Moreover, the hardening procedure deliberately removes
marginally extreme anomalies.

That changes the kinds of anomalies a detector encounters.

Marginal detectors may look excellent before hardening but
much less competitive afterward.

Local-density detectors may show the opposite pattern.

This is a mechanism capable of producing large rank changes.

However, there is an important distinction:

The prevalence of contamination does not imply an equally
large prevalence of statistically significant ranking reversals.

Many contaminated tasks may preserve detector ordering.

Therefore 35–40% is an ambitious ceiling, not an inference
directly established by the existing contamination statistics.


(d) SINGLE CHEAPEST CONFIRM-OR-KILL EXPERIMENT

Experiment:

"Matched before/after leaderboard inversion."

Select 30 datasets using a pre-registered rule:

- 10 with substantial duplicate/leakage artifacts.
- 10 with substantial hardening removals.
- 10 comparison datasets with relatively small changes.

Use only five representative classical detectors:

1. LOF k=10.
2. KNN, using a fixed configuration from the paper.
3. HBOS.
4. COPOD.
5. Isolation Forest.

These span neighborhood, distribution, and isolation behavior.

Choose detector implementations and hyperparameters exactly
as in the existing benchmark.

Reconstruct two versions of each task:

A: Source-style construction.
B: Controlled ADReal construction.

Use the same detector hyperparameters and preprocessing
wherever the compared protocols permit.

Important design detail:

Changing the test anomaly population is not equivalent to
correcting data leakage.

Therefore record both:

- The total A -> B change.
- Which test cases have been added, removed, or reweighted.

A full study must separate these effects.

For the smoke test, estimate the total treatment effect.

Fit the five detectors under both protocols.

Compute per-task ap_norm.

For each of the 10 detector pairs:

D_AB = mean_task(ap_norm_A - ap_norm_B)

Compute the paired difference under each protocol.

The principal outcome is a reversal when:

sign(D_AB_source) != sign(D_AB_clean)

But mere numerical sign changes are insufficient for the
paper's headline.

Also compute a paired bootstrap confidence interval for
each difference, resampling original source datasets or
independent task groups.

Count strong inversions only when the pairwise difference
is supported on both sides and its sign reverses.

Primary metric:

Strong pairwise reversal fraction:

F = number of strongly reversed detector pairs
    / number of pairs with statistically supported
      conclusions in at least one protocol

Report the denominator and also the percentage among all
10 comparisons.

For inferential testing, use paired task-level differences,
not individual anomaly points as independent replicates.

Positive lead:

At least 3 of 10 detector pairs reverse numerical ranking,

AND at least 2 have bootstrap-supported reversals.

Additionally, the reversal must occur in at least two
detector-family comparisons.

This would justify expanding to all 32 detectors and the
full set of source datasets.

Kill:

At most 1 of 10 pairs reverses,

OR no reversal is statistically supported,

OR apparent reversals disappear when protocol differences
are decomposed into their constituent interventions.

Intermediate results are inconclusive.

Important feasibility qualification:

The state brief says raw data and recovered construction
code are available; it does not establish that five-detector
before/after scores already exist.

Unlike pivots based solely on the score matrix, this test
requires reconstruction and refitting.

It is cheap relative to a full benchmark rerun, but the
under-one-hour limit depends on selecting small tasks.


(e) MAIN RISK

This pivot has two substantial risks.

RISK 1: Known result.

Benchmark sensitivity and rank instability are not new.

Steinbuss and Böhm, Campos et al., Wu and Keogh, and
Röchner et al. already occupy adjacent scientific territory.

A paper that simply demonstrates another ranking change
will appear incremental.

The defensible contribution must be a new measurement
framework with sufficiently decisive counterfactual evidence.

RISK 2: Confounding by changed research objective.

Removing easy anomalies does not merely correct bad data.

It changes the detection problem.

A detector that excels at easy marginal anomalies is not
necessarily a bad detector.

It might be the correct detector for the original task.

Thus, the paper cannot automatically declare the cleaned
ranking "the true ranking."

It must distinguish:

- Objective-preserving corrections:
  leakage, duplicate evaluation weighting, ambiguous labels.

- Objective-changing interventions:
  hardening, anomaly prevalence changes.

The resulting narrative should be about ranking dependence
on benchmark construction, not about one universally
correct leaderboard.

A further methodological concern:

The current benchmark applies post-scoring inclusion floors.

Those are detector-pool-dependent.

For counterfactual ranking studies, dataset inclusion must
be fixed before comparing detector rankings, or the selection
mechanism itself can bias conclusions.

My recommendation is to use the entire 173-task constructed
pool where feasible, with a pre-specified analyzable subset.

Do not select tasks conditional on the final detector oracle
when estimating population-wide ranking reversals.

Publication potential:

NeurIPS D&B: VERY HIGH if results are decisive.
DMKD: HIGH.
TMLR: HIGH.
KDD: MODERATE-HIGH.
ICLR: LOWER unless accompanied by a general theory
      of benchmark-induced evaluation bias.

Overall recommendation:

BEST BENCHMARK-CENTERED PIVOT.

It is also the clearest route to reusing the original
hardening pipeline as the main scientific asset.


============================================================
PIVOT 3 — REPLACE SELECTION WITH PORTFOLIOS
============================================================

Proposed title:

"Beyond Selecting One: Budgeted Detector Portfolios
for Reliable Anomaly Detection"

Alternative:

"Selection Is the Wrong Objective:
Complementary Detector Portfolios for Anomaly Detection"


(a) NEW HYPOTHESIS / REFRAMING

Selecting a single anomaly detector is unnecessarily brittle
when different detectors identify different anomaly mechanisms.

A small portfolio of two or three complementary detectors,
calibrated using validation normals and combined under a shared
false-alarm budget, may recover much of the oracle's advantage
without predicting which individual detector will win.

The objective changes from model selection to deployment
under a constrained alerting budget.


(b) NOVELTY AND WHO WOULD CITE IT

This is an important reframing:

The best single detector need not be the best deployable system.

Instead of:

dataset -> select detector -> score anomalies

we use:

dataset -> choose small portfolio ->
normal-calibrate scores -> combine alerts

The crucial challenge is to combine detectors without
sacrificing false-positive control.

Candidate construction:

For detector j, estimate its upper-tail probability from
validation-normal scores:

p_j(x) =
  (1 + #{v in V: score_j(v) >= score_j(x)})
  / (|V| + 1)

This gives a common empirical scale.

For a portfolio S, define:

T_S(x) = max_{j in S}[-log(p_j(x))]

Equivalent interpretation:

An observation is suspicious when at least one detector
assigns it an unusually extreme score.

This can capture complementary anomaly types.

However, max-combination increases false alarms.

To address that:

Calibrate the final combined score against held-out
normal observations.

Do not use the same validation normals both to choose a
portfolio and to claim distribution-free false-alarm control.

A defensible protocol requires separate calibration data,
or cross-fitting.

A practical initial portfolio could be:

LOF k=10 + COPOD + PCA

A two-detector version could be:

LOF k=10 + COPOD

The portfolio is fixed globally in the first experiment.

Do not initially use a learned per-task router.

The research hypothesis is that deployment robustness can
come from diversity without accurate per-task selection.

Closest prior work:

1. Rayana and Akoglu (2016),
   "Less is More: Building Selective Anomaly Ensembles."

2. Aggarwal and Sathe (2015),
   "Theoretical Foundations and Algorithms for
   Outlier Ensembles."

3. Consensus and model-centrality approaches already
   evaluated in the submitted paper.

These create a serious novelty obstacle.

The distinction from those approaches must be substantial.

The rejected approaches seek a consensus ranking or select
detectors by agreement.

The proposed portfolio instead preserves minority signals.

In fact, agreement is not the objective.

Disagreement can be useful if it represents complementary
anomaly mechanisms.

The new research object is:

"How much complementary detection value can a small set
of models capture under a fixed false-alarm budget?"

The publication-worthy contribution would combine:

1. A budget-constrained portfolio objective.
2. Calibration using normals alone.
3. A novel analysis of coverage complementarity.
4. A statistically strong gain over the fixed detector.
5. Evaluation of portfolio size versus operational cost.

Potential audiences:

- Outlier ensembles.
- Monitoring and alerting systems.
- Industrial anomaly detection.
- Multi-model deployment.
- Statistical calibration and false-alarm control.


(c) PLAUSIBLE QUANTITATIVE CEILING

Headline target:

R_portfolio <= 0.080

Compared with:

R_fixed = 0.130

This represents approximately 38% relative regret reduction.

Ambitious ceiling:

R_portfolio <= 0.060

This would recover approximately 54% of the fixed-to-oracle
gap when both are scored against the same oracle.

Why this might be reachable:

The oracle family is heterogeneous.

The fixed LOF detector cannot simultaneously act as:

- a local-density detector;
- a marginal-tail detector;
- a global projection detector.

A carefully constructed portfolio could detect multiple
failure modes.

The existing selection results provide suggestive evidence:

The distribution family produces the oracle on approximately
24% of tasks, despite weak average performance.

That suggests specialists can add value outside their
usual operating regime.

However, there is a decisive missing measurement.

The available summary reports per-detector AP values,
not whether detectors find different individual anomalies.

Complementary task-level oracle wins do not guarantee
point-level complementarity.

Two detectors can have different AP despite identifying
largely the same anomalies.

The large-ceiling hypothesis therefore depends on
individual test-score overlap, not only the existing
oracle-family histogram.

This is precisely what the proposed test must measure.


(d) SINGLE CHEAPEST CONFIRM-OR-KILL EXPERIMENT

Experiment:

"Does a fixed, calibrated two-detector portfolio beat LOF?"

Use the same 30 stratified tasks from Pivot 1.

Detectors:

- LOF k=10.
- COPOD.

Use one deliberately simple ensemble.

No learned weights.
No per-task model selection.
No synthetic anomalies.
No clustering.
No routing features.

Split validation normals deterministically into:

V1: score calibration.
V2: portfolio calibration and normal-score diagnostics.

For each detector:

1. Fit on train normals.

2. Convert raw scores to empirical normal-tail probabilities
   using V1.

3. Define the portfolio score:

   T(x) = max(
       -log(p_LOF(x)),
       -log(p_COPOD(x))
   )

4. Use V2 for combined-score calibration.

5. Evaluate on the untouched test set.

Compute:

- ap_norm for LOF.
- ap_norm for COPOD.
- ap_norm for the combined portfolio.
- Test alert performance at fixed validation-calibrated
  false-positive budgets.

Use the all-32-detector oracle from the current protocol
where available, preserving the same regret denominator.

Primary metric:

R_portfolio =
mean_task(APnorm_oracle - APnorm_portfolio)

Secondary, more operational metric:

Detection recall at a fixed 5% normal false-alarm budget.

The combined portfolio is not restricted to the original
32 candidate detector rankings.

It can outperform their oracle on a task.

Therefore portfolio "regret" can be negative.

This must be reported transparently rather than clipped
to zero.

Positive lead:

R_portfolio <= 0.085 on the smoke subset,

AND the combined detector improves recall at 5% FPR
by at least 20% relative to LOF,

AND it does not increase the empirically measured
false-positive rate above the pre-specified tolerance
under the calibration protocol.

The two metrics serve different purposes:

- Regret measures aggregate detection quality.
- Recall at fixed FPR measures operational value.

Kill:

R_portfolio >= 0.115,

OR relative recall improvement at 5% FPR is below 5%,

OR the apparent AP gain disappears once the combined
score is correctly calibrated and evaluated.

Intermediate results are inconclusive.

If the simple two-detector portfolio passes, the next
experiment should test three-detector portfolios and
portfolio-selection objectives.

Do not introduce additional ensemble complexity before
the fixed two-detector version demonstrates real value.


(e) MAIN RISK

The biggest risk is that this is merely another ensemble.

There is extensive prior work on:

- Combining anomaly scores.
- Rank aggregation.
- Selective ensembles.
- Detector diversity.
- Score normalization.
- False-alarm calibration.

A naive max-score ensemble is not itself a novel algorithm.

The current paper has already shown that consensus-based
methods fail.

Although a minority-preserving portfolio differs from
consensus, that distinction must create measurable value.

Additional risks:

1. Deep and classical detectors may react to the same
   underlying anomaly characteristics.

2. Complementarity may be apparent only at the task level,
   not the observation level.

3. Normal-tail calibration can be unstable in extreme tails.

4. Combining detectors may magnify harmless deviations.

5. Validation-normal calibration assumes future normal data
   are reasonably exchangeable with calibration normals.

6. A portfolio may outperform the baseline on AP while
   worsening alert quality at realistic low false-positive
   rates.

That last possibility is particularly dangerous.

A deployment paper should report precision, recall and
alert burden at realistic operating points, not just AP.

Publication potential:

KDD: HIGH if a new, effective portfolio algorithm emerges.
TMLR: HIGH if improvements are broad and statistically strong.
ICLR: MODERATE, contingent on substantive methodological novelty.
DMKD: HIGH.
NeurIPS D&B: MODERATE unless the emphasis is a new portfolio
             benchmarking protocol.

Overall recommendation:

THIRD PRIORITY.

The possible improvement is large, but novelty is more
difficult to establish than for Pivot 2.


============================================================
ONE UNIFIED TRIAGE EXPERIMENT
============================================================

Name:

"ADReal Pivot Triage: Supervision, Benchmark Stability,
and Portfolio Complementarity"

Objective:

Determine which research direction has enough quantitative
signal to justify substantial investment.

The experiment should not attempt to produce a paper result.

It should determine whether each hypothesis has a plausible
path to meeting the impact bar.

Use only local CPU.

No GPU training.
No hyperparameter search.
No additional data collection.


1. INPUTS

Use 12 representative ADReal tasks.

Four each from:

- OddBench.
- OvrBench.
- TSB-AD.

Within each source, take two tasks from the upper half and
two from the lower half of fixed-detector regret.

This intentionally balances easy and difficult selection
cases.

It is a diagnostic smoke sample, not an unbiased
population estimate.

Select task IDs using a fixed, reproducible seed.

For the rank-reversal arm, favor tasks whose raw
reconstruction can be run cheaply.

Record any deviation from the source-balanced sample.

Use the following five detectors:

D1: LOF k=10
D2: KNN, fixed configuration
D3: HBOS
D4: COPOD
D5: Isolation Forest

Why five?

They cover multiple detector families and support
fast classical inference.

The full paper may eventually use all 32 models.

For this experiment, speed and falsifiability matter more
than exhaustive coverage.

Assume existing fitted detector artifacts are not available
unless verified locally.

Fit the five detectors once per task and cache all scores.

For expensive tasks, cap the normal training sample using
a fixed, label-independent sampling procedure.

Use the same cap across relevant arms.

Do not silently change preprocessing or detector parameters.


2. SHARED SCORE MATRIX

Construct one common score matrix for each clean task.

Rows:

- Validation normals.
- Test normals.
- Test anomalies.

Columns:

- Five detector anomaly scores.

Also retrieve or compute:

- Test labels.
- Task source.
- Existing per-task AP values.
- Number of removed hardening cases.
- Duplicate/leakage indicators.
- Raw source identifier.

This one score matrix supports Pivots 1 and 3.

Pivot 2 additionally needs raw-data reconstruction.


3. THREE TRIAGE ARMS

------------------------------------------------------------
ARM A — FIVE POSITIVE EXAMPLES
------------------------------------------------------------

For each task:

- Sample five distinct test anomalies.
- Select a detector using positive-versus-validation-normal
  ranking.
- Remove the sampled anomalies from evaluation.
- Measure gap recovery relative to the fixed LOF reference.
- Repeat 30 times.

Do not fit a new detector during these repetitions.

Statistic:

G5 = 1 - R5 / Rfixed

Positive signal:

G5 >= 0.40

Weak signal:

0.20 <= G5 < 0.40

Kill signal:

G5 < 0.20

Also record sampling instability.

A large mean improvement accompanied by frequent
catastrophic losses is not a strong signal.


------------------------------------------------------------
ARM B — RANKING REVERSAL
------------------------------------------------------------

For each task:

- Reconstruct source-style and ADReal-clean versions.
- Fit the same five detectors under both protocols.
- Compute ap_norm under both protocols.
- Compare the 10 detector pairs.

Statistic:

F = fraction of detector pairs that reverse ordering.

Record both numerical and statistically supported reversals.

Positive signal:

F >= 0.30 numerically,

with at least two supported reversals in the pooled
pairwise analysis.

Weak signal:

0.10 < F < 0.30

Kill signal:

F <= 0.10 and no supported reversal.

The main analysis must avoid incorrectly treating
multiple derived tasks from the same raw dataset as
independent observations.

At smoke scale, wide confidence intervals are expected.


------------------------------------------------------------
ARM C — TWO-DETECTOR PORTFOLIO
------------------------------------------------------------

For each clean task:

- Calibrate LOF and COPOD scores from validation normals.
- Compute max negative-log tail-probability score.
- Evaluate APnorm and recall at 5% FPR.

Statistics:

Gportfolio = 1 - Rportfolio / Rfixed

and

RecallGain =
  Recall_portfolio / Recall_LOF - 1

Positive signal:

Gportfolio >= 0.35

AND

RecallGain >= 0.20

Weak signal:

0.10 <= Gportfolio < 0.35

Kill signal:

Gportfolio < 0.10

OR

RecallGain < 0.05

If the LOF recall is zero on a task, use an absolute
recall difference for that task rather than dividing
by zero.

Calibration validity is mandatory.

A portfolio that obtains higher recall by exceeding
its false-positive budget does not pass.


4. EXPECTED RESOURCE USE

A CPU budget under one hour is realistic only with
appropriate sample selection and cached results.

Suggested allocation:

- 5 minutes: task loading and verification.
- 15 minutes: five-detector clean score generation.
- 5 minutes: few-shot resampling and evaluation.
- 5 minutes: portfolio construction and evaluation.
- 20 minutes: selected raw benchmark reconstruction
              and contaminated-version detector scoring.
- 10 minutes: metrics, plots and result table.

Total target: 60 minutes.

This is an engineering budget, not a guaranteed runtime.

If raw reconstruction or a selected task exceeds its
time budget, reduce task count according to a fixed
pre-declared rule.

Do not omit slow tasks opportunistically after examining
their results.

If the benchmark reconstruction cannot finish, classify
Pivot 2 as "not evaluated," not "killed."


5. TRIAGE DECISION MATRIX

Let:

A = Pivot 1 passes its positive threshold.
B = Pivot 2 passes its positive threshold.
C = Pivot 3 passes its positive threshold.

Mapping:

| A | B | C | Decision |
|---|---|---|---|
| Yes | No | No | Pursue few-shot detector selection |
| No | Yes | No | Pursue ranking-stability benchmark |
| No | No | Yes | Pursue calibrated portfolios |
| Yes | Yes | No | Rank by 50% gap-closure viability versus 25% supported reversals |
| Yes | No | Yes | Prefer few-shot if G5 >= 0.50; otherwise compare operational costs |
| No | Yes | Yes | Prefer benchmark pivot unless portfolio has >=50% regret reduction |
| Yes | Yes | Yes | Select by robustness and novelty; do not combine papers prematurely |
| No | No | No | Kill all three at smoke scale and reopen the search |

The thresholds are gates, not directly comparable
measurements of scientific value.

Do not rank a 35% portfolio improvement against a
30% reversal rate as if the two percentages measured
the same quantity.

Among positive leads, assess:

1. Can the direction plausibly clear its full impact bar?
2. Is the result stable across task sources?
3. Does the result survive a simple baseline?
4. Is the contribution already substantially known?
5. Is the full experiment practical with existing assets?

A positive smoke result does not establish significance
on the 151-task benchmark.


6. PRE-REGISTERED OUTPUT

Produce one compact result table.

| Metric | Pivot 1 | Pivot 2 | Pivot 3 |
|---|---|---|---|
| Primary metric | G5 | Strong reversal fraction | Gportfolio |
| Smoke positive bar | >=0.40 | >=2 supported pair reversals | >=0.35 |
| Secondary safeguard | >=70% draws improve | >=2 family contrasts | >=20% recall gain |
| Smoke kill bar | <0.20 | <=1 numerical reversal | <0.10 |
| GPU needed | No | No | No |
| Additional raw reconstruction | No | Yes | No |
| Primary scientific concern | Few-shot novelty | Ranking confounding | Ensemble novelty |

Report alongside:

- Per-task metrics.
- Bootstrap confidence intervals.
- Results by source.
- Worst-task degradation.
- Runtime.
- Number of tasks excluded and why.
- Any implementation departures from the preregistration.

Do not tune the thresholds after seeing the data.


============================================================
FINAL STRATEGIC RANKING
============================================================

RANK 1: FEW-SHOT SELECTION

Best combination of:

- Large possible regret reduction.
- Minimal implementation effort.
- Strong decision relevance.
- Clear success/failure measurement.

The decisive question:

Can five real anomalies recover at least half the
fixed-to-oracle gap?

If yes, the project can become a study of the
information economics of anomaly supervision.

The primary research risk is not the size of the effect.
It is demonstrating something sufficiently novel beyond
existing few-shot and active-selection methods.


RANK 2: BENCHMARK RANKING STABILITY

Best reuse of the original contribution.

Potentially the strongest NeurIPS Datasets & Benchmarks paper
if a large number of supported ranking conclusions reverse.

The decisive question:

Does benchmark construction change not merely measured
performance, but which detectors scientific studies
would declare superior?

The primary risk is that the result is already known,
or that the reversal comes from intentionally changing
the anomaly population rather than removing artifacts.


RANK 3: CALIBRATED PORTFOLIOS

Potentially the strongest practical deployment algorithm.

The decisive question:

Can two complementary detectors substantially outperform
the strong LOF default without learning a task-specific
selector and without exceeding the false-alarm budget?

The primary risk is that standard score-ensemble methods
already achieve the same improvement.


============================================================
CROSS-CUTTING WARNINGS
============================================================

1. Avoid reusing the existing "negative result" framing
   as the headline.

   It can remain part of the motivation.

2. The 151-task benchmark was partly selected using
   detector-pool-dependent floors.

   For new claims involving meta-learning, detector
   comparisons, or population-level inference, analyze
   the consequences of this selection.

3. Do not use the oracle to choose which tasks to
   demonstrate in the final paper.

4. A large effect on 12 or 30 tasks does not establish
   a reproducible effect on 151 tasks.

5. Current regret is defined against the best detector
   among the existing candidate pool.

   For ensembles, this is a reference frontier rather
   than a theoretical performance upper bound.

6. Base-rate-normalized AP is not itself an operational
   alerting metric.

   Deployment claims should additionally use precision
   or recall under an explicit alert budget.

7. Strong claims about real-world deployment must account
   for the fact that the current test prevalence is often
   much higher than production anomaly prevalence.

   The manuscript reports median test prevalence of 0.35.

   This makes controlled false-positive evaluation
   particularly important.

8. Closest-prior-work comparisons here are provisional.
   No new literature search was performed, as requested.

9. None of the three hypotheses has yet demonstrated the
   quantitative effect required for publication.

   Their value is that each has a clear, cheap rejection
   experiment and a plausible route to a decisive result.


BOTTOM LINE

I would first test whether five genuine anomaly examples
can recover at least 50% of the detector-selection oracle
gap.

In parallel, I would use the existing raw-versus-hardened
datasets to test whether benchmark construction reverses
statistically supported detector rankings.

I would pursue the portfolio direction only if a fixed,
calibrated two-detector combination produces an immediately
large improvement.

These three experiments test fundamentally different
scientific explanations of the same existing evidence:

- Pivot 1: the missing information is anomaly supervision.
- Pivot 2: the apparent detector hierarchy is benchmark-dependent.
- Pivot 3: requiring a single detector is the wrong constraint.

The best eventual paper should be built around whichever
explanation survives falsification and crosses the impact bar,
not around preserving the current benchmark narrative.
````