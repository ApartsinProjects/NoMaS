# Cycle 2: Make this a label-budget and anomaly-type generalization paper

## Recommendation

**Stop pursuing a new selector or ensemble algorithm. Pursue a controlled empirical paper about what a handful of anomaly labels actually buy—and what they do not.**

The strongest candidate thesis is:

> **With the same tiny anomaly-label budget, fitting a simple classifier is more valuable than selecting an unsupervised detector, but that advantage depends on anomaly-type coverage; retaining an unsupervised default can recover some of the lost transfer to unseen types.**

This is narrower—and more defensible—than “five anomalies solve anomaly detection” or “a new hedge handles open-set anomalies.”

The current evidence is strong enough to justify this pivot. It is **not yet sufficient to establish novelty over open-set supervised AD or superiority over modern weakly supervised detectors**.

---

## (a) Novelty: what is genuinely available here?

### The sharpest defensible claim

The potentially novel contribution is the **joint comparison of label use and anomaly-type coverage**:

> **Under matched absolute budgets of 1–5 labeled anomalies, compare spending labels on detector selection, classifier fitting, and supervised–unsupervised combination. Measure how their ordering changes when the labeled anomalies do not cover the anomaly types encountered at deployment.**

Your distinctive result would be a reproducible decision rule:

- **Representative anomaly examples:** spend the labels on fitting, not merely selecting.
- **Incomplete anomaly-type coverage:** fitting alone can be harmful; retain the normality-based detector.
- **No anomaly examples:** a strong fixed default remains a demanding reference for the evaluated label-free selectors.

That is an empirical deployment result, not a novel learning primitive.

### What is already known?

| Finding | Novelty assessment |
|---|---|
| Labeled anomalies improve detection over unsupervised methods | **Known.** ADBench already establishes this broadly. |
| Very few labels can be useful | **Not a safe novelty claim.** Weakly supervised AD and the described OneShot work occupy this territory. |
| A classifier can outperform unsupervised detectors | **Known in principle.** The constant-budget magnitude on ADReal may be interesting, not conceptually new. |
| Supervision on known anomaly types can hurt detection of unknown types | **Known motivation of open-set supervised AD**, including DRA. |
| Combining supervised and unsupervised information can help | **Not a new methodological idea.** Rank averaging is particularly difficult to claim as an algorithmic contribution. |
| Five labels let LR exceed the oracle of 25 unsupervised detectors | **A useful quantitative result**, provided the comparison is leakage-free and statistically supported. It is not “beating the anomaly-detection oracle.” |
| Training beats selection with the identical examples from \(k=2\) | **Potentially distinctive as a systematic, budget-matched result.** This needs strong training and selection competitors. |
| The training-versus-selection ordering changes under anomaly-type exclusion | **The strongest candidate contribution**, especially if established using semantic types rather than only clusters. |
| A simple hedge remains useful across representative and missing-type conditions | **Potentially valuable baseline evidence**, not yet a new open-set method. |
| All evaluated label-free selectors fail to beat LOF | **A bounded negative result.** It does not establish that label-free selection generally fails. MetaEns is an explicit threat to that broader claim. |

### Against the three named threats

**ADBench.**  
Do not distinguish your paper merely by replacing “1% labeled anomalies” with “five labeled anomalies.” Your distinction must be:

1. absolute, matched label budgets;
2. selection versus fitting versus combination;
3. explicit control of anomaly-type coverage;
4. strong simple classifiers included as first-class baselines.

**OneShot, as described in the supplied material.**  
Do not claim first use of 1–8 labels for detector weighting. The useful contrast is:

> Before using scarce labels to weight detectors, test whether fitting a classifier directly is a better use of them.

Your time-series evidence does **not** rebut OneShot: these tasks have one anomaly event, overlapping windows, and a nearly optimal default. They do not provide a convincing independent few-event generalization test.

**DRA.**  
Do not claim discovery of the known-versus-unknown anomaly problem. Your opportunity is to show what happens in **extreme-label-budget tabular detection**, against DRA and other relevant approaches, and whether a trivial baseline is unexpectedly competitive.

### Two reporting corrections are necessary now

1. **“The hedge is best in every setting” needs matched comparisons.** The brief reports LR gap closure of 107% in one experiment and hedge closure of 88% in a matched stress-test setting where LR is 61%. Those numbers cannot be compared across experiments. Reconcile cohorts, draws, and denominators before writing the abstract.

2. **The oracle terminology must remain precise.** Say “the per-task oracle over 25 unsupervised detectors.” A supervised model can legitimately exceed that reference, producing negative reference regret.

---

## (b) Proposed paper

### Title

**A Few Anomaly Labels: Fit, Select, or Combine?**

Subtitle:

**Label-Budget Allocation and Unseen-Type Generalization in Tabular Anomaly Detection**

This asks a consequential question without promising that five examples are universally sufficient.

### One-sentence thesis

> At tiny, matched anomaly-label budgets, fitting a classifier can outperform detector selection, but the benefit depends on anomaly-type coverage; combining it with an unsupervised default offers a simple baseline for retaining unseen-type detection.

Use stronger language—“does outperform” and “retains”—only after confirmation.

### Three contributions

1. **A budget-matched evaluation of how anomaly labels are used.**  
   Compare label-free deployment, label-based detector selection/weighting, direct classifier fitting, weakly supervised AD, and supervised–unsupervised combinations using identical labeled examples and test sets.

2. **An anomaly-type coverage protocol.**  
   Separate representative-example evaluation from missing-type evaluation, using semantic anomaly categories where available and clearly labeled geometric stress tests elsewhere. Report performance on known types, unknown types, and their mixtures.

3. **A strong simple-baseline finding.**  
   Quantify when ordinary classifiers outperform detector selection and whether the pre-specified LR–LOF rank average remains competitive with specialist weakly supervised and open-set methods. Release the splits, budgets, type annotations, draws, and predictions.

**Contribution 3 is conditional on the results. Do not present rank averaging itself as a new algorithm.**

### Where ADReal fits

ADReal becomes the **evaluation substrate**, not a competing headline.

Retain:

- grouped, leakage-controlled normal splits;
- anomaly deduplication;
- the construction audit;
- frozen task definitions;
- clear provenance and source-family identifiers.

Change the interpretation:

- Hardening defines a **residual-anomaly regime**. It is not evidence that the retained anomalies are universally more realistic.
- Exact feature collisions indicate label ambiguity in the available representation, not necessarily incorrect original labels.
- Post-scoring inclusion floors are detector-pool-dependent. Include a sensitivity analysis on the pre-floor constructed tasks.

The most valuable new benchmark artifact is not another detector leaderboard. It is the **fixed label-budget × type-coverage evaluation suite**.

### Where the label-free negative result fits

Keep a concise opening result:

> “Before allocating anomaly labels, we establish the performance of strong zero-label references.”

Move the long selector catalogue and detailed detector-family analysis to the appendix. Include modern meta-learned competitors before repeating a broad negative headline.

**Cut time series from the main few-label claims.** Retain it as a limitation or appendix case study explaining why windows are not independent labeled incidents.

---

## (c) Confirmatory design

## 1. Separate confirmation from development

The existing tasks have already informed the pivot. Repeating more draws on them improves precision but does **not** create independent confirmation.

Use:

- **Development:** the already-inspected ADReal tasks.
- **Confirmation:** eligible, previously uninspected tabular source-dataset families from the raw collections already on disk.

Freeze eligibility, hardening, preprocessing, methods, and endpoints before evaluating the confirmation set. Do not select confirmation tasks using detector performance.

Group related tasks by **original source dataset**, not benchmark task name. Different one-versus-rest conversions of the same table must not cross development and confirmation boundaries.

If no genuinely uninspected families remain, say so. Use grouped outer evaluation, but describe the resulting evidence as internally validated rather than pristine confirmation.

Also reconcile **112 analyzed versus 113 listed tabular tasks** and publish the exclusion reason.

## 2. Required baselines

### Core comparison set

| Category | Baselines | Why required |
|---|---|---|
| Fixed unsupervised | Frozen LOF \(k=10\); development-chosen global best | Establish whether per-task adaptation earns its cost |
| Label-free selection | Strongest existing injection selector; **MetaOD** and **MetaEns-style** approaches | The present negative claim does not cover strong transfer-based selection |
| Label-based selection | Your positive-versus-normal selector; label-based detector weighting; **OneShot-style weighting** where transferable | Tests the central “fit versus select” claim |
| Plain supervised | Regularized class-balanced **LR**, linear SVM, RF, gradient-boosted trees | Avoid attributing a generic classification result to AD-specific machinery |
| Pretrained tabular classification | A **TabPFN-style classifier** | A serious few-shot tabular comparator; disclose pretraining and context limits |
| Score-feature learning | **XGBOD** | Directly bridges detector outputs and supervised fitting |
| Weakly supervised AD | **DevNet, PReNet, FEAWAD, RoSAS** | Established methods designed to exploit scarce anomaly labels |
| Open-set supervised AD | **DRA** | Direct competitor to the missing-type claim |
| Simple combinations | LR–LOF rank average; strongest plain-classifier–LOF combination; strongest specialist–LOF combination | Tests whether the finding is generic combination rather than LR-specific |

For DRA and any architecture not directly applicable to tabular vectors, document the adaptation. **A poorly adapted image method is not a meaningful defeated baseline.**

For MetaOD/MetaEns-style methods:

- train on development families only;
- exclude overlapping source datasets;
- use grouped outer evaluation;
- distinguish faithful implementation from an approximation.

For TabPFN-style methods, disclose pretrained status and possible source-data overlap. They are not equivalent to models trained only on this benchmark.

### Fairness rules

- Identical anomaly labels and available normal data.
- No test-label hyperparameter tuning.
- Globally fixed hyperparameters or development-only tuning.
- No treating duplicated positive examples as additional labels.
- Explicit accounting of feature engineering, pretraining, and source-task supervision.
- Preserve each method’s documented training assumptions; identify adaptations.

The full comparison is **not a few-hour GPU exercise**. Prioritize the \(k=5\) primary experiment before extending every expensive method across every budget.

## 3. Label budgets and split structure

Use:

\[
k\in\{0,1,2,3,5,10,20\}.
\]

Primary endpoint: **\(k=5\)**.  
Important secondary endpoints: \(k=1,2,10\).

Create a disjoint anomaly calibration reservoir and evaluation set **before drawing labels**. Use nested labeled subsets where possible, so the first two examples at \(k=5\) are the examples used at \(k=2\).

Crucially:

> Every method and every budget should be evaluated on the same untouched test set—not on a test set that changes as more positives are removed.

For cheap methods, use at least 30 label draws. For expensive methods, use at least 10 matched draws at the primary budget, with training-seed variability assessed separately.

Add a secondary “1% of available calibration anomalies” condition to connect to ADBench, but do not substitute it for the absolute-budget experiment.

State the resource assumption explicitly:

> The budget counts supplied, verified anomaly examples—not all annotations required to discover those examples.

Without this qualification, “five labels” can imply an unjustified operational acquisition cost.

## 4. Define unseen types defensibly

### Primary: semantic types

Use existing original labels or metadata:

- **OvrBench:** original classes collapsed into the anomaly class. Label from one anomaly class and test on excluded anomaly classes, where the construction retains this information.
- **Other sources:** documented attack categories, fault modes, diagnoses, or event categories when available.

Release the mapping and provenance. Do not infer semantic categories from binary labels.

A minimum of **20 independent source-dataset families with usable semantic exclusions** is a reasonable project gate. Fewer would make a broad open-set headline fragile; additional label draws cannot replace independent families.

### Secondary: geometric stress tests

For binary-only sources:

- retain the existing k-means exclusion;
- add a materially different, fixed partitioning approach, such as hierarchical clustering;
- pre-specify cluster count, minimum size, and handling of unclustered observations;
- use evaluation labels only to construct evaluator-side partitions, never as learner inputs.

Call these **cluster-exclusion stress tests**, not ground-truth anomaly types.

### Three evaluation conditions

1. **Representative examples:** labels sampled from the anomaly mixture.
2. **Unknown-only evaluation:** labels from one type; evaluation exclusively on other types.
3. **Mixed deployment:** evaluation contains known and unknown anomalies.

For mixed deployment, vary the unknown-anomaly share:

\[
q\in\{0,\;0.25,\;0.5,\;0.75,\;1\}.
\]

Keep total anomaly prevalence fixed while changing \(q\). Otherwise, prevalence and novelty become confounded.

Balance or average across eligible labeled types. Do not report only the easiest type to learn.

## 5. Prevalence and operational metrics

Use:

- **2% prevalence:** primary deployment-oriented condition;
- **1% and 5%:** sensitivity conditions;
- **native ADReal prevalence:** continuity with existing results.

When there are insufficient held-out normals, use prevalence-standardized, weighted precision–recall evaluation under an explicit prior-shift assumption. Do not duplicate observations to manufacture a larger test set.

Report:

- AP at the specified prevalence;
- AP improvement over the frozen default;
- gap closure against the frozen unsupervised-pool oracle;
- recall at 1% and 5% FPR where normal sample sizes support estimation;
- per-task loss frequency and lower-tail degradation.

**Normalized AP does not remove all prevalence dependence.** Its baseline normalization is useful, but it does not replace this analysis.

Freeze the hedge’s ranking definition now. If it uses test-batch ranks, describe it as **batch/transductive score combination**. Do not silently reinterpret it as an inductive, single-observation deployment method.

## 6. Statistics

The independent inferential unit is the **original source-dataset family**, not:

- individual anomalies;
- label draws;
- overlapping windows;
- multiple derived tasks from the same original table.

Use a paired, source-family-clustered bootstrap. Average draws within each task, then aggregate using a pre-specified family-weighting rule. Report conventional task-macro results as a sensitivity analysis.

For gap closure, use the ratio of aggregate gains:

\[
G(M)=
\frac{\overline{A_M-A_D}}
{\overline{A_O-A_D}},
\]

where \(D\) is the frozen default and \(O\) is the frozen unsupervised-pool oracle.

Do **not** average per-task gap ratios: small denominators make them unstable.

The primary statistical effect should remain the paired AP difference. Gap closure is an interpretable secondary effect size.

Pre-register a small primary comparison family and adjust it, for example with Holm correction. Report confidence intervals, not only signed-rank \(p\)-values.

## 7. Numeric decision rules

Freeze these before confirmation. Here \(A\) denotes AP at **2% prevalence**, and \(H\) is the pre-specified hedge.

| Component | Confirmatory success bar | Result that would undermine or falsify it |
|---|---|---|
| **Fit rather than select, representative labels, \(k=5\)** | LR exceeds the strongest label-based selector by **≥0.01 AP**, with paired 95% CI above zero; LR closes **≥50%** of the frozen-default oracle gap | Selector exceeds LR by ≥0.01 AP with CI above zero; or the upper CI for LR gap closure is below 50% |
| **Type coverage matters** | LR’s gain over the default falls by **≥0.02 AP** from representative to semantic unknown-only evaluation, with CI supporting a decrease | Upper CI on the decrease is below 0.02 AP; the claimed material coverage effect is absent |
| **Hedge helps unknown types** | On semantic unknown-only evaluation, \(H\) closes **≥15%** of the default-to-oracle gap, with AP-gain CI above zero | Upper CI on AP gain is ≤0; or a supported loss of ≥0.01 AP |
| **Hedge retains representative-label utility** | \(H\) closes **≥50%** of the gap with representative labels | Upper CI on closure is below 50% |
| **Simple baseline is competitive with specialists** | \(H\) is within **0.01 AP** of the strongest specialist in both representative and unknown-only settings, using a pre-specified noninferiority analysis | A specialist beats \(H\) by **>0.02 AP** with supported differences in either primary setting |
| **Result is not peculiar to hardening** | Training-versus-selection ordering persists without hardening, and hedge is not worse than default by >0.01 AP on unknown types | Supported reversal outside the hardened regime |

These distinguish **positive evidence, contrary evidence, and inconclusive intervals**. A point estimate below a target with a wide interval is not definitive falsification—but it is not grounds for the intended headline either.

Two further boundaries:

- If RF, boosted trees, or TabPFN-style models beat LR, retain the broader **fit-versus-select** conclusion and stop centering LR.
- If only geometric exclusions exhibit failure and semantic exclusions do not, abandon the open-set generalization headline.

---

## (d) The objection most likely to sink it

> **“This is already the motivation of open-set supervised anomaly detection. You manufactured unknown types with k-means, omitted the relevant methods, and recovered them with a standard ensemble.”**

This combines the paper’s two weakest current points: semantic validity and missing incumbents.

### The experiment that pre-empts it

Run a **frozen \(k=5\), semantic leave-type-out comparison on independent source families**, at 2% prevalence, including:

- strongest label-based selector/weighting method;
- LR and the strongest ordinary classifier;
- XGBOD;
- strongest applicable weakly supervised method;
- DRA;
- the pre-specified LR–LOF hedge;
- strongest specialist–LOF hedge.

Evaluate unknown-only and 50/50 known–unknown mixtures.

Repeat the same comparison on:

1. the frozen hardened tasks;
2. matched tasks without anomaly hardening.

This one experiment answers:

- Is the coverage problem semantic rather than clustering-induced?
- Does the result survive serious open-set competitors?
- Is hardening creating a special advantage for LOF?
- Is the hedge a useful simple baseline or just a weak substitute for an established method?

**If DRA or RoSAS decisively dominates the hedge, do not invent another combination to rescue it.** Reframe the result as a label-allocation benchmark with a strong specialist winner. If that leaves only “labels help,” the pivot has failed its novelty bar.

---

## (e) Venue and honest odds

### Best target: **TMLR**

This is best suited to a carefully controlled empirical contribution with useful negative results, strong simple baselines, and explicit limits. It does not require dressing rank averaging up as a new algorithm.

**DMKD is the strongest alternative**, particularly if the paper becomes a comprehensive evaluation of label-budget allocation and type coverage.

| Venue | Fit | Rough acceptance judgment after successful confirmation |
|---|---|---:|
| **TMLR** | Best overall fit | **45–60%** |
| **DMKD** | Strong empirical/data-mining fit | **40–55%** |
| **KDD** | Needs a decisive deployment result or stronger methodological contribution | **15–25%** |
| **AAAI** | Plausible, but novelty remains a challenge | **15–25%** |
| **NeurIPS D&B** | Requires the new protocol/resource to be the central reusable contribution | **10–20%** |
| **ICLR** | Weak fit without a deeper methodological or conceptual advance | **5–10%** |

These are subjective strategic ranges, not calibrated forecasts. **For the current evidence alone, before semantic-type confirmation and specialist baselines, I would put TMLR below 25%.**

## Final decision

Proceed—but with a narrower claim and a clear stop condition.

The paper should not be:

> “Five anomalies beat unsupervised detection, and we propose a hedge.”

It should be:

> **“At tiny label budgets, how should anomaly labels be used—and how does the answer change when those labels fail to cover future anomaly types?”**

The decisive next investment is **semantic leave-type-out evaluation against strong weakly supervised and open-set baselines**, not another selector, probe, router, or portfolio search.