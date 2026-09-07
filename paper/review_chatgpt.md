# Peer Review: *No Easy Wins: A Contamination-Controlled Benchmark for Evaluating Anomaly Detection and Model Selection*

## Overall assessment

This is a timely and potentially valuable benchmark paper. The strongest contribution is not SPARC itself, but the attempt to make anomaly-detection evaluation substantially more defensible by explicitly controlling duplicate-induced leakage, anomalies that are exact copies of normals, duplicated anomaly templates, extreme base rates, trivially separable cases, and tasks on which model selection is effectively meaningless. The paper is also right to insist that a per-dataset model selector should be compared against the much harder baseline of **doing no per-dataset selection at all and deploying one strong detector everywhere**, rather than only against random model choice.

However, I do not think the current evidence fully supports the strongest model-selection claims. In particular, the headline SPARC result—mean regret 0.113 versus 0.130 for the best fixed detector, with \(p=0.04\)—is statistically and procedurally fragile. The fixed detector is itself selected using the full labeled benchmark; SPARC's design choices appear to have been developed/ablated on the same 151 tasks; the reported significance is marginal and apparently uncorrected for the number of comparisons made; the tasks are unlikely to be statistically independent at the level assumed by the Wilcoxon test; and most of the SPARC advantage comes from only 39 OddBench tasks. In addition, SPARC is allowed to restrict its candidate set to classical detectors, while the paper does not show that the competing selectors receive the same candidate-pool treatment.

My current view is therefore: **the benchmark idea is strong and potentially publishable, but the central claims about SPARC and about the general failure of unsupervised model-selection criteria need a more careful experimental design and narrower wording.** I would favor a major revision / weak reject in the current form, with a clear path to acceptance if the issues below are addressed.

---

## 1. Soundness of the central claims

### 1.1 The contamination-audit claim is the strongest and best-supported part

Sections 3.2–3.5 make a convincing case that the source benchmarks contain artifacts that can materially affect detector and selector evaluation. The paper gives concrete counts rather than only conceptual objections. For example, Section 3.3 reports 4,869 anomalies identical to normal points across 32 candidate tasks and 8,285 normal points that would leak across random splits in 62 of 173 tasks. Table 2 is particularly effective: the `http` example, with 2,211 anomaly rows but only 72 distinct anomaly vectors and 2,092 anomalies exactly matching a normal vector, illustrates a genuine pathology that should not be silently treated as 2,211 independent anomalous cases.

The grouped normal split in Section 3.3 is also a clear improvement over naive random splitting: exact duplicate normal vectors and all copies are assigned to the same split, and the time-series case uses contiguous blocks plus purging of boundary-straddling windows. Figure 1 gives a useful overview of the construction pipeline.

I therefore find the general claim that **benchmark contamination is widespread enough to warrant systematic controls** well supported.

### 1.2 The claim that ADReal is “detector-free” needs qualification

The paper repeatedly describes the construction as a “detector-free pipeline” (Introduction, Section 3, Figure 1), but Section 3 explicitly applies two **post-scoring floors computed from the detector pool**: the separability floor and the selection-trivial floor. The separability floor removes 17 tasks for which the oracle-best detector has \(ap_{norm}\leq 0.05\), and the selection-trivial floor is also defined through detector performance.

This is not necessarily wrong, but the terminology is too strong. The *pre-scoring contamination controls* may be detector-free; the final membership of the 151-task benchmark is not.

This distinction matters for two reasons:

1. A benchmark intended to compare **future anomaly detectors** arguably should not discard tasks merely because the current detector pool cannot solve them. Such tasks may be exactly where a future method demonstrates progress.
2. A benchmark intended to compare **model selectors** has a more defensible reason to exclude tasks on which no candidate detector has signal.

I suggest splitting the release into at least two tracks: a contamination-cleaned set before detector-dependent floors, and the 151-task “selection-evaluable” subset after the floors. The claims should then be scoped accordingly.

### 1.3 “Popular unsupervised model-selection criteria fail” is too broad for the protocol tested

Section 5 defines a very specific setting: the selector sees a validation set containing **known normals only**. That is a legitimate and useful one-class model-selection setting, but it is not equivalent to all label-free or unsupervised model-selection settings.

This is especially important for consensus, model-centrality, and HITS. The paper criticizes prior transductive evaluation because those methods see the unlabeled sample that contains anomalies. But in many unsupervised deployments, this is not label leakage: the deployment sample is unlabeled, and using its unlabeled score structure is precisely what the method is designed to do. “Label-free” does not imply “anomaly-free input.”

Thus, the result in Section 5 is sounder if stated as:

> Under a clean-normal-validation / one-class selection protocol, the tested agreement-based and internal criteria do not outperform a strong fixed detector.

That is a narrower but defensible conclusion. The current wording in the Introduction and Conclusion sometimes reads as a field-wide conclusion about unsupervised model selection.

### 1.4 The detector-comparison claims should also be softened

Section 4 and Figure 5 conclude that classical detectors “carry this benchmark” and that deep methods do not beat classical ones. The result is descriptive for the configurations tested, but the comparison is not symmetric: several classical methods receive hyperparameter sweeps (six LOF variants, six KNN variants, four HBOS variants, four PCA variants), whereas each deep method receives one configuration.

Consequently, the paper can say that **under this benchmark harness and these configurations**, classical methods dominate. It should not imply a broader conclusion about deep versus classical anomaly detection.

---

## 2. The best-fixed-detector baseline: right idea, but the current implementation needs stronger statistical treatment

### 2.1 The baseline is conceptually the right bar

I strongly support the inclusion of the best-single-detector baseline in Section 5 and Figure 6. A selector that uses dataset-specific machinery should not be considered useful merely because it beats random choice. If a single robust detector performs better on average, then the selector has not justified its additional complexity.

This is an important framing contribution.

### 2.2 But LOF \(k=10\) is an in-sample benchmark-optimal fixed detector, not an externally available deployment baseline

The best fixed detector is defined as the detector with the lowest average regret on the same 151 benchmark tasks used for the final comparison. This uses true anomaly labels across the whole benchmark to decide that LOF \(k=10\) is the fixed reference.

That makes it an intentionally strong reference, but it is not literally a label-free baseline that a practitioner could choose *a priori* for a new domain. It is better described as the **benchmark-optimal fixed detector**.

More importantly, because the comparator is selected using the same evaluation tasks, the statistical comparison is data-dependent. The paper should provide at least one nested across-dataset evaluation, for example:

- leave-one-task-out or K-fold cross-validation across tasks, where the best fixed detector is chosen on the other tasks and evaluated on the held-out task;
- preferably grouped by original source dataset/domain rather than random task folds;
- an external or predeclared fixed baseline, such as a standard LOF configuration chosen before looking at ADReal results.

A particularly informative baseline would be the best **fixed detector per modality** learned only from training benchmark tasks, because a practitioner normally knows whether the dataset is tabular or time-series. The current global LOF baseline may be either too weak or too strong depending on benchmark composition.

### 2.3 The entropy / mass-volume result is substantially more convincing than the SPARC-vs-fixed result

Figure 6 and Section 5 report:

- best fixed detector: regret 0.130;
- mass-volume: 0.174;
- entropy / EM: 0.176;
- both significantly worse than the fixed detector at \(p=0.002\).

The absolute gaps of 0.044–0.046 are large relative to the 0.017 SPARC gain, and the reported \(p=0.002\) is much less marginal. Within the paper's normal-only validation protocol, I find the conclusion that these two internal criteria are worse than simply using one strong detector reasonably convincing.

However, it must be caveated in three ways:

1. the result is for the **normal-only validation protocol**, not all unsupervised/transductive settings;
2. the benchmark hardening explicitly removes marginally easy anomalies and therefore shifts the benchmark toward local/joint-structure anomalies (Sections 7–8);
3. the terminology in Section 5 is confusing: “Entropy (excess-mass)” and “Entropy (EM)” appear to conflate entropy with excess mass. The manuscript should define exactly which criterion is implemented and use one name consistently.

### 2.4 The SPARC 0.113 vs 0.130, \(p=0.04\) result should be treated as suggestive, not definitive

The mean regret reduction is 0.017, or about 13% relative to the fixed baseline's regret. That is useful if stable, but \(p=0.04\) is a borderline result.

Several issues make the current inferential claim weaker than the prose suggests:

- Multiple methods are compared with the fixed reference, and additional subgroup tests are reported. I do not see a correction for multiple comparisons or a declaration that SPARC-vs-fixed was the single preregistered primary hypothesis.
- SPARC appears to have been developed on the same benchmark used for the final test. Table 7 explicitly shows design choices selected because they improve regret on the 151 tasks.
- The resampling fraction is “about 40%,” which is a hyperparameter even if it is not tuned per dataset.
- The tasks may not be independent experimental units; many come from the same benchmark families and potentially from related original datasets.
- The paper promises median and tail information in Section 3.5, but the headline result is presented mainly as mean regret plus a Wilcoxon \(p\)-value. A Wilcoxon test is about paired ranks, while the visual emphasis is on means.

The central SPARC claim would be much stronger with a held-out benchmark-development split, grouped resampling over source datasets, and confidence intervals on the paired regret improvement.

### 2.5 “Only SPARC significantly beats the fixed detector” is not equivalent to “SPARC is significantly better than the other injection selector”

This distinction is important. Goswami's injection selector has regret 0.122 versus SPARC's 0.113, but the paper reports only that Goswami does not significantly beat the fixed baseline (\(p=0.88\)) whereas SPARC does (\(p=0.04\)).

A significant-vs-nonsignificant contrast does **not** establish a significant difference between SPARC and Goswami. If the manuscript wants to claim SPARC is the better selector, it should report the direct paired SPARC-vs-Goswami comparison and its confidence interval/effect size.

The safe current statement is only that SPARC is the only tested method whose comparison to the fixed baseline crosses the chosen significance threshold.

---

## 3. SPARC: compelling idea, but the “single probe / no tuning” and generality claims are over-sold

### 3.1 The probe itself is simple and well motivated

Section 6 is one of the clearest parts of the paper. Resampling a subset of features from their observed marginals preserves coordinate-level realism while breaking joint dependencies. Using probe-vs-normal ROC-AUC to grade detectors is easy to understand and cheap to implement.

The observed-value ablation in Table 7 is also sensible: avoiding interpolated values prevents discrete-valued features from producing obviously artificial coordinates.

### 3.2 “Single probe” should mean one corruption *mechanism*, not imply an absence of stochastic variability

The probe is stochastic: a random subset of features is resampled, and replacement values are sampled from empirical marginals. The paper should report sensitivity across random seeds or repeated probe draws. A single corruption family may be sufficient, but the numerical result should not depend materially on one stochastic realization.

The phrase “single probe” is therefore potentially misleading unless the implementation averages over a sufficiently large synthetic sample and the seed variance is shown to be negligible.

### 3.3 “No tuning” is not accurate as currently written

SPARC has at least one explicit parameter: the fraction of resampled features, approximately 40%. In addition, Table 7 shows that the observed-value sampling choice and the restriction to classical candidates materially improve the benchmark result.

These are legitimate design choices, but they are method development. “No tuning” should be replaced by “no per-dataset tuning” unless the authors can demonstrate that all choices, including 40%, were fixed independently of ADReal.

At minimum I would require a sensitivity curve over the resampling fraction and multiple random seeds. Better still, choose the probe design on a development subset of datasets and report the final result on a disjoint test subset.

### 3.4 The candidate-set asymmetry is a serious fairness issue

Section 6 states that SPARC selects only among **classical detectors**, because deep detectors tend to overfit the synthetic probe, while the oracle still includes the deep models. Section 5, however, describes the competing selectors as selecting from the fixed 32-detector pool.

If SPARC is allowed to improve by excluding a class of confusing candidates, then competing selectors should also be evaluated on the same classical-only candidate pool. The fact that deep detectors rarely supply the oracle does not resolve this: non-oracle candidates can still degrade a selector's ranking mechanism.

I would require a 2x comparison:

- all selectors on all 32 candidates;
- all selectors on the identical classical-only candidate set.

Otherwise, part of SPARC's advantage may be candidate-pool engineering rather than the probe itself.

### 3.5 The OddBench-concentrated win is interesting, but does not yet establish a broadly useful selector

Table 6 and Figure 7 are appropriately revealing:

- time series: SPARC 0.037 vs fixed 0.038;
- OvrBench: SPARC 0.137 vs fixed 0.145;
- OddBench: SPARC 0.153 vs fixed 0.198, \(p=0.029\).

The paper deserves credit for exposing this decomposition, and Section 8 explicitly acknowledges that the overall win comes “almost entirely” from OddBench. That is a good limitation statement.

Still, the Conclusion's phrase that SPARC wins “where it matters” is too rhetorical. What the evidence currently supports is more specific: **SPARC appears useful on the OddBench subset, where the oracle detector varies substantially across tasks; it adds little on the other two large subsets.**

This result is potentially meaningful because OddBench is described as real-world semantic-anomaly data, but it needs stronger support:

- paired per-dataset scatter plots or win/tie/loss counts;
- median improvement and IQR, not only means;
- a direct SPARC-vs-Goswami test on OddBench;
- a source-aware confidence interval;
- correction or explicit exploratory labeling for the subgroup \(p=0.029\).

The current 39-task subgroup result is promising, not definitive.

---

## 4. Benchmark construction and contamination controls

### 4.1 Deduplication: strong motivation, but exact-match-only control is narrower than the prose suggests

The exact-duplicate handling is a strength. Keeping duplicate normals but grouping them into one split preserves their empirical frequency while preventing train/test identity leakage. Removing anomalies identical to normals is also defensible if the benchmark representation literally makes them indistinguishable.

However:

- Section 3.5 refers to “near-identical” anomalies, but the actual rule is exact equality after float32 conversion and six-decimal rounding.
- “Exact match at float32 precision” and “rounded to six decimals” are not quite the same notion and should be stated more precisely.
- Cross-source deduplication “by name and dimensionality” is weak. Renamed copies, reordered columns, standardized copies, or subsets may survive.

For a benchmark paper, I would expect content-based dataset fingerprints or explicit cross-source overlap checks, plus a sensitivity analysis for near-duplicate rows.

### 4.2 Leak-free grouped splitting is good, but all preprocessing must be train-fitted

The grouped 60/20/20 split is one of the most defensible design choices in the paper. For time series, contiguous blocks with purging is also appropriate.

The manuscript should nevertheless state explicitly that every learned preprocessing transformation—scaling, imputation, encoding, PCA where relevant, and any data-dependent detector preprocessing—is fitted using the training normals only and then applied to validation/test. “Same standardized features” is not enough to establish absence of preprocessing leakage.

### 4.3 The base-rate normalization is reasonable but does not by itself prove comparability across prevalence

Section 3.4 defines

\[
ap_{norm} = \frac{AP-r}{1-r},
\]

mapping the random-ranking baseline \(AP=r\) to 0 and a perfect ranking to 1. This is a useful normalization.

But AP depends on prevalence in more than an additive-baseline way. The transformation does not automatically establish that the same underlying ranking quality will have the same \(ap_{norm}\) at different anomaly rates. Thus the paper's language that the metric is “comparable across tasks of different base rates” is stronger than what is demonstrated.

This matters because Table 3 shows very different median prevalences: 0.48 for OvrBench, 0.35 for OddBench, and 0.15 for time series.

I would ask for an empirical prevalence-sensitivity experiment: for datasets with enough anomalies/normals, evaluate the same detector rankings at several controlled anomaly fractions and show how stable \(ap_{norm}\), detector rankings, and selector conclusions are. Reporting a secondary metric such as ROC-AUC or PR-gain would also help.

The cap at 0.5 prevents the denominator from becoming small, but 50% anomalies is still far from the regime usually associated with anomaly detection. The paper should explain why 0.5, rather than a much lower common prevalence, is the right canonical cap.

### 4.4 Hardening is the most consequential benchmark-design choice and needs stronger counterfactual analysis

Section 3.2 removes anomalies that are easy under per-feature rarity rules and PCA-whitened coordinates. Figures 2 and 3 make the intended effect visually clear.

However, Sections 7–8 also admit the key consequence: the hardening filter preferentially removes marginal anomalies and leaves locally sparse, joint-structure anomalies. This is exactly the regime in which LOF/KNN and SPARC's dependency-breaking probe should be favored.

That creates a potential **benchmark–method alignment**: the benchmark is intentionally reshaped toward the anomaly geometry that the proposed selector probes.

The paper does acknowledge this, which is good, but acknowledgement is not enough for the headline SPARC claim. I would require one of the following:

1. evaluate SPARC on both a contamination-cleaned-but-not-hardened track and the hard track;
2. stratify anomalies by hardening severity instead of deleting the easy ones;
3. use several hardening definitions and show that the selector ranking is stable.

The current result may be a valid statement about the “hard local/joint-anomaly regime,” but it is not yet evidence of broad model-selection superiority.

There is also a reproducibility issue in the description of the hardening rule. The text says each feature is thresholded at a 5% false-positive rate and then OR'ed. If that means 5% per feature, the family-wise false-positive rate will grow rapidly with dimension. Figure 3 instead suggests a single aggregated severity statistic with a 5% cutoff. The exact calibration must be written mathematically.

Similarly, the claim that retained anomalies “cannot be flagged by any single feature or direction” is too strong if the tested directions are only original coordinates and PCA-whitened axes.

### 4.5 The post-scoring floors are defensible for selector evaluation, but not neutral benchmark curation

The separability floor \(ap_{norm}\le 0.05\) removes tasks on which the current pool cannot beat random. This makes sense if the sole goal is to compare selectors: selecting among equally ineffective detectors is not informative.

But it is less appropriate for a benchmark that also aims to compare future detectors. It removes exactly the datasets on which a genuinely new detector might make the largest advance.

The “selection-trivial floor” is also under-specified: the manuscript says all detectors are “within a hair of the best,” but does not give the threshold in the main text. Even though the two flagged tasks also fail the separability floor, a benchmark paper should specify the rule exactly.

I recommend releasing and reporting results on:

- all contamination-cleaned tasks;
- the separability-qualified subset;
- the final selector-evaluation subset.

### 4.6 The initial curation from 1,192 available datasets to 174 candidates is insufficiently documented

Table 1 shows 1,192 available datasets but only 174 entering the candidate pool. This is a very large preselection step: 754 OvrBench datasets become 76 candidates, 187 OddBench become 46, 200 TSB-AD become 41, etc.

Section 3.1 mentions source selection, fixed-length windowing, triviality pre-filters, and a window-purity filter, but the paper needs a fully auditable inclusion/exclusion ledger. For a datasets-and-benchmarks venue, this is essential.

I would expect a supplementary table listing every source dataset, whether it was included, and the exact exclusion reason.

### 4.7 Statistical independence of the 151 tasks is not established

The paired Wilcoxon test treats tasks as the experimental units. But OvrBench contains one-vs-rest conversions, and multiple tasks may be derived from related source datasets; time-series tasks may likewise share generators or source systems. Equal weighting of all 151 tasks can therefore overstate the effective sample size.

This is especially important for \(p=0.04\).

The paper should group tasks by original parent dataset / data-generating source and use a group-aware bootstrap, permutation test, or hierarchical analysis. It should also report source-balanced averages because the overall result is heavily influenced by the benchmark composition (66 OvrBench, 39 OddBench, 38 time-series, but only 8 total ADBench+DAMI tasks).

---

## 5. Presentation and clarity

The manuscript is generally concise and readable. Figure 1 is a good high-level pipeline; Table 2 provides memorable evidence for the duplication problem; Figure 6 makes the fixed-detector argument immediately visible; and Figure 7 is an important decomposition that prevents the SPARC result from appearing more universal than it is.

Several presentation issues should nevertheless be fixed:

1. **Method-count mismatch.** Section 5 says eight label-free selectors are evaluated: entropy/EM, mass-volume, consensus, model-centrality, HITS, UDR, IFOREST-R, and IREOS, plus Goswami and SPARC. Figure 6 does not display UDR, IFOREST-R, or IREOS, and Table 6 omits even more methods. The paper should report every stated method or explain exclusions/failures.
2. **Entropy vs excess mass terminology.** “Entropy (excess-mass)” / “Entropy (EM)” is ambiguous. EM usually reads as excess mass in the paper's own description. Define the implemented formula unambiguously.
3. **Missing uncertainty visualization.** Figure 6 is a bar chart of mean regret with no confidence intervals or paired-difference distribution, yet the central claim is statistical. A paired improvement plot, violin/box plot, or bootstrap CI would be more informative.
4. **Section 3.5 promises median and tail reporting**, but these are not prominent in the results shown. Please report them explicitly.
5. **Several thresholds are insufficiently justified or incompletely specified**, including the 5% hardening threshold, the 6.7% time-series threshold, the selection-trivial threshold, the 100-anomaly floor, and the 0.5 base-rate cap.
6. **The time-series scope needs clearer wording.** The benchmark windows multivariate series into fixed-length feature vectors and evaluates them with the same detector pool. This is a valid benchmark design, but it is not equivalent to evaluating the full range of native sequence-aware time-series anomaly detectors. The claims should say “windowed multivariate time-series tasks.”
7. Some rhetorical phrases are too strong for the evidence, particularly “the one method that beats the reference,” “where it matters,” and “no tuning.” These should be replaced with more literal statements.

---

## 6. Strongest objections and what would address them

### Objection 1: The SPARC significance claim is not a clean out-of-sample result

**Why this matters:** The same 151 tasks appear to be used to develop SPARC, choose key design decisions (Table 7), choose the best fixed detector, and test the final hypothesis. A \(p=0.04\) result in this setting is vulnerable to researcher degrees of freedom.

**What would address it:** Split benchmark tasks into development and final evaluation sets, preferably grouped by original source dataset/domain. Freeze the 40% fraction, candidate restriction, and probe design on development data. Then evaluate once on held-out tasks. Alternatively, use nested cross-validation across dataset groups and aggregate held-out paired differences.

### Objection 2: SPARC receives a favorable candidate restriction that competitors may not receive

**Why this matters:** Section 6 explicitly excludes deep detectors from SPARC selection because they hurt SPARC, but the competing selectors are described as operating on the full 32-detector pool.

**What would address it:** Re-run every selector on the same classical-only pool and on the full 32-detector pool. Report both. If SPARC remains best under matched candidate sets, the claim becomes much stronger.

### Objection 3: The benchmark hardening may create the anomaly geometry that SPARC is designed to detect

**Why this matters:** Sections 7–8 admit that hardening removes marginal anomalies and retains local/joint anomalies; SPARC explicitly injects joint-dependency violations. This could create method-specific benchmark alignment.

**What would address it:** Add a non-hardened contamination-clean track, stratify by anomaly-hardness level, and/or use alternative hardening rules. Show that SPARC's ranking advantage is not an artifact of the chosen hardening operator.

### Objection 4: The fixed-detector baseline is selected in-sample from labeled test tasks

**Why this matters:** LOF \(k=10\) is a useful strong bar, but it is benchmark-optimized using the same task labels on which significance is reported.

**What would address it:** Use a nested or leave-source-out procedure for selecting the fixed detector. Also report a priori fixed baselines and best-fixed-per-modality baselines.

### Objection 5: The 151 tasks may not be independent

**Why this matters:** The Wilcoxon \(p=0.04\) assumes the paired task differences can reasonably be treated as independent. OvrBench one-vs-rest tasks and related time-series tasks may violate this.

**What would address it:** Identify parent/source groups and use cluster-aware inference. Report effective group counts, source-balanced means, and hierarchical/bootstrap confidence intervals.

### Objection 6: “Only SPARC is significant” may overstate separation from the Goswami injection baseline

**Why this matters:** SPARC has regret 0.113 and Goswami 0.122, but the paper does not show that SPARC is significantly better than Goswami. One being significant against fixed and the other not being significant against fixed is not a significant pairwise difference.

**What would address it:** Report the direct SPARC-vs-Goswami paired test, paired effect size, confidence interval, and source-wise decomposition.

### Objection 7: The benchmark is not fully reproducible from the paper

**Why this matters:** The large 1,192-to-174 preselection, some hardening details, and the selection-trivial threshold are not fully specified.

**What would address it:** Add an appendix with exact formulas, all thresholds, windowing rules, train-only preprocessing details, dataset-level inclusion/exclusion reasons, random seeds, and full results for every selector.

---

## Required analyses I would ask for before acceptance

1. **Matched candidate-pool experiment:** all selectors on classical-only candidates and all 32 candidates.
2. **Nested across-dataset evaluation:** choose the best fixed detector and SPARC design choices without using the held-out task/group.
3. **Direct SPARC-vs-Goswami comparison.**
4. **Multiplicity-aware or clearly pre-specified primary inference**, with confidence intervals and paired effect sizes.
5. **Group-aware statistical testing** by original dataset/source, not only task-level Wilcoxon.
6. **Hardening sensitivity:** non-hardened clean track and at least several hardening thresholds/operators.
7. **SPARC sensitivity:** resampling fractions, random seeds, probe sample size, and deep/classical candidate restriction.
8. **Prevalence sensitivity:** demonstrate that \(ap_{norm}\) and the main rankings remain stable across controlled anomaly rates.
9. **Complete selector table:** report UDR, IFOREST-R, IREOS, model-centrality, and HITS everywhere relevant or explain why they are excluded.
10. **Full curation ledger** from 1,192 source datasets to 174 candidates to 151 benchmark tasks.

---

## Final recommendation

**Recommendation: Weak Reject / Major Revision**

The paper has a strong benchmark thesis and several valuable engineering controls. The best-fixed-detector baseline is an important contribution to evaluation practice, and the evidence that entropy/excess-mass and mass-volume fail to beat that baseline under the paper's clean-normal-validation protocol is fairly persuasive.

The SPARC result is more tentative. Regret 0.113 versus 0.130 is promising, but the \(p=0.04\) headline is too fragile to support the current “only method that significantly beats the fixed detector” framing without a cleaner held-out design, matched candidate pools, cluster-aware inference, and a direct comparison with the Goswami injection selector. The fact that the gain is concentrated almost entirely in OddBench is not fatal—the paper correctly reports it in Table 6, Figure 7, and Section 8—but it changes the interpretation from a general model-selection result to evidence of conditional usefulness in a particular heterogeneous anomaly regime.

If the authors strengthen the experimental separation between benchmark construction, SPARC development, fixed-baseline selection, and final hypothesis testing, and if they report the proposed robustness analyses, I would view ADReal as a potentially strong benchmark contribution.