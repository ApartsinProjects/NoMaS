# Cycle 2: from "no easy wins" to a few-label deployment paper

I did no browsing. Statements about ADBench, DRA, OneShot, and related work come from memory and the brief, and the ones that need checking are marked **[verify]**.

## (a) Novelty

### Already known (do not claim)

- **Labels beat unsupervised detection.** Supervised or semi-supervised methods beat the best unsupervised method with a small labeled fraction (ADBench, DeepSAD, DevNet, and others). That headline is old.
- **Supervision fails on unseen anomaly types.** This is the motivation for open-set supervised AD (DRA, and the "gray and black swans" line **[verify]**).
- **Hybrids with unsupervised signal are standard.** XGBOD, DeepSAD, and unsupervised-score stacking all exist. Rank-averaging a classifier with an unsupervised detector will be called "XGBOD-lite" unless we show something XGBOD does not.
- **A handful of labels can weight or select detectors.** OneShot does this with k=1–8 on time series, and semi-supervised CASH does it for selection.

### Not known, and what the evidence supports

1. **Absolute-k regime against a clairvoyant unsupervised reference.** ADBench's axis is a labeled fraction (1%, 5%, …), which is tens to hundreds of labels on typical tasks. **[verify]** by computing what 1% of train size is on our tasks.
   - Ours is k=1–10.
   - The reference is the per-task test-label oracle over 25 unsupervised detectors. That is stronger than "best unsupervised method on average."
   - The result is that k=5 labels plus a linear model beat that oracle (G=1.07).
2. **Train versus select.** The same k labels are worth more spent on training than on selecting among detectors.
   - Training overtakes selection at k=2, and the gap widens to 1.34–1.44 versus 0.63 at k=10.
   - This directly challenges the premise of OneShot-style weighting and CASH-style selection that labels should pick or weight detectors. This is the sharpest claim.
3. **Type shift and the hedge.** The pre-specified rank-average hedge with a fixed strong unsupervised default is the only configuration that stays above the default under unseen-type labels.
   - Pure LR gives −0.25, the hedge +0.22.
   - Why it is not just XGBOD: the hedge partner is a single fixed default, not stacked features.
   - To claim this, we must show that hedging other learners helps less or more (see (c)).
4. **Boundary condition.** On single-event time series, labels hurt because the default is already near-optimal. A deployment rule needs this, and the other papers do not report it.

### Defensible thesis claim

> With 2–10 labeled anomalies, spend the labels on training a simple classifier, not on selecting among unsupervised detectors, and hedge it with a strong unsupervised default. This is robust where pure supervision is not.

**Make the thesis independent of LR specifically.** If DevNet or TabPFN does better, the claim "training beats selection, and hedging protects" still stands. "Plain LR is best" would be fragile.

### Two ADReal-specific biases to check before any claim

- **Floor selection works against labels.**
  - The separability and selection-trivial floors used unsupervised oracle scores, so they dropped tasks where *no unsupervised detector works*. Those are exactly where labels should help most.
  - Report results on all 173 constructed tasks, including the 22 dropped.
- **Hardening filters on marginal rarity calibrated on normals.** It is detector-free but not supervision-free.
  - It may disadvantage unsupervised marginal methods more than supervised ones.
  - Only an external unhardened replication settles this (see (d)).

Also, LOF k=10 was chosen as the default using test results on these same tasks. For held-out evaluation, freeze it as a pre-specified constant.

## (b) Framing

**Title:** *Spend the Labels on Training, Not Selection: Few-Label Anomaly Detection Against a Strong Unsupervised Default*

**Thesis:** On contamination-controlled tabular tasks, 2–10 labeled anomalies are worth more as training signal for a regularized classifier than as a selector among unsupervised detectors, and hedging that classifier with a fixed unsupervised default keeps the gain under anomaly-type shift, where pure supervision falls below the default.

**Contributions**
1. **Label-efficiency frontier.** Regret against the per-task unsupervised oracle for k=1–50, across prevalence and unlabeled-contamination settings, with strong baselines on a held-out replication pool.
2. **Train-over-select dominance.** The crossover at k≈2, plus the negative result that label-free selection cannot beat a fixed default. This yields a zero-label recommendation (LOF k=10) and a few-label recommendation.
3. **Shift-aware protocol and hedge.** A dose-response evaluation of performance versus label–test anomaly shift, an incumbent-biased label-acquisition protocol, and a deployment rule for when to hedge. Time series is a documented negative boundary.

**What happens to ADReal**
- It becomes the evaluation substrate, shown as "ADReal-FewLabel": released splits, k-draw seeds, shift protocols. It is not the headline.
- **Keep:** the construction summary (about 1.5 pages), key contamination numbers, the detector pool, and the label-free selection result, compressed into a "why not label-free selection" section. That section motivates the pivot.
- **Move to appendix:** Table 2, Figures 2–3, the deep-pool analysis, and the Goswami per-source breakdown.
- **Cut:** the "no easy wins" central-lesson framing and contribution (2) as a standalone claim.
- The contamination audit is a separable short paper. Don't overload this one.

## (c) Confirmatory design

**Data**
- **Frozen exploratory set:** the 112 current tabular tasks. All numbers there are exploratory from now on.
- **Held-out replication pool:** at least 150 tabular tasks built with the same frozen pipeline from the roughly 1,018 unused source datasets. Apply no detector-dependent floors, only the detector-free gates.
- **External unhardened replication:** ADBench's original datasets with their original splits.
- **Time series:** report as a separate stratum with purged event-level splits. Add multi-event TSB-AD series if available. Do not put it in the headline.
- **Additional pool:** the 22 floor-dropped tasks, as a sensitivity pool.

**Methods compared (all with fixed defaults and no per-task tuning, since k≤5 forbids CV)**
- **Ours:** LR, and HYB (rank-average of LR and LOF k=10).
- **Supervised and semi-supervised:**
  - class-weighted GBDT (XGBoost/LightGBM) and RF;
  - nearest-labeled-anomaly distance;
  - PU-style classifier (unlabeled = negative);
  - XGBOD (the key hybrid baseline), DevNet, PReNet, DeepSAD, FEAWAD, RoSAS;
  - DRA, adapted to tabular data;
  - TabPFN-v2-style in-context classification (subsample normals into context, add the k anomalies).
- **Label-based selection and weighting:** SEL (our version), a OneShot-style weighting adapted to tabular data, and stacked LR over the 25 detector scores.
- **Label-free:** fixed LOF k=10, MetaOD (leave-source-out meta-training), MetaEns if code is available, Goswami, EM/MV.
- **Hedge generality:** apply the same rank-average hedge to every strong learner (XGBOD, DevNet, TabPFN, GBDT). This tests whether the hedge is generic or LR-specific.

**Label budgets:** k ∈ {1, 2, 3, 5, 10, 20, 50}, plus ADBench-style 1% and 5% for continuity. Use 30 paired draws per task, with identical draws across methods.

**Prevalence and contamination**
- Test anomaly fraction in {0.5, 0.1, 0.02, 0.005}.
- Unlabeled training contamination (anomalies hidden in the "normal" train set) in {0, 1%, 5%}. The current clean-train assumption is itself a reviewer target.

**Metrics:** ap_norm (primary), ROC-AUC (for ADBench comparability), recall at 1% FPR and 5% FPR, and tail regret (worst decile and CVaR of per-task regret versus the default).

**Unseen-type protocol, as a dose-response rather than one binary split**
1. **Leave-class-out.** For OvrBench, where anomalies are actual source classes, and for OddBench where category metadata exists: labels come from anomaly classes C_L, tests from disjoint classes. This is semantic and non-arbitrary. Report how many tasks have at least 2 anomaly classes.
2. **Shift dose-response.** For every label/test anomaly partition, compute a shift magnitude (classifier two-sample AUC or MMD between labeled-pool and test anomalies). Plot gain over default against shift. This answers "where does LR cross below the default and does HYB ever?" and replaces a k-means anecdote. Use multiple clusterings and cluster counts as a robustness check.
3. **Incumbent-missed split.** Labels come from anomalies LOF ranks in its top decile, and the test set is anomalies LOF ranks in the bottom half.
4. **Temporal or block split** for time series.

**Statistics**
- The unit of analysis is the task. Resample by source dataset, since OvrBench derives several tasks from one dataset.
- Use a cluster bootstrap for 95% CIs on paired ap_norm differences. Holm-correct the 6 pre-specified primary contrasts.
- Fit a mixed-effects model with random dataset, task, and draw effects as a check.
- Report absolute Δap_norm. The gap-closed ratio is unstable when per-task denominators are small, so use it only as a pooled summary with a bootstrap CI.
- Freeze code and hyperparameters and hash them before touching the held-out pool.

**Numeric falsification criteria (held-out tabular pool, unless stated)**

| Claim | Falsified if |
|---|---|
| T1: train > select | At k=3 and k=5, LR − SEL ≤ 0.02 ap_norm, or the cluster-bootstrap CI includes 0. |
| T2: label-efficiency | HYB at k=5 closes <50% of the fixed-to-oracle gap (point estimate) or the lower CI bound is <30%. |
| T3: hedge under shift | In leave-class-out or incumbent-missed, HYB − LOF default < −0.01 (non-inferiority breached), or HYB − LR < +0.03 in the shifted conditions. |
| T4: not tied to LR | A published method beats HYB by >0.03 ap_norm (CI excluding 0) at k=5. The headline then becomes "train, don't select, and hedge" with that learner. If hedging that learner also fails to help under shift, T3's generality is gone. |
| T5: realistic regimes | At 2% prevalence with 5% unlabeled contamination, HYB − default < +0.02. |
| Replication | On unhardened ADBench originals, the sign of T1 or T2 flips, or the effect size is <50% of the ADReal effect. |

If T1 or T2 fails on the held-out pool, the thesis is dead and the exploratory result was a selection artifact.

## (d) Most likely sinking objection

> "Supervised beats unsupervised given labels is the oldest result in the field. Your headline comes from a benchmark you built with filters tuned against unsupervised rules, from i.i.d. label draws out of the test anomaly pool."

Of the available attacks, iid label draws are the one with a real-deployment-realism hole. Real labels are the anomalies that were reported or caught, which are biased toward what the incumbent already detects.

**Pre-empting experiment: "Label-acquisition realism on external data."**
- Run on both the held-out pool and unhardened ADBench originals.
- Simulate three acquisition mechanisms:
  - **(i)** iid draws.
  - **(ii)** Incumbent-caught: labels are the anomalies among the default LOF's top-m alerts, for m ∈ {20, 50, 100}. k is random, and if no anomaly is flagged then k=0 and HYB reduces to the default.
  - **(iii)** Analyst triage over the top-m alerts from a mixed random plus top-score queue.
- **Pass:** under (ii), HYB beats the default by ≥ +0.02 ap_norm with CI above 0 on both datasets, and HYB is within 0.01 of the best method in the baseline set.
- **Predicted outcome:** the gain shrinks under (ii), because LR sees mostly LOF-detectable anomalies.
  - Report that honestly as the main limitation.
  - It also motivates the remedy: query beyond the top alerts. Do not hide it.

This one experiment answers the benchmark-circularity objection (via ADBench originals), the realism objection (via biased acquisition), and the novelty objection (via the baseline set).

## (e) Venue and honest odds

| Venue | Odds if (c) and (d) are executed and the effects hold | Comment |
|---|---|---|
| **TMLR** | ~55–65% | Best fit. It judges correctness and evidence, not novelty hype, and a clean empirical design plus a baseline sweep is what it wants. |
| **DMKD** | ~45–55% | Good fit for a long empirical study. Slower. |
| NeurIPS D&B | ~15–20% | The paper is no longer a dataset contribution. ADReal competes with MacrOData, TSB-AD, and Röchner et al. Only viable if "ADReal-FewLabel" is a strong released protocol. |
| KDD research | ~10–15% | "Linear model plus rank-average" is not a method. The applied-data-science track is better, at about 20%. |
| AAAI | ~15% | |
| ICLR | <8% | |

**Recommendation:** target TMLR, with DMKD as the fallback.

Conditioning matters here. These odds assume the held-out replication confirms the effect. If T1/T2 fail on held-out data, odds go to roughly 0 for this thesis. My prior is about 70% that T1 and T2 survive (the exploratory margin is large, 107% against a 50% bar) and about 40% that T3's non-inferiority holds under the incumbent-missed split. A prior of 40% on a key claim means the paper may have to be written as "gains shrink under realistic label bias." That is still publishable at TMLR but not at the top venues.