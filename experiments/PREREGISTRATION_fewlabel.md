# Pre-registration: "Spend the labels on training, not selection" (frozen 2026-10-16)

Frozen BEFORE any result on the held-out pool exists. Committed to git; the commit hash is the timestamp.
Any later change is recorded below under "Amendments" with its reason, and results are reported under both
the original and the amended rule.

## Development vs confirmation data

- DEVELOPMENT (exploratory, already inspected): the 112 ADReal tabular tasks (artifacts/modal/bv2, keep
  set). All numbers there are exploratory.
- CONFIRMATION (held-out): tabular tasks built with the SAME detector-free 5-stage pipeline
  (experiments/2026-09-04_streamlined-benchmark-construction.md: OR-solvable drop, marginal + whitened
  joint hardening at 5% normal FP, >= 100 distinct hard anomalies, fingerprint dedup, >= 800 normals),
  grouped leak-free split, anomalies identical to a normal removed, from OvrBench / OddBench source datasets
  NOT among the development candidates, and NOT sharing an original source family with any development
  task (family = MacrOData tablib_origin / github_link, else normalized name stem). NO detector-dependent
  floors. Built and frozen before any method is evaluated on it.
- EXTERNAL UNHARDENED replication: ADBench / DAMI originals (no hardening; grouped split; anomalies
  identical to a normal removed).
- SEMANTIC TYPES: ADBench CV/NLP embedding families with recovered anomaly classes (CIFAR10,
  FashionMNIST, 20news, agnews; recovery by exact embedding match, experiments/pivot/recover_classes.py).
- Time series: reported as a separate boundary stratum only, not in any headline claim.

## Frozen protocol

- Splits: per task, test anomalies split 50/50 into a LABELING POOL and an EVALUATION set; test normals
  split 50/50 likewise. The evaluation set is identical for every method, budget and acquisition mode.
- Label budgets k in {1, 2, 3, 5, 10, 20, 50}; nested draws (the first k of a draw are the labels at k).
  Primary k = 5. 30 draws for cheap methods, 10 for deep methods.
- Acquisition modes: IID; CAUGHT (pool anomalies in the fixed default's top-m alerts, m in {20, 50});
  MISSED (from the lower half of the default's ranking of pool anomalies).
- Evaluation prevalence: native, 10%, 2% (primary), 0.5% (anomaly subsampling of the evaluation set, 10
  subsamples). Training contamination: 0%, 1%, 5% of evaluation-disjoint anomalies hidden in the training
  normals.
- Fixed default D: LOF with k=10 (frozen constant; chosen on development data).
- Oracle O: per-task best of the 25 classical unsupervised detectors on the evaluation set (reference only).
- Hedge H(learner): mean of the learner's and D's empirical CDFs, each estimated on the validation normals
  (inductive; no test-batch information). Applied to every learner.
- Methods: D; label-free selectors (EM, MV, consensus, MetaOD with leave-source-family-out meta-training);
  label-based selection SEL; supervised LR (class-balanced), RF, gradient-boosted trees (XGBoost); XGBOD;
  weakly supervised DevNet, PReNet, DeepSAD, FEAWAD, RoSAS; open-set DRA (tabular adaptation documented);
  TabPFN (pretrained, disclosed); and H(.) of each learner. Fixed hyperparameters (library defaults or
  set on development data); no per-task tuning.
- Metric: ap_norm (primary), AP at the evaluation prevalence, ROC-AUC, recall at 1% / 5% FPR.
  Gap closed G(M) = mean(A_M - A_D) / mean(A_O - A_D) (ratio of means, never mean of ratios).
- Statistics: per task average draws; inference unit = original source family; paired cluster bootstrap
  (10,000 resamples) for CIs of paired differences; Holm correction over the 6 primary contrasts below.

## Primary claims and falsification rules (confirmation pool, k = 5, 2% prevalence unless stated)

T1 TRAIN > SELECT: A(LR) - A(SEL) >= 0.02 ap_norm with cluster-bootstrap 95% CI above 0, at k=3 and k=5.
   FALSIFIED if the difference <= 0.02 or the CI includes 0 at either budget.
T2 LABEL EFFICIENCY: H(LR) closes >= 50% of the D-to-O gap at k=5 (point), lower 95% CI >= 30%.
   FALSIFIED otherwise.
T3 HEDGE UNDER SHIFT: under semantic leave-class-out AND CAUGHT acquisition, A(H(LR)) - A(D) >= -0.01
   (non-inferior) and A(H(LR)) - A(LR) >= +0.03 in the unseen-class condition.
   FALSIFIED if either fails.
T4 NOT LR-SPECIFIC: if any published learner beats H(LR) by > 0.03 ap_norm (CI excluding 0), the headline
   becomes "train, don't select, and hedge" with that learner; T3 is then re-tested with that learner's
   hedge and must hold.
T5 REALISTIC REGIME: at 2% prevalence with 5% training contamination, A(H(LR)) - A(D) >= +0.02 (CI > 0).
   FALSIFIED otherwise.
R  REPLICATION: on unhardened ADBench/DAMI, the signs of T1 and T2 hold and their effect sizes are >= 50%
   of the confirmation-pool effect. FAILURE means the effect is tied to the hardened regime and is
   reported as such.

Stop rule (strategist): if DRA, RoSAS or another specialist dominates H(LR) on both seen and unseen
classes by > 0.02 with CI excluding 0, no new combination is invented to rescue the hedge; the paper is
reframed as a label-allocation benchmark with that specialist winning.

## Amendments

A1 (2026-10-16, before any held-out result): DRA dropped. It has no tabular implementation (the official
code and deepod 0.4.1 do not provide one), and a home-made adaptation of an image method would not be a
meaningful baseline. The open-set role is taken by PReNet (KDD 2023; designed for unseen-anomaly
generalization on tabular data) and RoSAS (IP&M 2023), both from deepod 0.4.1. The T3/stop-rule references
to DRA apply to PReNet and RoSAS instead.

A2 (2026-10-16, before any held-out result): baselines run in an isolated environment (.venv-baselines:
Python 3.11, scikit-learn 1.7.2, numpy 2.4, torch 2.14, xgboost 3.2, tabpfn 9.2, deepod 0.4.1 installed
without its stale torch<1.13 pin after verifying all five models train). The 25 unsupervised detector
scores keep the main environment (scikit-learn 1.8, pyod 3.6.1) used for development. "MetaOD" is a
MetaOD-style meta-learner (dataset meta-features -> per-detector expected score, trained on development
tasks, leave-source-family-out), because the released metaod package selects from its own fixed pool, not
our 25 detectors. "XGBOD" is XGBoost on raw features plus the standardized scores of the 25 detectors
(the XGBOD feature-augmentation idea on our pool).
