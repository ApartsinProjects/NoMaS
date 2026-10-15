# Prior-art scout, pivot cycle 2 (2026-10-16): few-label / hedged / open-set tabular AD

Closest and biggest overlap risk: WSADBench (Yao, ..., Han, Jiang; arXiv 2605.26068; KDD 2026 D&B):
absolute label counts 1/5/10, Known vs Unknown anomaly classes (split by PCA distance from the normal
centroid), XGBoost/CatBoost/TabM/TabR/TabPFN vs "Median Unsup."; finds specialized weakly supervised methods
win only under extreme scarcity and general classifiers / tabular foundation models win OOD. It has NO
per-dataset unsupervised oracle, NO unsupervised+supervised hedge, apparently no logistic regression.

Known: few labels beat unsupervised detectors (ADBench, NeurIPS 2022 D&B; label ratios 1-100%, no absolute
counts, no per-dataset oracle; "supervised needs ~10%"). Label-trained detectors miss unseen types (PReNet,
KDD 2023, 60 labels, UNSW-NB15 held-out attack classes; DRA, CVPR 2022, image data). Unsupervised prior +
supervised correction (XGBOD, IJCNN 2018; SSDO, ICDM 2018). Deep weakly supervised tabular methods (FEAWAD,
TNNLS 2021; RoSAS, IP&M 2023, 30 labels). OneShot (arXiv 2610.16219, Oct 2026): k=1-8 labels weight a TS
detector pool, gap closed 82.8%, no training baseline.

Novel and defensible: (1) with ABSOLUTE k=2-5, plain class-balanced LR beats the PER-DATASET UNSUPERVISED
ORACLE, an upper bound on ANY label-free selector -> "a few labels are worth more than perfect model
selection", linking to ADReal's no-selector-beats-default result; (2) a parameter-free rank-average hedge
with the strong unsupervised default is never worse than either component, incl. held-out types and 2%
prevalence. Must beat WSADBench-style baselines (XGBoost, TabPFN, DevNet/PReNet) at the same k, and read
WSADBench Appendix E.1 (1/5/10-count results) + OOD tables.
