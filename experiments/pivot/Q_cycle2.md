Cycle 2. The attached state brief (section 4, "Cycle 1" block) now holds the VALIDATED triage outcomes of
your three pivots. Summary:

1. Portfolio pivot: validated dead (every LOF+specialist combination is worse than LOF alone; not a tie
   artifact).
2. Few-label SELECTION works on tabular data (55% of the fixed-to-oracle gap at k=5) but is DOMINATED by
   using the same labels to TRAIN a plain class-balanced logistic regression (normal training data vs the k
   anomalies): it overtakes selection from k=2 and beats the per-task oracle of the 25 unsupervised
   detectors at k=5 (gap closed 107%; OddBench 138%). Negative control with mislabeled normals fails as it
   should. Holds at 2% prevalence (130%).
3. Pure few-label supervision FAILS on unseen anomaly types (labels from one anomaly cluster, test on the
   others: -25%, worse than the default), while a pre-specified hedge, the rank-average of the logistic
   regression and the unsupervised default LOF k=10, is best in every setting (seen 88%, unseen +22%,
   2% prevalence 144%). Time-series tasks each contain one anomaly event and the default is near-optimal
   there; selection hurts.

Questions (be decisive and concrete; no browsing; use the attached files):

(a) NOVELTY. Against ADBench's semi-supervised findings (1% labeled anomalies beat the best unsupervised
    method), OneShot (arXiv 2610.16219, k=1-8 labels to weight detectors, time series), and open-set
    supervised AD (DRA, CVPR 2022), what is the sharpest claim this evidence supports that would be novel
    and defensible? Which of our facts are already known?
(b) FRAMING. Propose the paper: title, the one-sentence thesis, 3 contributions, and how the existing
    ADReal benchmark and the label-free-selection negative result fit (or should be cut).
(c) CONFIRMATORY DESIGN. The larger experiment that would make this airtight: which baselines reviewers at
    the target venue will require (name them: e.g. DevNet, PReNet, DRA, XGBOD, FEAWAD, RoSAS, TabPFN-style
    few-shot classifiers, label-free MetaOD/MetaEns-style selectors), label budgets, the unseen-type
    protocol (how to define anomaly types defensibly beyond k-means clusters), prevalence settings,
    statistics, and the exact numeric outcomes that would falsify the thesis.
(d) The single most likely reviewer objection that could sink it, and the experiment that pre-empts it.
(e) VENUE: best target (TMLR / NeurIPS D&B / KDD / ICLR / AAAI / DMKD) and honest acceptance odds.

Return the full answer as Markdown.
