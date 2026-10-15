# Prior-art scout, pivot cycle 1 (2026-10-15, web-researcher; citations verified by page visit unless noted)

Saturated: label-free UOMS (Ma et al. SIGKDD Explor. 2023; Goswami et al. ICLR 2023; ELECT; HYPER KDD 2024;
Idan ECAI 2024; MetaEns ICML 2026 arXiv 2605.16567) and benchmark-quality critiques (Wu & Keogh TKDE 2023;
TSB-AD NeurIPS 2024 D&B; ADBench 2022; Rochner et al. NeurIPS 2026 position arXiv 2507.15584; MacrOData
arXiv 2602.09329; Kluttermann et al. arXiv 2608.04613 ranking instability over 690 OddBench datasets;
Deng arXiv 2609.29580 train/test overlap in 355/690 OddBench datasets).

THREAT to current headline: MetaEns (ICML 2026) beats a "Global Best" fixed model, AP 0.4308 vs 0.3787, on 39
ADBench/ODDS datasets.

Open (least saturated):
1. Few-label selection: how many labeled anomalies (1-10) make label-based selection reliably beat the default;
   active querying to discriminate candidate detectors. Closest: Perini et al. arXiv 2301.02909 (label budget:
   active learning vs learning to reject); Li et al. arXiv 2302.07832; semi-supervised CASH arXiv 2211.13681.
2. Abstain-or-deviate selection with the default as fallback / predicting when selection pays off. Closest:
   Kashgarani & Kotthoff PMLR v140 2021 (SAT); Perini & Davis NeurIPS 2023 (per-prediction rejection).
3. Difficulty-stratified benchmark (marginally-detectable vs only-jointly-detectable anomalies) tied to where
   selection/detectors win. Builds on Rochner et al. and MacrOData SynBench archetypes.
