# Appendix A. Curation ledger

Every dataset available in the source collections is recorded with the stage at which it left or entered the benchmark, so the path from 1,192 available datasets to the 151 in ADReal is fully auditable. Table A1 gives the per-source funnel across the three phases of construction; Table A2 lists the exact reason each excluded dataset was removed. The complete per-dataset ledger, 1,192 rows giving the source, dataset name, exclusion stage, and benchmark membership of every dataset, is released with the code as `curation_ledger.csv`.

**Table A1.** Per-source curation funnel. Each dataset passes an upstream curation stage, the detector-free construction gates, and three post-scoring floors.

| Source | Modality | Available | Removed: curation | Removed: construction | Removed: floors | In benchmark |
|---|---|---:|---:|---:|---:|---:|
| OvrBench | tabular | 754 | 618 | 60 | 10 | 66 |
| OddBench | tabular | 187 | 80 | 61 | 7 | 39 |
| TSB-AD | time series | 200 | 0 | 159 | 3 | 38 |
| ADBench | tabular | 35 | 0 | 28 | 2 | 5 |
| DAMI | tabular | 16 | 0 | 12 | 1 | 3 |
| **Total** | | **1,192** | **698** | **320** | **23** | **151** |

**Upstream curation.** The two large tabular collections pass a pre-filter before entering the construction pipeline. It removes datasets with too few normal points to form a held-out modeling set, fewer than 800 after the split (159 OvrBench, 35 OddBench), and datasets whose anomalies are separable by a single feature or a permutation test (459 OvrBench, 45 OddBench). ADBench, DAMI, and TSB-AD enter the construction pipeline directly from their raw sources, so their reductions are recorded entirely under the construction gates.

**Construction gates.** The detector-free construction pipeline removes, per dataset: point or short time-series whose windowing yields only diluted labels (7), datasets whose anomalies are at least 90% separable by the single-feature rule (52), datasets left with fewer than 800 held-out normals (158) or fewer than 100 distinct hard anomalies after hardening (79), and datasets failing the time-series length or basic size gates (24). These leave 174 candidate tasks.

**Post-scoring floors.** Three floors computed after scoring remove 23 further tasks. The separability floor drops 17 tasks whose oracle-best detector does not clear a base-rate-normalized average precision of 0.05; the minimum-anomaly floor drops 5 tasks with fewer than 100 test anomalies; and the selection-trivial floor removes the remaining task on which every detector performs within a negligible margin, so selection cannot matter. 151 tasks remain.

**Table A2.** Exclusion reasons across all 1,192 available datasets.

| Phase | Reason | Datasets |
|---|---|---:|
| Upstream curation | Single-feature or permutation triviality, usability pre-filter | 504 |
| Upstream curation | Fewer than 800 normal points after split | 194 |
| Construction gate | Fewer than 800 held-out normal points | 158 |
| Construction gate | Fewer than 100 distinct hard anomalies | 79 |
| Construction gate | At least 90% single-feature separable | 52 |
| Construction gate | Time-series length or basic size gate | 24 |
| Construction gate | Point or short time-series | 7 |
| Post-scoring floor | Separability: oracle ap_norm at most 0.05 | 17 |
| Post-scoring floor | Fewer than 100 test anomalies | 5 |
| Post-scoring floor | Selection-trivial | 1 |
| In benchmark | Retained | 151 |

The construction gates and the post-scoring floors are recorded exactly, per dataset, and reproduce the funnel of Figure 1 and Table 1. For the upstream curation of the two tabular collections, the size component is exact per dataset and the triviality and usability component is recorded as a class; both are reproduced in `curation_ledger.csv`.
