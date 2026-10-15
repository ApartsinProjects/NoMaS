# -*- coding: utf-8 -*-
import io
p = r'E:\Projects\Submitted\ADRank\paper\adrank.md'
s = io.open(p, encoding='utf-8').read()

E = [
# 1 abstract pipeline
("For each detector, we cluster the data, hide a subset of clusters, fit the detector on the complement, and score the held-out cluster against a held-out normal fold.",
 "We cluster the data, hide a subset of clusters, fit each detector on the complement, and score the held-out clusters against a held-out normal fold."),
# 2 abstract define regret first
("The ranking signal comes from the geometry of normal data alone: in a contamination study the selection regret is flat as real anomalies are added to the unlabeled input and is unchanged at 0% contamination (pure normal data), so ADRank needs no anomalies in its input to rank the detectors that find them. We measure a detector selector by <em>regret</em>: the true-AUC gap between the best detector in the panel and the one the selector picks, a threshold-free, decision-relevant quantity that stays well-behaved when detectors tie.",
 "We measure a detector selector by <em>regret</em>: the true-AUC gap between the best detector in the panel and the one the selector picks, a threshold-free, decision-relevant quantity that stays well-behaved when detectors tie. The ranking signal comes from the geometry of normal data alone: in a contamination study regret is flat as real anomalies are added to the unlabeled input and unchanged at 0% contamination (pure normal data), so ADRank needs no anomalies in its input to rank the detectors that find them."),
# 3 abstract we run
("leaves on text: run a bank of pseudo-anomaly regimes and weight each by how strongly it separates the detectors, a purely label-free signal.",
 "leaves on text: we run a bank of pseudo-anomaly regimes and weight each by how strongly it separates the detectors, a purely label-free signal."),
# 4 abstract oracle sentence
("and 90% on windowed time-series, matching within 0.01 of AUC, on these modalities, an oracle that picks the best regime using labels.",
 "and 90% on windowed time-series. On these modalities it matches, within 0.01 AUC, an oracle that picks the best regime using labels."),
# 5 abstract smaller set
("69 anomaly-detection tasks drawn from a smaller number of source corpora",
 "69 anomaly-detection tasks drawn from a smaller set of source corpora"),
# 6 sec1 list
("Isolation Forest wins on some, LOF on others, ECOD on a third.",
 "Isolation Forest wins on some datasets, LOF on others, ECOD on yet others."),
# 7 sec1 contribution2 repeat
("across four modalities (tabular, image, text, time-series) with 9 canonical PyOD detectors and five seeds. ADRank cuts selection regret vs random by 73\u201390% across all four modalities (tabular, image, text, time-series).",
 "across four modalities (tabular, image, text, time-series) with 9 canonical PyOD detectors and five seeds. ADRank cuts selection regret versus random by 73\u201390% on all four."),
# 8 sec1 contribution3 object
("and cuts the two embedding modalities a further 2\u20134\u00d7 (text from 0.022 to 0.006)",
 "and cuts regret on the two embedding modalities a further 2\u20134\u00d7 (text from 0.022 to 0.006)"),
# 10 sec4 skill comma
("To summarize a benchmark in one number we pool, $S = 1 - \\sum_i \\mathrm{regret}_i / \\sum_i \\mathrm{regret}_{\\text{random},i}$: a Murphy-style skill score",
 "To summarize a benchmark in one number we pool across datasets: $S = 1 - \\sum_i \\mathrm{regret}_i / \\sum_i \\mathrm{regret}_{\\text{random},i}$, a Murphy-style skill score"),
# 12 sec5.5 CIFAR
("CIFAR10 class 1, for instance, moves from",
 "CIFAR-10 class 1, for instance, moves from"),
# 13 sec5.6 to->by
("Auto-calibration cuts regret to 78\u201393% below random on all four modalities, and improves",
 "Auto-calibration cuts regret by 78\u201393% relative to random on all four modalities, and improves"),
# 14 sec5.7 top-3
("Reporting the top-3 handoff halves the weak spot with no method change: regret@3 = 0.020 \u00b1 0.004.",
 "Handing the user ADRank's top three picks instead of one halves this gap with no method change: regret@3 = 0.020 \u00b1 0.004."),
# 15 sec5.7 cells
("ADRank picks a local-density detector (LOF/KNN/CBLOF) on 98% of DAMI cells, but the true best is local on only 48%; when the true best is a global detector (HBOS/COPOD/ECOD/PCA), no cluster-based regime in the bank can make it win, and the oracle-in-bank regret on those cells (0.039) is 5\u00d7 that on local-best cells (0.008).",
 "ADRank picks a local-density detector (LOF/KNN/CBLOF) on 98% of DAMI dataset-seed cases, but the true best is local on only 48%; when the true best is a global detector (HBOS/COPOD/ECOD/PCA), no cluster-based regime in the bank can make it win, and the best-achievable-in-bank regret on those cases (0.039) is 5\u00d7 that on local-best cases (0.008)."),
# 16 sec5.7 instructive
("is more subtle and instructive. Off-the-shelf VaDE",
 "is more subtle. Off-the-shelf VaDE"),
# 17 sec5.7 para break VaDE
("cannot distinguish a high-variance-but-misaligned regime from an informative one. A second candidate fix, replacing the PCA-then-MiniBatchKMeans",
 "cannot distinguish a high-variance-but-misaligned regime from an informative one.\n\nA second candidate fix, replacing the PCA-then-MiniBatchKMeans"),
# 18 sec5.7 para break marginal
("the collapse is a reconstruction-weighting artifact and its onset is observable without labels via effK. The remaining ceiling is partly closed",
 "the collapse is a reconstruction-weighting artifact and its onset is observable without labels via effK.\n\nThe remaining ceiling is partly closed"),
# 19 sec5.8 freeze parenthetical
("We then freeze the method entirely, the smallest+random+hard regime bank, K in {30, 50}, discriminative weighting, no tuning, and run it on 260 OddBench [22] datasets",
 "We then freeze the method entirely (the smallest+random+hard regime bank, K in {30, 50}, discriminative weighting, no tuning) and run it on 260 OddBench [22] datasets"),
# 20 sec5.8 honest ceiling
("rather than beating it, which is the honest ceiling of a selector when one detector happens to dominate a corpus.",
 "rather than beating it; matching the dominant detector is the ceiling of any selector when a single detector dominates a corpus."),
# 21 sec6 bounded honestly
("The strength is bounded honestly: on the DAMI benchmark",
 "The strength has clear bounds: on the DAMI benchmark"),
# 23 sec8 opening
("Clusters of the unlabeled training distribution behave, for the purpose of ranking anomaly detectors, close enough to real anomalies.",
 "For the purpose of ranking anomaly detectors, held-out clusters of the unlabeled data behave enough like real anomalies to stand in for them."),
# 24 sec8 to->by
("cuts detector-selection regret to 78\u201393% below random on tabular, image, text, and time-series data",
 "cuts detector-selection regret by 78\u201393% relative to random on tabular, image, text, and time-series data"),
# 22a sec7 break
("partially closing the ceiling. Replacing PCA-plus-k-means",
 "partially closing the ceiling.\n\nReplacing PCA-plus-k-means"),
# 22b sec7 break
("is the main open problem. The method requires an expressive fixed-length representation",
 "is the main open problem.\n\nThe method requires an expressive fixed-length representation"),
# flag26 fix logic error
("The local-density detectors LOF and KNN stay positive everywhere too, but at a lower floor (0.21 and 0.23), and on the embedding modalities only because ADRank itself selects LOF there; no fixed detector combines an above-random skill on every benchmark with the top skill on the two where selection is hardest, tabular and OddBench, which ADRank alone does.",
 "The local-density detectors LOF and KNN stay positive everywhere too, but at a lower floor (0.21 and 0.23), and neither takes the top skill on tabular or OddBench, the two benchmarks where selection is hardest, which ADRank alone does."),
# flag25 harmonize text random in Table 7
("<tr><td>Text (13)</td><td><b>0.006</b></td><td>0.087 (p = 5e-4)</td><td>0.006</td><td>0.081</td></tr>",
 "<tr><td>Text (13)</td><td><b>0.006</b></td><td>0.087 (p = 5e-4)</td><td>0.006</td><td>0.082</td></tr>"),
]

fail = []
for old, new in E:
    c = s.count(old)
    if c != 1:
        fail.append((c, old[:70]))
        continue
    s = s.replace(old, new)
io.open(p, 'w', encoding='utf-8').write(s)
print("applied", len(E) - len(fail), "of", len(E))
for c, o in fail:
    print(f"  FAILED (count={c}): {o}")
