:::writing{variant="document" title="ADReal Final Clarity and Coherence Review" id="74183"}
# ADReal — Final Clarity and Coherence Review

The revision is substantially more coherent than the preceding version. The principal numerical results are now consistent across the main tables, the full-32 oracle is explicitly defined, the UOMS additions are present, and the new uncertainty and win/tie/loss reporting improves the central comparison. The remaining work is primarily a final reconciliation and structural edit. There is one clear stale numerical value, several claims that are stronger than the evidence or contradict their own qualifications, and substantial repetition of the same results across Sections 5–9. The most important structural improvement is to make Section 5 answer the selection question before using the three-pool analysis to explain its scope, and then let Sections 6 and 7 explain the method and its limitations rather than repeat the leaderboard.

## A. Stale or self-contradictory content

### 1. The time-series SPARC regret is inconsistent with Table 6b

**Location:** Section 6, “Selection helps only where the oracle detector varies,” HTML line 629.

**Current text:** “0.037 versus 0.038, p=0.56” for time series. Table 6b reports SPARC 0.038 and fixed 0.038. The user-supplied current value is also 0.038. This is the clearest stale numerical value in the revision.  

**Fix:** Change the prose to “0.038 versus 0.038, p=0.56,” assuming Table 6b contains the authoritative rounded result. Check the underlying unrounded values before changing Figure 7 itself.

### 2. Figure 7 still contains the old “ties” interpretation

**Location:** Figure 7 image alternative text, HTML line 631.

**Current text:** “SPARC beats the fixed detector on OddBench and ties it elsewhere.”

The visible caption has already been improved to say that SPARC “does not significantly differ” on time series and OvrBench, with p=0.56 and p=0.42. The alternative text still uses the earlier, stronger interpretation. 

**Fix:** Replace “ties it elsewhere” with “shows no statistically significant difference on time series or OvrBench.” Check the actual figure image for any embedded “tie” labels as well.

### 3. The full-pool limitation is still missing from the abstract and conclusion

**Locations:** Abstract, HTML line 101; Introduction contribution 4, line 114; Section 6, line 628; Conclusion, line 680.

The repeated claim that SPARC is “the only label-free method that significantly beats” the fixed detector is true only for the reported classical-25 comparison. Section 5 explicitly says that no label-free method beats the fixed detector on the full-32 pool and that SPARC does not significantly differ from Goswami on either the classical or full pool. The abstract and conclusion omit this scope, making the paper sound more universally positive than its central results.    

**Fix:** State “among the evaluated methods, on the classical-25 pool” at the first substantive claim. The abstract should also briefly acknowledge that the advantage disappears when deep detectors are eligible. Keep “only” as a statement about which method reaches nominal significance against the fixed reference, not as proof that SPARC is statistically superior to every alternative.

### 4. “Matching” the fixed detector remains an equivalence overstatement

**Locations:** Introduction, line 108; Section 5, line 451; Section 6, line 628.

**Current text:** “a recent injection-based selector [35] only matches it” and “Goswami's injection selector reaches 0.122, matching the fixed detector but not beating it (p=0.88).”

A nonsignificant test does not establish equivalence. The new Section 5 reporting is otherwise careful about SPARC versus Goswami, so this older wording stands out.   

**Fix:** Use “does not significantly improve on the fixed detector” or “has no detected advantage over the fixed detector.” Do not replace it with “is equivalent” unless equivalence is actually tested.

### 5. The claim of “no tuning” conflicts with the discussion of design choices

**Locations:** Abstract, line 101; Introduction, line 114; Section 6, lines 603–604; Conclusion, line 680.

The revised Section 6 now says that the fraction 0.4 and 1000-point budget were set a priori and not selected on ADReal. That is a useful clarification. However, the same section says that the classical-candidate restriction “earns the rest” and explains that deep detectors are excluded because of their observed proxy mismatch. The broad phrase “no tuning” obscures the distinction between fixed deployment hyperparameters and research-stage choices about the method and candidate pool.  

**Fix:** Use “no per-dataset label-based tuning” consistently. Retain the a-priori statement for the fraction and budget, but distinguish the classical-candidate restriction as a deliberate design choice supported by the pool analysis.

### 6. Section 6 contradicts itself about whether deep detectors belong to SPARC's candidate pool

**Location:** Section 6, “Candidates,” HTML line 603.

**Current text:** “SPARC selects among the classical detectors while the pool and the oracle still include the deep detectors.”

The distinction is understandable after reading Table 6, but the sentence uses “pool” in two incompatible senses. SPARC's default candidate pool is classical; the full evaluation pool and common oracle include deep detectors. 

**Fix:** “SPARC's default configuration selects from the 25 classical detectors. Its regret is nevertheless measured against the oracle over all 32 detectors, including the deep models. We also evaluate a full-pool variant in Section 5.”

### 7. The Table 8 caption has not caught up with the revised interpretation

**Location:** Table 8 caption, HTML line 639.

**Current text:** “The probe over-ranks the deep family, does not see the distribution family, and is matched to the neighbor family.”

The paragraph immediately above now correctly says that neighbor-family accuracy is modest—39 correct family selections out of 78 neighbor predictions, recovering 39 of 68 neighbor-oracle tasks—and concludes that Table 8 shows a preference rather than reliable per-task identification. “Matched to the neighbor family” is therefore a stale, stronger description. “Does not see” is also too absolute when the probe ranks a distribution detector first once. 

**Fix:** “The probe strongly favors deep and neighbor detectors, rarely ranks distribution detectors first, and identifies the neighbor-oracle family with modest per-task accuracy.”

### 8. The deep-detector explanation is still more categorical than the evidence

**Locations:** Section 5, line 450; Section 6, lines 603–604; Section 7, line 638.

**Current text:** “The deep detectors are the cause.” Section 7 also says that deep models separate the probe “whatever the model's skill on real anomalies.”

The reported averages—probe ROC-AUC 0.72 versus 0.63 and real-anomaly ap_norm 0.12 versus 0.13—support proxy mismatch, but they do not establish that every deep model behaves this way or that the same mechanism explains every selector's deterioration. The revised Table 8 gives a stronger descriptive basis for this interpretation, but not a causal proof.   

**Fix:** Replace “The deep detectors are the cause” with “The results suggest a proxy mismatch associated with deep-detector eligibility.” Replace “whatever the model's skill” with “even when that model does not perform comparably well on real anomalies.” Use “proxy mismatch” rather than “overfitting” unless an actual training-stage overfitting mechanism is demonstrated.

### 9. The “detector-free pipeline” description is still internally overbroad

**Locations:** Introduction, line 107; Related Work, line 119; Section 3 opening, line 124; Conclusion, line 680.

The introduction and conclusion describe a detector-free pipeline that closes contamination channels, while Section 3 explicitly states that two post-scoring floors are computed from the detector pool. The contribution bullet now makes this distinction correctly, but the surrounding prose still contains the older absolute formulation.  

**Fix:** Refer consistently to “detector-independent construction followed by detector-dependent post-scoring filters.” Avoid “no design choice can be tuned to flatter a detector” and “each channel is closed,” which are stronger than the implemented controls establish.

### 10. Section 3.5 contradicts the manuscript's actual reporting practice

**Location:** Section 3.5, HTML line 277.

**Current text:** “A raw difference of means that a handful of easy tasks could produce is never reported as a result.”

The paper reports mean regret throughout, including the headline 0.021 advantage. The new confidence interval, median, tail, and win/tie/loss reporting are precisely the improvements that make those mean differences interpretable. The sentence is therefore an outdated rhetorical contrast, not an accurate description of the method.  

**Fix:** “We report mean regret together with paired significance tests and distributional summaries, so that average differences can be interpreted alongside their uncertainty and task-level variability.”

### 11. Section 6's opening is false if read literally

**Location:** Section 6, “Motivation,” HTML line 601.

**Current text:** “The surviving label-free criteria score a detector from the shape of its output on normal data alone; none confronts the detector with an anomaly.”

Goswami is already introduced as an injection-based selector in Section 5. The intended contrast is between internal score-distribution criteria and synthetic-probe methods, not between all label-free methods and SPARC.  

**Fix:** “Internal score-distribution criteria evaluate detectors without testing their response to a constructed anomaly. Synthetic-probe methods, including Goswami's, instead introduce a proxy detection task. SPARC follows this second approach with a single observed-value corruption probe.”

### 12. The deep-only fixed reference is not identified clearly

**Location:** Table 6 and Figure 6 caption, HTML lines 454–457.

Table 6 gives the best fixed detector as 0.130 on classical, 0.163 on deep-only, and 0.130 on full. The caption describes the dashed line as LOF(k=10), which cannot be the fixed detector selected from the deep-only pool. The common full oracle explains the classical/full equality, but not the identity of the deep-only reference. 

**Fix:** Identify the best fixed deep detector in the table caption or a note. If Figure 6 uses a different fixed-reference line for each pool, say so. If it uses one LOF line across all groups, label it as the common LOF reference rather than the best fixed detector of each pool.

### 13. “Every method receives the same inputs” overlooks the UDR exception

**Location:** Section 5, “Protocol,” line 449.

The next paragraphs correctly explain that UDR is evaluated on a separate nine-configuration iForest sub-pool. The opening “every method receives the same inputs and is scored the same way” should therefore be qualified. 

**Fix:** “For each applicable candidate pool, methods use the same training and normal-validation splits and are evaluated on the same test set. UDR is evaluated separately on its stochastic iForest sub-pool.”

### 14. “Selection adds value only where the oracle detector varies” is too strong

**Locations:** Section 5, line 536; Section 6, line 629; Section 7, line 636; Section 8, line 678.

Oracle variation creates an opportunity for selection, but it is not sufficient for successful selection. The probe must also identify which detector is suitable from the available normal data. Section 7's new 50% neighbor precision and 57% recall make this qualification particularly important.   

**Fix:** “Selection has the greatest potential value where the best detector varies across tasks, provided the selection criterion can identify those differences.” For the empirical conclusion, say “SPARC's observed advantage is concentrated in OddBench.”

### 15. The paper still treats all retained anomalies as local joint-structure violations

**Locations:** Section 6, line 602; Section 7, lines 636–637.

The probe paragraph says it is sensitive to “exactly the local, joint structure that real anomalies inhabit.” Section 7 says that retained anomalies are marginally typical and “sit in locally sparse joint regions.” The hardening rules remove anomalies flagged by specified marginal and whitened-coordinate tests; they do not prove that every retained anomaly has this mechanism.   

**Fix:** Use “can include,” “is intended to emphasize,” or “is consistent with” rather than defining the entire retained population as local joint-structure anomalies. This also prevents the mechanism discussion from becoming circular.

### 16. The revised numbers otherwise reconcile

I found no additional direct numerical disagreement among the supplied current values and Tables 6, 6b, 7, and 8. The nine applicable three-pool methods are now present; eight worsen from classical to full and IFOREST-R is unchanged. IREOS is correctly identified as the deep-only exception. The 63/46/42 win/tie/loss counts sum to 151, and Table 8's family counts sum to 151 in each column. Its neighbor counts imply precision 39/78 = 50% and recall 39/68 ≈ 57%, consistent with the new prose. The Table 6b caption now explains that the eight ADBench/DAMI tasks are included in the total but not shown separately. The common full-32 oracle also resolves the former apparent contradiction in the fixed-reference values.   

The HTML metadata and footer still retain the older title, “Anomaly Detector Model Selection by Normal Manifold Separability.” These should be synchronized with the displayed title before submission.  

## B. Prose clarity and redundancy

### 1. The abstract's SPARC sentence is overloaded

**Location:** Abstract, line 101.

**Current sentence:** “Finally, we introduce SPARC, a normal-only model-selection method that generates synthetic anomalies from the normal data, by resampling a subset of each point's features from their own observed values so the combination is unusual while every value stays realistic, and ranks candidate detectors by how well they separate them.” 

The sentence combines the method name, data source, corruption mechanism, intuition, and ranking rule. The comma before “by resampling” also interrupts the natural flow.

**Rewrite:** “Finally, we introduce SPARC, a normal-only model selector that generates synthetic anomalies by replacing a subset of each observation's features with values sampled from the corresponding empirical marginals. It then ranks detectors by how well they separate these corrupted observations from normal data.”

### 2. The introduction's benchmark/protocol sentence tries to do too much

**Location:** Introduction, line 107.

**Current sentence:** “ADReal then serves a second purpose: because a detector must in practice be chosen from normal data alone, the benchmark is also a controlled testbed for unsupervised model selection, where a method picks one detector without labels and is scored by its regret against the oracle-best detector, its criterion computed on held-out normals only.” 

The final participial phrase is grammatically awkward and forces the reader to infer which object “its criterion” belongs to.

**Rewrite:** “ADReal also provides a controlled testbed for unsupervised model selection. Each method chooses a detector using only held-out normal data, and its choice is evaluated on labeled test data by regret against the best available detector.”

### 3. The “fixed detector” concept should be defined once in a clean sentence

**Location:** Section 5, line 448.

**Current sentence:** “The bar is the best fixed detector, always deploying the single detector with the lowest average regret across the benchmark (LOF with k=10), which uses no per-task information at all.” 

“Always deploying” is attached awkwardly to “detector,” and “no per-task information at all” obscures the fact that the reference was selected retrospectively using benchmark labels.

**Rewrite:** “Our stronger reference is the best single detector on average across ADReal: LOF(k=10). It is selected retrospectively using benchmark labels but makes no dataset-specific selection decision at deployment.”

### 4. The Section 5 protocol can be divided into three clear operations

**Location:** Section 5, line 449.

**Current sentence:** “A detector is fit on the train normals; the method computes its criterion using only the validation normals and the detectors' scores on them, with no access to any anomaly; it returns one chosen detector; and we score that choice by regret, the gap in test ap_norm between the oracle-best detector and the chosen one.” 

This is a semicolon chain carrying the entire experimental protocol.

**Rewrite:** “Detectors are fitted on the training normals. Each selector then uses the validation normals, and any synthetic data permitted by its method, to choose one detector without access to real anomalies. The chosen detector is evaluated on the held-out test set by its regret relative to the full-32 oracle.”

### 5. The three-pool paragraph combines results, exceptions, mechanism, and conclusion

**Location:** Section 5, line 450.

The paragraph beginning “Adding deep detectors to the candidate pool raises regret for nearly every selector” contains the complete pool comparison, significance claims, exceptions, an explanatory hypothesis, probe statistics, and the conclusion that classical pools are preferable. It is the largest obstacle to the logical flow. 

**Rewrite as three paragraphs:** First state the experimental reason for the pool comparison and the observed changes. Second state the exceptions and distinguish deep-only from full-pool behavior. Third introduce the proxy-mismatch hypothesis, supported by the 0.72/0.63 probe grades and 0.12/0.13 real-anomaly scores, and point forward to Table 8 for the mechanism analysis.

### 6. The phrase “the random floor” is misleading

**Location:** Section 5, lines 448 and 451.

**Current text:** “A uniform-random pick over the pool is the floor” and “sit at or above the random floor.” 

Random selection is a baseline, not a mathematical lower bound on performance. Several methods can be worse than random, and the paper reports exactly that.

**Rewrite:** Use “random-selection reference” throughout. For a specific comparison, say “has higher regret than random selection” or “does not significantly improve on random selection.”

### 7. The new uncertainty reporting is valuable but too compressed

**Location:** Section 5, line 451.

The sentence reporting 0.109, the 0.021 paired advantage, confidence interval, p-value, 63/46/42 counts, and SPARC–Goswami p=0.15 contains several distinct findings in one sentence. 

**Rewrite:** “On the classical pool, SPARC obtains mean regret 0.109, compared with 0.130 for the fixed detector. The paired mean advantage is 0.021 (95% CI [0.006, 0.038]; p=0.03), with 63 wins, 46 ties, and 42 losses across the 151 tasks. Its difference from Goswami is not statistically significant (p=0.15).”

Move the median and 90th-percentile figures into a short separate paragraph or a compact supplementary table, rather than appending them to the already dense result paragraph.

### 8. The probe description should separate the operation from its interpretation

**Location:** Section 6, line 602.

**Current text:** “Each takes a validation normal and resamples a random subset, a fraction of 0.4, of its features, drawing each replacement from that feature's own observed values, so every coordinate stays a value the data actually takes while the combination violates the normal joint structure.” 

“Resamples a random subset, a fraction of 0.4” is awkward, and “so” turns a plausible effect into a guarantee.

**Rewrite:** “Each probe starts from a validation normal and independently replaces 40% of its features with values sampled from their empirical marginals. This preserves observed coordinate values while disrupting some of the dependencies among features.”

### 9. “That is the whole method” is too informal and inaccurate

**Location:** Section 6, line 603.

**Current text:** “That is the whole method: one probe, one grade per detector, an argmax, with a single parameter (the resample fraction, 0.4) and no thresholds to tune.” 

The method also has a fixed budget, candidate-pool restriction, and implementation choices. “That is the whole method” is rhetorical rather than explanatory.

**Rewrite:** “The selection rule is therefore simple: generate one probe set, compute one separation grade per eligible detector, and select the highest-graded detector. The corruption fraction is fixed at 0.4, and no per-dataset label-based tuning is performed.”

### 10. The Table 7 interpretation should avoid “earns the result”

**Location:** Section 6, line 604.

**Current text:** “One design choice is not obvious and earns the result (Table 7): drawing the resampled value from the feature's observed values rather than a continuous interpolation.” 

This is promotional, and the ablation establishes an observed difference rather than the full causal explanation of SPARC's performance.

**Rewrite:** “Table 7 isolates the effect of the corruption operator. Replacing observed-value sampling with bin interpolation increases mean regret from 0.109 to 0.117, suggesting that avoiding unsupported feature values improves the probe's usefulness.”

### 11. The hardening explanation needs shorter, more precise sentences

**Location:** Section 3.2, line 190.

The opening staging analogy is useful, but it occupies several sentences before the actual filter is described. The later sentence beginning “At the opposite end, a task on which no detector beats a random ranking...” combines both post-scoring floors, counts, overlap, and the final task total. 

**Rewrite:** “ADReal is intended to evaluate the residual detection problem after simple rules have removed easily identifiable anomalies. We therefore apply a calibrated marginal and whitened-coordinate filter to remove anomalies flagged by these rules. After detector scoring, we also exclude tasks with very low oracle performance or negligible differences among detectors. The separability floor removes 17 tasks, the selection-trivial floor flags two of those same tasks, and the minimum-anomaly floor removes five further tasks, leaving 151.”

The detailed filter definition should remain immediately after this overview, not be deleted.

### 12. The Section 7 mechanism paragraph should be split by family

**Location:** Section 7, line 638.

The paragraph moves from deep over-ranking to distribution blindness to neighbor precision/recall, family-size confounding, and future probe design. Each is important, but the current paragraph reads like a compressed response to several reviewer comments. 

**Rewrite structurally:** Use one paragraph for the deep and distribution biases, one for the neighbor precision/recall and family-size caveat, and a final sentence for the implication: a complementary marginal probe may address a regime that SPARC currently underserves.

### 13. Redundancy: the same leaderboard is repeated too many times

The result “SPARC 0.109 versus fixed 0.130, p=0.03” appears in the abstract, introduction, contribution list, Section 5, Section 6, limitations, and conclusion. The repetition is not always wrong, but it makes Sections 6–9 feel as though they are repeatedly re-proving the same result.    

**Keep:** The abstract's concise headline, the full statistical result in Section 5, and a one-sentence conclusion. **Cut or shorten:** The complete numerical reprise in Section 6, the repeated numbers in limitations, and the redundant “Why the references matter” paragraph in Section 7. Section 6 should focus on how SPARC works and what the ablation establishes; Section 7 should explain its behavior.

### 14. Redundancy: the OddBench conclusion is stated before and after its explanation

The OddBench result is introduced in Section 5, repeated in Section 6 with Figure 7, repeated at the start of Section 7, and repeated in limitations. The reader receives the conclusion several times before the mechanism analysis has added much new information.   

**Keep:** One full source-level analysis with Table 6b and Figure 7. **Cut:** The short Section 5 source-summary paragraph if the detailed analysis is moved later, or reduce it to one forward reference. Section 7 should connect source behavior to probe bias rather than restating the same numbers.

### 15. Redundancy: the fixed-reference argument is reintroduced rather than developed

The best-fixed-detector reference is explained in the introduction, contribution list, Section 5, Section 7's “Why the references matter,” and the conclusion. The Section 7 paragraph adds almost no new analysis after Section 5.   

**Keep:** A short motivation in the introduction and the formal definition in Section 5. **Delete or replace:** Section 7's “Why the references matter” paragraph. The conclusion can retain the single key lesson that the fixed reference changes the interpretation of internal-metric results.

## C. Logical arc and section organization

### 1. The overall argument is sound, but the paper currently presents its main result before fully defining the method

The intended arc is strong: contaminated benchmarks undermine comparisons; ADReal constructs a harder, controlled evaluation; classical detectors carry much of the performance; normal-only selection must beat a strong fixed reference; most internal methods fail that bar; SPARC achieves a modest classical-pool advantage; the probe's family bias explains both its promise and limitations; and the observed benefit is concentrated in OddBench. The current section order broadly follows this arc, but Section 5 reports SPARC's results before Section 6 has explained the algorithm. Section 5 is also doing three jobs at once: defining the protocol, presenting the main comparison, and analyzing candidate-pool sensitivity.   

**Concrete bridge into Section 5:** At the end of Section 4, add:

> “The detector results establish both the need for selection and the difficulty of the task: no detector is best everywhere, yet a single neighbor detector is already strong on average. The next question is therefore not whether a selector can beat a random choice, but whether it can improve on this fixed reference using only normal data. We evaluate that question under a common full-32 oracle and then examine how the candidate pool changes the answer.”

This makes the fixed-detector bar the natural consequence of Section 4 rather than a new benchmark rule introduced abruptly.

### 2. The three-pool analysis is important, but it interrupts the central comparison too early

Section 5 currently introduces methods and references, defines the protocol, then immediately enters a long three-pool discussion before presenting the main classical-pool results. The reader is asked to process all the deep-pool exceptions and proxy-mismatch claims before seeing the central finding that motivates the paper. The three-pool analysis is not a digression in substance; it is essential to defining the scope of SPARC's result. Its current placement and length, however, make it feel like a detour. 

**Preferred ordering within Section 5:**

1. **5.1 Protocol and references:** Define training, validation, test evaluation, regret, the common full-32 oracle, and the retrospective fixed reference.
2. **5.2 Main classical-pool comparison:** Present the internal and agreement baselines, Goswami, and SPARC's principal result with uncertainty. A one-sentence reminder can refer to SPARC as the synthetic-probe method detailed in Section 6.
3. **5.3 Candidate-pool sensitivity:** Present Table 6's three-pool comparison, explain that the favorable result is restricted to classical eligibility, and report the exceptions.
4. **5.4 Interpretation:** Give only a short proxy-mismatch hypothesis, explicitly reserving the family-level evidence for Section 7.

This ordering lets the reader first understand the question and answer, then learn why the answer is conditional.

### 3. The transition into the three-pool analysis should state the question it answers

The existing transition—“A selector's candidate pool is itself a design choice”—is correct but does not sufficiently explain why the paper suddenly changes from comparing selectors to comparing candidate pools. 

**Suggested bridge:**

> “The preceding comparison holds candidate eligibility fixed, but a practitioner must also decide which detectors to make available. Because the deep models improve the oracle only modestly, we ask whether including them nevertheless improves selection. We therefore repeat the comparison on classical, deep-only, and full candidate pools, while retaining the same full-32 oracle.”

This explicitly connects Section 4's +0.021 oracle gain to Section 5's pool experiment.

### 4. The transition out of the three-pool analysis should lead directly to SPARC's design

At present, Section 5 ends with a source breakdown, and Section 6 begins by reintroducing the general motivation for synthetic anomalies. The reader has just learned that deep eligibility worsens SPARC's regret, but the next section does not immediately explain how that finding shapes the method.  

**Suggested bridge at the end of Section 5:**

> “These results motivate two design choices in SPARC: using a synthetic task that directly tests anomaly separation, and restricting the default candidate set to classical detectors when the probe is poorly aligned with deep-model performance. Section 6 defines the probe and evaluates its construction; Section 7 examines the resulting family preferences and their limitations.”

This makes the pool analysis pay off methodologically rather than remain a separate leaderboard result.

### 5. Section 6 should explain the method, not repeat the conclusion of Section 5

Section 6 currently contains a method description, ablation, a full repetition of the main leaderboard, and a source-level analysis. The method description and Table 7 belong here. The repeated result paragraph and Figure 7 source analysis are better suited to the results/discussion sequence. 

**Recommended role for Section 6:** “What SPARC does and why this construction was chosen.” Keep the probe definition, scoring rule, a-priori parameter statement, candidate restriction, and Table 7. Replace the full “SPARC is the only selector...” paragraph with one short sentence referring back to Section 5. Move the detailed source analysis to Section 7, where it can be interpreted alongside Table 8.

### 6. Section 7 should move from mechanism to source behavior, not the reverse

The current Section 7 begins with “Where selection helps,” then discusses hardening, then Table 8's probe bias. This presents the source-level conclusion before the evidence that is supposed to explain it. The strongest logical sequence is the opposite: first explain what the benchmark selects for, then what the probe sees and misses, then how those properties relate to the observed source-level performance. 

**Recommended Section 7 order:**

**7.1 What hardening selects for.** Briefly connect the construction rules to the retained anomaly regime, with careful language that does not claim every retained anomaly is a local joint-structure violation.

**7.2 What the probe sees and misses.** Present Table 8, the deep and distribution biases, neighbor precision/recall, and the family-size caveat. Conclude that the probe provides a useful but imperfect proxy.

**7.3 Where selection helps.** Present the source breakdown and Figure 7, then interpret the OddBench concentration in light of the preceding mechanism analysis. If Table 6b is moved here, renumber the tables consistently; otherwise retain it in Section 5 and refer back to it without repeating the complete result.

**7.4 Implication.** State that complementary probes may be needed for marginal or other anomaly regimes, leading naturally into limitations.

This ordering turns Section 7 into an explanation rather than a second results section.

### 7. The “oracle diversity” argument is set up but not fully paid off

The paper repeatedly claims that selection helps where the oracle detector varies across datasets and that OddBench has the greatest such variation. However, Table 5 reports oracle-family shares over the entire benchmark, and Table 8 reports aggregate family preferences; neither directly shows source-specific oracle diversity. Table 6b establishes the source-level regret pattern, but not the claimed explanatory variable.   

**Concrete fix:** Either add a compact source-level oracle-diversity statistic—such as the share of tasks won by the best fixed detector, the number of winning algorithms or families, or oracle-family entropy—or soften the explanation. A suitable bridge is: “OddBench shows the largest observed advantage for SPARC, consistent with greater opportunity for per-dataset selection. Establishing whether oracle diversity predicts that advantage requires a separate analysis.”

### 8. The fixed-detector bar should be framed as a strong reference, not as a necessary condition for all selection research to matter

The paper says that a selector “earns its keep only by beating” the best fixed detector. This is an effective rhetorical bar, but it is stronger than the benchmark's actual inference. A selector could still be useful in a different candidate pool, application, or deployment setting even if it does not beat the retrospectively best fixed detector on ADReal. The current limitations acknowledge the finite pool and hardening bias, but the earlier framing remains categorical.  

**Concrete fix:** Present the fixed detector as a demanding benchmark reference: “To demonstrate added value on ADReal, a selector should improve on this strong fixed policy.” This preserves the paper's central insight without turning one benchmark result into a universal criterion for usefulness.

### 9. The conclusion should close the full argument, including its negative result

The current conclusion repeats the benchmark contamination problem and the classical SPARC victory, but does not mention the full-pool failure, the nonsignificant SPARC–Goswami comparison, or the probe's family bias. Those are now central findings, not peripheral limitations. 

**Suggested concluding arc:** ADReal provides a controlled hard-anomaly benchmark; the fixed reference shows that normal-only selection is difficult; SPARC obtains a modest classical-pool advantage, concentrated in OddBench; deep eligibility and the probe's family bias limit that advantage; and future selection methods should address these proxy and candidate-pool limitations. This would make the conclusion consistent with the title “No Easy Wins” and with the paper's strongest contribution: a more realistic account of when unsupervised model selection does and does not add value.

**Overall editorial judgment:** The manuscript no longer needs a broad conceptual rewrite. It needs a targeted final pass: correct the 0.037/0.038 discrepancy, synchronize captions and metadata, qualify the remaining universal claims, remove repeated leaderboard paragraphs, and reorder the Section 5–7 discussion so that the central result precedes pool sensitivity and the mechanism precedes the source-level interpretation. These changes would make the paper's argument considerably clearer without changing its empirical contribution.
:::