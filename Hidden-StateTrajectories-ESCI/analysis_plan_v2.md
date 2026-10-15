# Analysis plan v2 (written BEFORE the new analyses are run on the real hidden states)

Status of this plan, stated honestly: the confirmatory H1-H3 analysis (RUNBOOK.md) was already run and its results are known to the
authors. The analyses below were chosen after seeing those results, in response to a reviewer-style critique, but BEFORE any of
them was run on the real data. They are therefore *secondary, pre-specified-before-running* analyses, not part of the original
pre-registration. Record the hash of this file before running anything (`sha256sum analysis_plan_v2.md`) and quote it in the paper.

## A. Statistical procedure (scripts/stats_robustness.py)
Reproduce the confirmatory AUROCs (max |difference| <= 0.003 to confirmatory_results.json, else stop and investigate). Then
re-test H1/H2/H3 with: pooled-score bootstrap (original), mean-of-repeats bootstrap, partition-resampling bootstrap and the
Nadeau-Bengio corrected resampled t-test on the 50 fold-level AUROC differences. Multiplicity: H1 and H2 single tests, H3 Holm
over three contrasts.
Decision: "H1 gain ruled out" if the upper 95% bound of AUROC(A)-AUROC(F) is below 0.05. Report whether this holds under ALL four
procedures. If it fails under any, the paper must say the exclusion of a 0.05 gain depends on the inference procedure.

## B. Length adjustment (scripts/length_adjusted_eval.py)
Adjustments: residualisation on a cubic polynomial of log token count (fitted on training folds only), length covariates
concatenated, within-length-stratum AUROC (5 quantile bins merged to >= 10 reliable and >= 10 failed each), mean/last-only
summaries (A_nomax), and a flexible length-only control (len_flex).
L1 (H1 robust to length): upper 95% bounds of AUROC(A_r)-AUROC(F_r) and of the within-stratum A-F are both < 0.05.
    Overturned if, under either, the lower bound > 0 and the point estimate >= 0.05. Otherwise inconclusive.
L2 (signal beyond length): lower 95% bounds of AUROC(A_r) > 0.5, of AUROC(A_r)-AUROC(len_flex) > 0 and of the within-stratum
    AUROC(A) > 0.5.
All rows of the output table are reported, including those that do not support the paper's story.

## C. Human validation (scripts/human_validation.py)
179 items: 80 random LLM-reliable and 80 random LLM-failed (adjudicated items excluded from these draws) plus all 19 adjudicated
items. Two people label blind to the LLM labels under the frozen GUIDELINES.md; disagreements are resolved by discussion under the
same guidelines. Report: pi, rho, gamma = pi - rho with 95% intervals; human-human and human-LLM kappa; unsure counts with bounds;
the Corollary bound U/gamma_lower for H1. Sensitivity reruns of stats_robustness.py with (i) labels_human_corrected.csv and
(ii) labels_adjudication_flipped.csv (all 19 adjudicated items flipped, the extreme case).
Decision: the H1 conclusion is called label-robust if U/gamma_lower < 0.05 and the two sensitivity reruns keep the upper bound < 0.05.

## D. Second model (and optionally a second dataset)
Same prompt, decoding, question list (same seed), classifier, folds and rules, new blind labelling and new freeze; never pooled
with the 1.5B data. H1 is re-tested with the same decision rule; no hypothesis is added.
