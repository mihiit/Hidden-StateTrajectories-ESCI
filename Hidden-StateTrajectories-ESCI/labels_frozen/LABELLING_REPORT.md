# Labelling report (read before using the labels)

## Who labelled
Both annotator sheets were labelled by **model annotators, not people**: independent Claude Sonnet 5.5 sub-agents, 100 rows each in separate contexts, one pass per sheet, under the frozen `GUIDELINES.md`, seeing only `question`, `gold_answer`, `model_output` (no detector output, no other sheet). The 19 contested items were adjudicated by the main assistant under the same rules. **The paper must say this.** It cannot claim two human annotators.

## Numbers
| | |
|---|---|
| Items | 900, 900 unique ids |
| Raw agreement A vs B | 98.8% |
| Cohen kappa (before adjudication) | 0.967 |
| Items needing adjudication | 19 (every one involved a `?`; the two sheets never disagreed 1 vs 0) |
| Adjudicated to 1 / 0 | 5 / 14 |
| Final reliable (1) / failure (0) | **199 / 701** (22.1% reliable) |
| labels.csv SHA-256 | a7467ae0b2bc1f6ed38436663747148b926188814a66996ea28e6f1f423dc60a |

**Do not read kappa 0.967 as human inter-annotator reliability.** Both passes are the same model family reading the same rules, so their errors are correlated and agreement is inflated. It shows the rules are applied consistently, not that the labels are free of error.

## Checks done
* Structure: every chunk file matched its input ids and order; all values were 0, 1 or ?.
* Two chunks (A8, A9) were re-run independently: 100/100 and 97/100 identical to the originals.
* Hand check: 25 random agreed items (14 labelled 0, 11 labelled 1) read against gold under the rules. No clear error. Three agreed 1s are lenient, not wrong: q00018 (misses "screenwriter"), q00234 (pen-name confusion but names the right person), q00204 (right region, wrong route detail). With 0 errors in 25, the 95% upper bound on the error rate is about 12%, so this is not proof of clean labels.

## Adjudication rule used (stated so you can overrule it)
An answer broader or less specific than the gold answer counts as a failure (musician vs singer, comedy vs gross-out humor, North America vs eastern North America). Reasons for each of the 19 are in `disagreements.csv`. Changing this rule changes up to about 7 labels.

## Strongly recommended before submitting
1. Have one person label a random 100-150 of the 900 items blind and report agreement with `labels.csv`. That is the real reliability number.
2. Re-run the headline analysis with the 19 adjudicated items flipped (sensitivity analysis).

## Using the files
Copy `labels.csv` and `FROZEN.json` unchanged into `annotation/` (do not open and re-save them in Excel: the hash would change), then:
`python scripts/rtm_protocol.py --feats-dir acts/ --labels annotation/labels.csv --frozen annotation/FROZEN.json --out results/`
