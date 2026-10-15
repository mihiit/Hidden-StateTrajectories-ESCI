# Reliability//Lab, package v2

New in v2 (all tested on synthetic data; the real-data runs are listed in RUNBOOK_v2.md and are NOT done yet):

| File | Purpose | Tested how |
|---|---|---|
| `scripts/rl_common.py` | shared CV engine (identical splits/model/grid to `rtm_protocol.py`), statistics | `tests/test_pipeline.py` T1-T3: rebuilds `families()` and `oof_scores()` exactly |
| `scripts/featurize.py` | caches feature blocks from `acts/` (few MB) | integrity check against `rtm_protocol.families` |
| `scripts/stats_robustness.py` | reproduces the confirmatory AUROCs, re-tests H1-H3 with 4 inference procedures | synthetic end-to-end |
| `scripts/length_adjusted_eval.py` | residualised, covariate, within-length-stratum and no-max analyses with pre-set decision rules | synthetic null-length world (leakage appears, then vanishes) and signal world (real signal survives) |
| `scripts/human_validation.py` | blinded stratified human sample, pi/rho/gamma estimation, label-sensitivity files | simulated humans on the real frozen label ids (recovery and 95% coverage) |
| `scripts/sim_calibration.py` | type-I error / coverage / rule operating characteristics of the paired tests | results in `sim_out/` |
| `scripts/make_qa_jsonl.py` | second dataset export (TriviaQA / NQ-Open) | conversion logic only |
| `human_validation/` | the REAL blinded sheets for two annotators (179 items) + instructions; `sample_key.csv` must stay private | `human_validation.py selftest` |
| `analysis_plan_v2.md` | decision rules fixed before the real-data runs | n/a |
| `RUNBOOK_v2.md` | commands, in order, with what to send back | n/a |

Original files (`RUNBOOK.md`, `scripts/rtm_protocol.py`, `labels_frozen/`, `confirmatory_results.json`, ...) are unchanged.
Run `python scripts/tests/test_pipeline.py` first on your machine (about a minute).
