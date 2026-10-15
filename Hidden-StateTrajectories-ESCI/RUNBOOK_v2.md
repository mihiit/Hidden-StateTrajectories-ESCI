# Runbook v2: the four remaining gaps (run on the machine that holds `acts/`)

Everything below was tested on synthetic data and, for the human-validation sheets, on the real frozen labels. NOTHING here has been run
on the real hidden states yet: the paper therefore still lists these items as not done. Send back the JSON files named in each step.

Setup once: copy the new `scripts/*.py`, `analysis_plan_v2.md` and the `human_validation/` folder next to your existing `scripts/`.
Run `sha256sum analysis_plan_v2.md` BEFORE step 1 and keep the hash (the paper quotes it).
```bash
pip install joblib            # the only new dependency
python scripts/tests/test_pipeline.py     # ~1 min, expect: ALL TESTS PASSED
```

## 1. Cache the features (reads the 15-20 GB once, writes a few MB)
```bash
python scripts/featurize.py --feats-dir acts/ --labels annotation/labels.csv --out cache/features.npz --n-jobs 4
```
Expect `integrity check OK` (it rebuilds the confirmatory families from the cache and compares them with `rtm_protocol.families`).

## 2. Statistics: reproduce, then robustness  (item 2)
```bash
python scripts/stats_robustness.py --features cache/features.npz --labels annotation/labels.csv --frozen annotation/FROZEN.json \
    --reference confirmatory_results.json --out results_v2/ --n-jobs 4
```
Expect `REPRODUCTION OK` (published AUROCs within 0.003). If it says MISMATCH stop and tell me (different scikit-learn version is the usual cause).
Send back `results_v2/stats_robustness.json`. Time: 20-60 min on CPU (F has 1536 features).

## 3. Length-adjusted evaluation on the real data  (item 1, the most important)
```bash
python scripts/length_adjusted_eval.py --features cache/features.npz --labels annotation/labels.csv --frozen annotation/FROZEN.json \
    --out results_v2/ --n-jobs 4
```
Reuses the cached out-of-fold scores of step 2 (same `--out`). Send back `results_v2/length_adjusted.json`. Time: 1-3 h on CPU.
Do not change settings after seeing numbers; the decision rules L1/L2 are in `analysis_plan_v2.md`.

## 4. Human validation  (item 3)
Two people (you and Priyanka), independently, blind to the LLM labels and to all detector output:
1. Give each person ONLY their sheet (`human_validation/human_sheet_H1.csv` / `_H2.csv`), `GUIDELINES.md`, `INSTRUCTIONS.md`. Keep `sample_key.csv` private.
2. They fill the `label` column (1 / 0 / ?) and save as `human_sheet_H1_filled.csv` / `human_sheet_H2_filled.csv`. About 2 h each (179 items).
3. Analyse:
```bash
python scripts/human_validation.py analyze --key human_validation/sample_key.csv --labels annotation/labels.csv \
    --sheet human_validation/human_sheet_H1_filled.csv --sheet human_validation/human_sheet_H2_filled.csv --out human_validation/
```
   It writes `human_adjudication_TEMPLATE.csv` for items the two humans disagreed on; fill `label` (1/0) by discussion under the guidelines, save as
   `human_adjudication.csv`, rerun with `--adjudicated human_validation/human_adjudication.csv`.
4. Label sensitivity reruns (exploratory flag is deliberate: the labels differ from the frozen file):
```bash
for L in labels_human_corrected labels_adjudication_flipped; do
  python scripts/stats_robustness.py --features cache/features.npz --labels human_validation/$L.csv --allow-unfrozen --out results_v2_$L/ --n-jobs 4
done
```
Send back `human_validation/human_validation_results.json` and the two `results_v2_*/stats_robustness.json`.

## 5. Second model (and optional dataset)  (item 4)
Use a GPU (Colab T4 is enough for Qwen2.5-3B in bfloat16; on CPU only a 0.5B model is realistic).
```bash
# same 900 questions as the main study: same seed, same prompt file, same local HaluEval file
python scripts/generate_activations.py --model Qwen/Qwen2.5-3B-Instruct --dtype bfloat16 --n 900 --seed 2026 \
    --local-jsonl qa_data.json --prompt-file pilot_prompt.txt --out acts_3b/
```
Then repeat RUNBOOK.md steps 6-8 with `acts_3b/`, `annotation_3b/`, `results_3b/` (new blind labelling and a new freeze; never pool with the 1.5B data),
then steps 1-3 above with `cache_3b/` and `results_v2_3b/`. Send me `acts_3b/manifest.jsonl` (small) and I can do the model-annotator pass as before
(it will be disclosed as model-annotated again); the human check in step 4 should be repeated on a sample if you can.
Optional second dataset: `python scripts/make_qa_jsonl.py --dataset trivia_qa --out triviaqa_val.jsonl` (needs Hugging Face access), then the same
generation command with `--local-jsonl triviaqa_val.jsonl --n 900`. Its loader logic is tested offline only.

## 6. Calibration simulation of the test procedure (already run in the cloud; optional to rerun)
`python scripts/sim_calibration.py --scenario null --reps 300 --out sim_null.json` (about 1 h on 2 cores); see `sim_out/`.
