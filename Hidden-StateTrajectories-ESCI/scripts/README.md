# Reliability//Lab confirmatory study: run order

Everything here is code. Nothing in this folder contains confirmatory results.

| Step | Script | Tested how |
|---|---|---|
| 1. Generate answers + hidden states for new HaluEval questions | `generate_activations.py` | capture logic tested offline on a tiny random Qwen2 model (`tests/test_capture.py`); model/dataset loading not tested |
| 2. Blind two-annotator labels, agreement, adjudication, freeze | `annotation_tools.py` | synthetic end-to-end self-test (`--selftest`) |
| 3. Matched static baseline, five feature families, length control, paired tests | `rtm_protocol.py` | synthetic self-test (`--selftest`) and a frozen-label hash check |
| Power, null, label-noise and length-confound simulations (already run) | `sim.py`, `label_noise.py`, `length_confound.py` | outputs in `figures/*.json` |
| Figures | `make_figs.py` | outputs in `figures/` |

## Order of use

```bash
# 0. List the 200 pilot questions (one per line) so they are not drawn again -> used_questions.txt

# 1. Generate ~700 new questions (total ~900). Use the SAME prompt as the pilot via --prompt-file.
python generate_activations.py --n 700 --seed 2026 --exclude used_questions.txt --prompt-file pilot_prompt.txt --out acts/
#    second model, same questions are NOT guaranteed: use the same --seed and --exclude
python generate_activations.py --model Qwen/Qwen2.5-3B-Instruct --n 700 --seed 2026 --exclude used_questions.txt --out acts_3b/

# 2. Blind labelling (annotators never see detector output or verifier labels)
python annotation_tools.py make-sheets --manifest acts/manifest.jsonl --out annotation/
#    A and B fill the `label` column independently (1 reliable, 0 failure, ? unsure)
python annotation_tools.py agree --a annotation/annotator_A.csv --b annotation/annotator_B.csv --out annotation/
#    fill adjudicated_label in annotation/disagreements.csv using GUIDELINES.md only
python annotation_tools.py freeze --a annotation/annotator_A.csv --b annotation/annotator_B.csv \
       --adjudicated annotation/disagreements.csv --out annotation/

# 3. Only now run the detectors (refuses to run if labels changed after freezing)
python rtm_protocol.py --feats-dir acts/ --labels annotation/labels.csv --frozen annotation/FROZEN.json --out results/
```

The pilot's 200 questions should be re-labelled the same way and added to the pool, so there is one labelling standard.

## Things to check before a long run

* **Prompt.** `generate_activations.py` ships a placeholder prompt. Pass the pilot's exact prompt with `--prompt-file`
  (text containing `{question}`), otherwise new and old questions are not comparable.
* **Dataset.** The HaluEval QA file is on GitHub: `curl -L -o qa_data.json https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json`,
  then pass `--local-jsonl qa_data.json`. Verified: 10,000 rows, 10,000 unique questions, fields `question` and `right_answer`; the local loader reads it correctly.
  Without `--local-jsonl` the script tries the Hugging Face copy (`pminervini/HaluEval`, config `qa`), which was not reachable where this was written.
* **Hidden-state convention.** `hidden[:, t]` is the state that *produced* generated token *t* (step 0 uses the last prompt
  position). Use the same convention as the pilot; if the pilot indexed differently, features will not be comparable.
* **Cost.** About 1-1.5 CPU-minutes per question for a 1.5B model at 256 tokens (the pilot's measured speed), so 700 questions
  is roughly 10-16 hours on CPU; far less on a GPU. The script is resumable.
* **Storage.** Hidden states are stored as float16; expect tens of MB per generation at 256 tokens for a 1.5B model.

## What `rtm_protocol.py` reports

For each family (F final state, Llast B4-style, L layer-axis, T token-axis, A = L+T = RTM, len length-only):
pooled out-of-fold AUROC with a bootstrap 95% interval and a one-sided p against 0.5; then paired DeLong p-values,
paired bootstrap intervals and Holm-corrected p-values for A against every other family.
Standardisation, L2 penalty selection (inner 3-fold CV) and all fitting happen inside training folds.
Pre-registered decisions (H1-H3) are in Section VII-D of the paper; do not change the classifier grid or folds after labels are frozen.

## Test status

* `annotation_tools.py --selftest` and `rtm_protocol.py --selftest` pass on synthetic data. Synthetic numbers carry no scientific meaning.
* The frozen-label check was exercised: a matching hash runs, an edited labels file and a missing `--frozen` both refuse.
* `tests/test_capture.py` passes: on a tiny randomly initialised Qwen2 model, the captured hidden states have the right shape and
  match a plain forward pass at generation steps 0 and 1 for every layer, and the arrays feed straight into the analysis features.
* NOT tested: downloading Qwen, loading HaluEval, the chat template and the main loop. The model download was blocked in the
  environment where this was written. Run `--n 3` first and inspect `manifest.jsonl` before starting the full run.
