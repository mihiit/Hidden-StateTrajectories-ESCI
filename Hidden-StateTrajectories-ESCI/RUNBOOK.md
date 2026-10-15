# Runbook: the confirmatory experiment, step by step

Goal: test H1, H2 and H3 from Section VII-D of the paper on about 900 new HaluEval QA questions, with detector-blind labels.
> Windows note: the scripts now read and write UTF-8 explicitly, so non-ASCII questions work with the default Windows encoding. Save annotator sheets from Excel as "CSV UTF-8" (plain "CSV" also works).

Run everything from one folder (call it `rl/`) that contains the `scripts/` folder from the zip. Commands are for Git Bash on Windows or any Linux/macOS shell.

## 0. Rules that make the result trustworthy (read first)

1. Do **not** look at any detector score, AUROC or plot until `FROZEN.json` exists.
2. Do **not** change the classifier grid, folds, seeds or hypotheses after labels are frozen.
3. Do **not** drop questions you dislike (short answers, odd outputs). Report every question you generated.
4. Annotators A and B label **independently**, in separate files, without discussing individual items.
5. Report every outcome, including a null result. A null on a fair test is still a result worth publishing.

## 1. Where to run it

| Option | Cost | Time for 900 questions |
|---|---|---|
| Colab / Kaggle / any GPU (T4 or better) | free tier works | roughly 2-4 h (my estimate, not measured here) |
| Your CPU | free | about 15-22 h (the pilot's measured 1-1.5 min per question) |

Disk: hidden states are float16, shape (29 layers, up to 256 tokens, 1536). That is up to about 22 MB per question before compression, so plan for **15-20 GB** for 900 questions. On Colab, write `--out` to Google Drive or a mounted disk with that much space.

## 2. Setup (once)

```bash
cd rl
python -m venv .venv && source .venv/Scripts/activate      # Windows Git Bash; on Linux/macOS: source .venv/bin/activate
pip install torch transformers numpy scipy scikit-learn
python scripts/annotation_tools.py --selftest               # expect: selftest OK
python scripts/rtm_protocol.py --selftest --out /tmp/st     # expect: selftest OK
python scripts/tests/test_capture.py                        # expect: capture test OK
curl -L -o qa_data.json https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json
python -c "import json;r=[json.loads(l) for l in open('qa_data.json')];print(len(r),r[0].keys())"   # expect 10000 and question / right_answer
```

## 3. The prompt (do not skip)

Create `pilot_prompt.txt` containing your pilot's exact user prompt, with `{question}` where the question goes. If you no longer have it, write one now, and then **treat the pilot as a separate earlier study** (use option 4B below). Any prompt you choose is frozen from this point.

## 4. Choose the question pool

**4A. Pilot activations and prompt are available.** List the pilot's 200 questions, one per line, into `used_questions.txt`, generate 700 new ones, and later re-label the pilot's 200 together with them (use `--n 700 --exclude used_questions.txt`).

**4B. Otherwise (recommended, cleaner).** Generate all 900 fresh with the frozen prompt and drop `--exclude` (use `--n 900`). The pilot stays a historical result in the paper.

## 5. Smoke test, then the full run

```bash
python scripts/generate_activations.py --n 3 --seed 2026 --local-jsonl qa_data.json --prompt-file pilot_prompt.txt --out smoke/
cat smoke/manifest.jsonl | head -c 1500            # read the 3 outputs: sensible answers? truncated flags sensible?
python -c "import numpy as np,glob;f=glob.glob('smoke/*.npz')[0];h=np.load(f)['hidden'];print(h.shape,h.dtype)"   # expect (29, T, 1536) float16
```

If the outputs look wrong (empty, no answer, odd chat formatting), fix the prompt now, before the full run. If it looks right:

```bash
python scripts/generate_activations.py --n 900 --seed 2026 --local-jsonl qa_data.json --prompt-file pilot_prompt.txt --out acts/
# add:  --exclude used_questions.txt --n 700   for option 4A
```

It is resumable: if it stops, run the same command again. Do not change `--seed` or the prompt between restarts.

Check the result (this is a data check, not a peek at detector performance):

```bash
python - <<'EOF'
import json
m=[json.loads(l) for l in open('acts/manifest.jsonl')]
n=[r['n_tokens'] for r in m]
print('questions',len(m),'unique ids',len({r['id'] for r in m}))
print('min/median/max tokens',min(n),sorted(n)[len(n)//2],max(n),'| truncated',sum(r['truncated'] for r in m))
EOF
```

Expect 900 unique ids. **If the minimum token count is below 6, tell me before continuing**: the layer-axis features need a few tokens, and I will adjust the analysis rule before labels exist, not after.

## 6. Blind labelling

```bash
python scripts/annotation_tools.py make-sheets --manifest acts/manifest.jsonl --out annotation/
```

This writes `GUIDELINES.md` and two sheets with independent random orders. **Freeze `GUIDELINES.md` now**: do not edit it after anyone starts labelling. Save its SHA-256 (`sha256sum annotation/GUIDELINES.md`) in your notes.

- A and B are two different people (you can be one of them; the second can be a co-author or a classmate who has not seen detector outputs).
- Each opens their own `annotator_X.csv` in Excel or Sheets and fills the `label` column with `1` (reliable), `0` (failure) or `?` (unsure), using only the guidelines and the gold answer. Save as CSV (UTF-8). Budget about 20-40 seconds per item, so 5-10 hours each; split it over days.
- Neither sees the other's sheet, any detector output, or the pilot's verifier labels.

## 7. Agreement, adjudication, freeze

```bash
python scripts/annotation_tools.py agree  --a annotation/annotator_A.csv --b annotation/annotator_B.csv --out annotation/
# open annotation/disagreements.csv; a third person (or A and B together, using GUIDELINES.md only) fills adjudicated_label with 0 or 1
python scripts/annotation_tools.py freeze --a annotation/annotator_A.csv --b annotation/annotator_B.csv --adjudicated annotation/disagreements.csv --out annotation/
cat annotation/FROZEN.json
```

Record Cohen's kappa from `agree` (it goes in the paper). Sanity checks on `FROZEN.json`:

- `n` is 900 (or 700 plus 200 under option 4A).
- `n_reliable` is at least about 100. At the pilot's 12.5% rate you expect about 112. If it is below about 60, the study is underpowered: generate more questions (same command with a larger `--n`, same seed) and label the new ones before continuing.
- Kappa below 0.6 means the guidelines were ambiguous. Report it honestly; do not edit labels afterwards.

## 8. Run the detectors (first time anyone sees a score)

```bash
python scripts/rtm_protocol.py --feats-dir acts/ --labels annotation/labels.csv --frozen annotation/FROZEN.json --out results/
```

The script refuses to run if `labels.csv` changed after freezing. It scores six feature families out-of-fold (F, Llast, L, T, A = full RTM, len) with 5-fold x 10 repeats, tuning the penalty inside training folds, and writes `results/results.json`, `results/results.csv` and `results/oof_scores.npy`. Expect tens of minutes on CPU.

## 9. How to read the result (decision rules are fixed in the paper)

| Test | Pass condition | If it passes | If it fails |
|---|---|---|---|
| H1 (primary) | lower 95% bound of AUROC(A) - AUROC(F) is above 0, and the point estimate is at least 0.05 | Report: trajectories add information over a fair static probe on this model and dataset | Upper bound below 0.05: report that the study rules out a meaningful gain. Otherwise: inconclusive |
| H2 | lower bound of AUROC(A) - AUROC(len) above 0 | Gain is not a length proxy | RTM cannot be told apart from response length; the paper's central interpretation becomes "length explains it" |
| H3 | A beats Llast, L and T, Holm-corrected | The full trajectory beats each partial one | Report which single axis carries the signal |

Look at `results.json`: `detectors` gives each family's AUROC with a 95% interval, and `paired_vs_A_full_RTM` gives the paired difference, interval, raw p and Holm p for A against each other family.

## 10. Optional: second model (transfer, separate experiment)

```bash
python scripts/generate_activations.py --model Qwen/Qwen2.5-3B-Instruct --n 900 --seed 2026 --local-jsonl qa_data.json --prompt-file pilot_prompt.txt --out acts_3b/
```

The 3B model writes different answers, so it needs its **own** blind labelling and freeze (repeat steps 6-8 with `acts_3b/`, `annotation_3b/`, `results_3b/`). Float32 for 3B needs about 12 GB of memory; use a GPU. Never pool its data with the 1.5B data. A second dataset (TruthfulQA) is not supported by `generate_activations.py` yet; it needs a small loader, which I can add.

## 11. What to send back

Send me `results/results.json`, `results/results.csv`, `annotation/agreement.json`, `annotation/FROZEN.json`, and the output of the step-5 data check. Do not send the hidden-state files. I will put the real numbers into the paper (abstract, Results, Table VII, Fig. 1, conclusion) and rewrite the claims to match whichever way the three hypotheses came out.
