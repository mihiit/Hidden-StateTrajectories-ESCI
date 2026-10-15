# Labelling brief (about 5 minutes to read)

Thank you for helping. You will judge whether short answers written by a small language model are correct. Your labels are the ground truth for a research study, so the only thing that matters is following the rules below, independently and honestly.

## What you receive

One file, `annotator_B.csv` (or `annotator_A.csv`), with 900 rows and these columns:

| Column | What it is |
|---|---|
| `id` | row identifier, do not change |
| `question` | a trivia question |
| `gold_answer` | the correct answer |
| `model_output` | the answer the language model wrote |
| `label` | **you fill this in** |
| `notes` | optional, for your own remarks |

## What to write in `label`

* `1` = the answer the model finally commits to is **correct** compared with `gold_answer`.
* `0` = otherwise.
* `?` = you are genuinely unsure. Unsure items go to a third person, so use it when it is true, not to skip work.

## The rules (identical for both labellers; do not add your own)

1. Judge the answer the model finally commits to. If it lists options and then picks one, the pick counts.
2. A correct answer stated early and never retracted counts as correct (1), even if unrelated text follows.
3. A correct answer that the model later contradicts or replaces counts as wrong (0).
4. Comparison questions ("which came first, A or B?"): the entity **and** the direction asserted must both be right.
5. Truncated outputs: judge only what was committed to before the text stops. If no answer was committed, label 0.
6. Repetition loops, or lists of near-identical options with no committed pick: label 0.
7. Use the gold answer only. Do not look anything up. If the gold answer itself looks wrong, label as if it were right and write a note.
8. Do not look at any other source: no detector scores, no other labeller's sheet, no previous labels.
9. When genuinely unsure, write `?`.

(These are the frozen `GUIDELINES.md`; its SHA-256 starts `3744633c2152`.)

## Four invented examples (not from the dataset)

| Question | Gold | Model output (shortened) | Label | Why |
|---|---|---|---|---|
| Which river flows through Paris? | Seine | "The Seine flows through Paris." | 1 | commits to the right answer |
| Who wrote *Emma*? | Jane Austen | "Charlotte Brontë wrote *Emma*." | 0 | wrong entity |
| Which is older, the Eiffel Tower or the Statue of Liberty? | Statue of Liberty | "The Eiffel Tower is older." | 0 | wrong entity (rule 4) |
| Capital of Australia? | Canberra | "Sydney is often mistaken for the capital, but the capital is Canberra. Interestingly, Canberra was founded in 1813." | 1 | commits to Canberra and never retracts it; the extra wrong date is unrelated text (rule 2) |

## How to do it in Excel

1. Open the CSV in Excel (or Google Sheets).
2. Fill the `label` column for every row with `1`, `0` or `?`. Do not change any other column, do not sort or delete rows, do not rename headers.
3. Save as **CSV UTF-8 (Comma delimited) (*.csv)**.
4. Work in sessions of about an hour and save often. Roughly 20–30 seconds per row is normal.

## Do not

* talk to the other labeller about specific items before both of you are finished;
* look at, or ask about, any model score, detector, plot or result of the study;
* change your labels after you send the file back.

## When finished

Send the saved CSV back unchanged in format. If any row was impossible to judge, write `?` and a short note instead of guessing.
