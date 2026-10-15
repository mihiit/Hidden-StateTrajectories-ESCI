# Instructions for the human annotators (179 items)

You are checking labels that were produced by language-model annotators. You do not see their labels and must not look for them.

1. Open ONLY your own sheet (`human_sheet_H1.csv` or `human_sheet_H2.csv`). Do not open the other person's sheet or `sample_key.csv`.
2. Read `GUIDELINES.md` first (SHA-256 prefix 35918e0d9a994d62). Apply exactly those rules; they are the rules the model annotators used.
3. For each row read `question`, `gold_answer`, `model_output` and fill `label`: `1` = reliable (correct under the guidelines), `0` = failure, `?` = genuinely unsure. Use `notes` for anything unusual.
4. Work independently of the other annotator; do not discuss individual items until both sheets are finished.
5. Save as **CSV UTF-8** under the name `human_sheet_H1_filled.csv` / `human_sheet_H2_filled.csv`. Do not reorder, add or delete rows.
6. About 30-40 seconds per item; split over two sittings. Do not run any detector or look at any detector output while labelling.
