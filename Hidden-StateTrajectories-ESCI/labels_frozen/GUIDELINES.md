# Labelling guidelines (freeze this file before annotating; do not edit afterwards)

Label 1 (reliable) if the answer the model commits to is correct with respect to the gold answer.
Label 0 (failure) otherwise.

1. Judge the answer the model finally commits to. If it lists options and then picks one, the pick counts.
2. A correct answer stated early and never retracted counts as reliable, even if unrelated text follows.
3. A correct answer that the model later contradicts or replaces counts as a failure.
4. Comparison questions ("which came first, A or B?"): the entity AND the direction asserted must be right.
5. Truncated outputs: judge only what was committed to before truncation. If no answer was committed, label 0.
6. Repetition loops or lists of near-identical options with no committed pick: label 0.
7. Use the gold answer only; do not look up facts. If the gold answer itself looks wrong, label as if it were right and note it.
8. Do not look at any detector score, verifier label or other annotator's sheet.
9. When genuinely unsure, write '?' in label. Unsure items go to adjudication.
