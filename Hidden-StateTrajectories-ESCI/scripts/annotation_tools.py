#!/usr/bin/env python3
"""
Detector-blind, two-annotator labelling workflow for Reliability//Lab.

  make-sheets : build one randomised sheet per annotator from manifest.jsonl (no detector scores, no verifier labels)
  agree       : Cohen's kappa, raw agreement, confusion matrix; writes disagreements.csv for adjudication
  freeze      : merge agreed + adjudicated labels into labels.csv and write FROZEN.json (SHA-256, counts, kappa, time)
  --selftest  : synthetic end-to-end check

Order of use:
  python annotation_tools.py make-sheets --manifest acts/manifest.jsonl --out annotation/
  (annotators A and B fill `label` independently: 1 = reliable, 0 = failure; neither sees detector output)
  python annotation_tools.py agree  --a annotation/annotator_A.csv --b annotation/annotator_B.csv --out annotation/
  (a third person, or A and B jointly using GUIDELINES.md only, fill `adjudicated_label` in disagreements.csv)
  python annotation_tools.py freeze --a ... --b ... --adjudicated annotation/disagreements.csv --out annotation/
Only after FROZEN.json exists should rtm_protocol.py be run (it checks the hash).
"""
import argparse, csv, hashlib, json, os, random, sys, tempfile, time
from collections import Counter

GUIDELINES = """# Labelling guidelines (freeze this file before annotating; do not edit afterwards)

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
"""

def read_manifest(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

def write_csv(path, rows, fields, enc="utf-8-sig"):
    # utf-8-sig so Excel shows non-ASCII text correctly; labels.csv is written plain utf-8
    with open(path, "w", newline="", encoding=enc) as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

def read_csv(path):
    """Read a CSV saved by Excel/Sheets: UTF-8 (with or without BOM), else Windows cp1252."""
    raw = open(path, "rb").read()
    try: text = raw.decode("utf-8-sig")
    except UnicodeDecodeError: text = raw.decode("cp1252")
    import io
    return list(csv.DictReader(io.StringIO(text, newline="")))

def cmd_make_sheets(a):
    rows = read_manifest(a.manifest); os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, "GUIDELINES.md"), "w", encoding="utf-8", newline="\n").write(GUIDELINES)
    for name, seed in (("A", a.seed), ("B", a.seed + 1000003)):
        order = rows[:]; random.Random(seed).shuffle(order)
        out = [dict(id=r["id"], question=r["question"], gold_answer=r["gold_answer"], model_output=r["output"], label="", notes="") for r in order]
        write_csv(os.path.join(a.out, f"annotator_{name}.csv"), out, ["id", "question", "gold_answer", "model_output", "label", "notes"])
    print(f"wrote sheets for {len(rows)} items to {a.out} (independent random order per annotator)")

def _labels(path):
    d = {}
    for r in read_csv(path):
        v = r["label"].strip()
        if v not in {"0", "1", "?"}: raise SystemExit(f"{path}: id {r['id']} has invalid label {v!r} (use 0, 1 or ?)")
        d[r["id"]] = v
    return d

def kappa(x, y):
    n = len(x); po = sum(i == j for i, j in zip(x, y)) / n
    cats = sorted(set(x) | set(y)); pe = sum((x.count(c) / n) * (y.count(c) / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0, po

def cmd_agree(a):
    A, B = _labels(a.a), _labels(a.b)
    if set(A) != set(B): raise SystemExit("annotators labelled different id sets")
    ids = sorted(A); x = [A[i] for i in ids]; y = [B[i] for i in ids]
    k, po = kappa(x, y); conf = Counter(zip(x, y))
    print(f"n={len(ids)}  raw agreement={po:.3f}  Cohen kappa={k:.3f}"); print("confusion (A,B):", dict(conf))
    ref = {r["id"]: r for r in read_csv(a.a)}
    dis = [dict(id=i, question=ref[i]["question"], gold_answer=ref[i]["gold_answer"], model_output=ref[i]["model_output"], label_A=A[i], label_B=B[i], adjudicated_label="", reason="")
           for i in ids if A[i] != B[i] or A[i] == "?"]
    os.makedirs(a.out, exist_ok=True)
    write_csv(os.path.join(a.out, "disagreements.csv"), dis, ["id", "question", "gold_answer", "model_output", "label_A", "label_B", "adjudicated_label", "reason"])
    json.dump(dict(n=len(ids), raw_agreement=po, kappa=k, n_to_adjudicate=len(dis)), open(os.path.join(a.out, "agreement.json"), "w", encoding="utf-8"), indent=1)
    print(f"{len(dis)} items need adjudication -> disagreements.csv")

def sha256(path): return hashlib.sha256(open(path, "rb").read()).hexdigest()

def cmd_freeze(a):
    A, B = _labels(a.a), _labels(a.b); adj = {r["id"]: r["adjudicated_label"].strip() for r in read_csv(a.adjudicated)} if a.adjudicated else {}
    final = {}
    for i in sorted(A):
        if A[i] == B[i] and A[i] != "?": final[i] = A[i]
        else:
            v = adj.get(i, "")
            if v not in {"0", "1"}: raise SystemExit(f"id {i}: disagreement/unsure with no adjudicated 0/1 label; cannot freeze")
            final[i] = v
    os.makedirs(a.out, exist_ok=True); lp = os.path.join(a.out, "labels.csv")
    write_csv(lp, [dict(id=i, label=final[i]) for i in sorted(final)], ["id", "label"], enc="utf-8")
    x = [A[i] for i in sorted(A)]; y = [B[i] for i in sorted(A)]; k, po = kappa(x, y)
    meta = dict(labels_file="labels.csv", sha256=sha256(lp), n=len(final), n_reliable=sum(v == "1" for v in final.values()),
                n_failure=sum(v == "0" for v in final.values()), kappa_before_adjudication=k, raw_agreement=po,
                n_adjudicated=sum(1 for i in A if A[i] != B[i] or A[i] == "?"), frozen_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    json.dump(meta, open(os.path.join(a.out, "FROZEN.json"), "w", encoding="utf-8"), indent=1); print(json.dumps(meta, indent=1))
    print("Labels frozen. Do not edit labels.csv; rtm_protocol.py will refuse to run if the hash changes.")

def selftest():
    rng = random.Random(0); d = tempfile.mkdtemp(); man = os.path.join(d, "m.jsonl")
    with open(man, "w", encoding="utf-8") as f:
        for i in range(60): f.write(json.dumps(dict(id=f"q{i}", question=f"Q{i}?", gold_answer="x", output=f"out{i}")) + "\n")
    ns = argparse.Namespace(manifest=man, out=d, seed=1); cmd_make_sheets(ns)
    truth = {f"q{i}": str(int(rng.random() < .2)) for i in range(60)}
    for name in "AB":
        rows = read_csv(os.path.join(d, f"annotator_{name}.csv"))
        for r in rows: r["label"] = truth[r["id"]] if rng.random() > .1 else str(1 - int(truth[r["id"]]))
        write_csv(os.path.join(d, f"annotator_{name}.csv"), rows, ["id", "question", "gold_answer", "model_output", "label", "notes"])
    oa = [r["id"] for r in read_csv(os.path.join(d, "annotator_A.csv"))]; ob = [r["id"] for r in read_csv(os.path.join(d, "annotator_B.csv"))]
    assert oa != ob, "annotator orders must differ"
    cmd_agree(argparse.Namespace(a=os.path.join(d, "annotator_A.csv"), b=os.path.join(d, "annotator_B.csv"), out=d))
    dis = read_csv(os.path.join(d, "disagreements.csv"))
    for r in dis: r["adjudicated_label"] = truth[r["id"]]; r["reason"] = "selftest"
    write_csv(os.path.join(d, "disagreements.csv"), dis, list(dis[0].keys()) if dis else ["id"])
    cmd_freeze(argparse.Namespace(a=os.path.join(d, "annotator_A.csv"), b=os.path.join(d, "annotator_B.csv"), adjudicated=os.path.join(d, "disagreements.csv"), out=d))
    fin = {r["id"]: r["label"] for r in read_csv(os.path.join(d, "labels.csv"))}
    assert fin == truth, "adjudicated labels should equal truth in the selftest"
    assert json.load(open(os.path.join(d, "FROZEN.json"), encoding="utf-8"))["sha256"] == sha256(os.path.join(d, "labels.csv"))
    print("selftest OK")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd"); ap.add_argument("--selftest", action="store_true")
    s = sub.add_parser("make-sheets"); s.add_argument("--manifest", required=True); s.add_argument("--out", required=True); s.add_argument("--seed", type=int, default=1)
    s = sub.add_parser("agree"); s.add_argument("--a", required=True); s.add_argument("--b", required=True); s.add_argument("--out", required=True)
    s = sub.add_parser("freeze"); s.add_argument("--a", required=True); s.add_argument("--b", required=True); s.add_argument("--adjudicated"); s.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.selftest: selftest()
    elif a.cmd == "make-sheets": cmd_make_sheets(a)
    elif a.cmd == "agree": cmd_agree(a)
    elif a.cmd == "freeze": cmd_freeze(a)
    else: ap.print_help()
