#!/usr/bin/env python3
"""Export a second short-answer QA dataset to the JSONL format generate_activations.py reads (--local-jsonl).
Needs the `datasets` package and network access to the Hugging Face hub (run it on your own machine / Colab).

    python scripts/make_qa_jsonl.py --dataset trivia_qa --out triviaqa_val.jsonl
    python scripts/make_qa_jsonl.py --dataset nq_open  --out nqopen_val.jsonl

Each line: {"question": ..., "answer": "<main answer> (also accepted: a; b; c)"}.  The aliases are folded into the gold string
because the labelling guidelines say 'use the gold answer only'.  Convert-logic is tested offline in --selftest.
"""
import argparse, json


def convert(dataset, rows, max_alias=6):
    out = []
    for r in rows:
        if dataset == "trivia_qa":
            main = r["answer"]["value"]; al = [x for x in r["answer"]["aliases"] if x.strip().lower() != main.strip().lower()]
        elif dataset == "nq_open":
            main, al = r["answer"][0], r["answer"][1:]
        else: raise ValueError(dataset)
        seen, al2 = set(), []
        for x in al:
            k = x.strip().lower()
            if k and k not in seen: seen.add(k); al2.append(x.strip())
        out.append({"question": r["question"].strip(), "answer": main + (" (also accepted: " + "; ".join(al2[:max_alias]) + ")" if al2 else "")})
    return out


def selftest():
    t = convert("trivia_qa", [{"question": "Q1? ", "answer": {"value": "Paris", "aliases": ["paris", "City of Light", "Paris, France", "City of Light"]}}])
    assert t == [{"question": "Q1?", "answer": "Paris (also accepted: City of Light; Paris, France)"}], t
    n = convert("nq_open", [{"question": "who?", "answer": ["A B", "A. B."]}, {"question": "x", "answer": ["Only"]}])
    assert n[0]["answer"] == "A B (also accepted: A. B.)" and n[1]["answer"] == "Only"
    print("selftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", choices=["trivia_qa", "nq_open"]); ap.add_argument("--out"); ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--limit", type=int, default=5000); a = ap.parse_args()
    if a.selftest: selftest(); raise SystemExit
    from datasets import load_dataset
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation") if a.dataset == "trivia_qa" else load_dataset("google-research-datasets/nq_open", split="validation")
    rows = convert(a.dataset, (ds[i] for i in range(min(a.limit, len(ds)))))
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {a.out}")
