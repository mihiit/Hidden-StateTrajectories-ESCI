#!/usr/bin/env python3
"""
Generate answers AND per-layer hidden-state trajectories for new HaluEval QA questions.

  python generate_activations.py --n 700 --seed 2026 --exclude used_questions.txt --out acts/
  python generate_activations.py --model Qwen/Qwen2.5-3B-Instruct --n 900 --out acts_3b/     # second model

Writes, per question, <id>.npz with
    hidden   float16 (layers+1, T, d)  state that PRODUCED each generated token (step 0 = last prompt position)
    logprob  float32 (T,)  entropy float32 (T,)  token_ids int32 (T,)
and appends manifest.jsonl (id, question, gold_answer, output, n_tokens, truncated).
Resumable: ids already present in manifest.jsonl are skipped.

IMPORTANT: use the SAME prompt as the pilot (--prompt-file, text containing {question}). The default below is a placeholder.
Verify the dataset field names before a long run: the loader expects question / right_answer, as in the HaluEval QA config.
Cost: ~1-1.5 CPU-minutes per question for a 1.5B model at 256 tokens (the pilot's measured speed); much faster on a GPU.
"""
import argparse, hashlib, json, os, random, re, sys
import numpy as np

DEFAULT_PROMPT = "Answer the question. Think briefly, then state your final answer clearly.\n\nQuestion: {question}"

def norm(q): return re.sub(r"\s+", " ", q.strip().lower())
def qid(i, q): return f"q{i:05d}_{hashlib.md5(norm(q).encode()).hexdigest()[:8]}"

def capture_generation(model, input_ids, max_new_tokens, eos_token_id=None):
    """Greedy generation with hidden states. Returns dict(hidden (L+1,T,d), logprob, entropy, token_ids, truncated)."""
    import torch
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False, output_hidden_states=True,
                             output_scores=True, return_dict_in_generate=True, eos_token_id=eos_token_id, pad_token_id=eos_token_id)
    steps = out.hidden_states                     # tuple over generated steps; each a tuple over (L+1) layers of (1, seq, d)
    H = []
    for s, layers in enumerate(steps):
        col = -1 if s == 0 else 0                 # step 0 holds the whole prompt: take its last position
        H.append(torch.stack([h[0, col, :] for h in layers], 0))          # (L+1, d)
    hidden = torch.stack(H, 1).float().cpu().numpy()                      # (L+1, T, d)
    gen = out.sequences[0, input_ids.shape[1]:]
    T = min(len(gen), len(out.scores))
    lp, ent = [], []
    for t in range(T):
        logp = torch.log_softmax(out.scores[t][0].float(), -1)
        lp.append(float(logp[gen[t]])); ent.append(float(-(logp.exp() * logp).sum()))
    truncated = not (eos_token_id is not None and int(gen[-1]) == eos_token_id) if T else True
    return dict(hidden=hidden[:, :T], logprob=np.array(lp, np.float32), entropy=np.array(ent, np.float32), token_ids=gen[:T].cpu().numpy().astype(np.int32), truncated=bool(truncated))

def load_questions(args):
    if args.local_jsonl:
        rows = [json.loads(l) for l in open(args.local_jsonl, encoding="utf-8") if l.strip()]
        return [(r["question"], r.get("right_answer") or r["answer"]) for r in rows]
    from datasets import load_dataset
    ds = load_dataset("pminervini/HaluEval", "qa", split="data")
    return [(r["question"], r["right_answer"]) for r in ds]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct"); ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=2026); ap.add_argument("--exclude", help="text file, one already-used question per line")
    ap.add_argument("--out", required=True); ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"]); ap.add_argument("--device", default="auto")
    ap.add_argument("--prompt-file"); ap.add_argument("--local-jsonl")
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    os.makedirs(a.out, exist_ok=True); mpath = os.path.join(a.out, "manifest.jsonl")
    done = {json.loads(l)["id"] for l in open(mpath, encoding="utf-8")} if os.path.exists(mpath) else set()
    template = open(a.prompt_file, encoding="utf-8-sig").read() if a.prompt_file else DEFAULT_PROMPT
    if "{question}" not in template: sys.exit("prompt must contain {question}")
    used = {norm(l) for l in open(a.exclude, encoding="utf-8-sig")} if a.exclude else set()
    qs = [(q, g) for q, g in load_questions(a) if norm(q) not in used]
    random.Random(a.seed).shuffle(qs); qs = qs[:a.n]
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=getattr(torch, a.dtype)).to(dev).eval()
    eos = tok.eos_token_id
    with open(mpath, "a", encoding="utf-8") as mf:
        for i, (q, gold) in enumerate(qs):
            idx = qid(i, q)
            if idx in done: continue
            msgs = [{"role": "user", "content": template.format(question=q)}]
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
            ids = (ids["input_ids"] if not torch.is_tensor(ids) else ids).to(dev)   # newer transformers return a BatchEncoding
            r = capture_generation(model, ids, a.max_new_tokens, eos)
            text = tok.decode(r["token_ids"], skip_special_tokens=True)
            np.savez_compressed(os.path.join(a.out, idx + ".npz"), hidden=r["hidden"].astype(np.float16), logprob=r["logprob"], entropy=r["entropy"], token_ids=r["token_ids"])
            mf.write(json.dumps(dict(id=idx, question=q, gold_answer=gold, output=text, n_tokens=int(len(r["token_ids"])), truncated=r["truncated"], model=a.model)) + "\n"); mf.flush()
            print(f"[{i+1}/{len(qs)}] {idx} tokens={len(r['token_ids'])}", flush=True)

if __name__ == "__main__":
    main()
