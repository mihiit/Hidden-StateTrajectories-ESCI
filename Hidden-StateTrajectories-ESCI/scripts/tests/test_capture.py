"""Offline test of hidden-state capture: tiny random Qwen2 model, no downloads.
Run from the folder containing generate_activations.py:   python tests/test_capture.py   (needs torch + transformers)"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, torch
from transformers import Qwen2Config, Qwen2ForCausalLM
import generate_activations as G
torch.manual_seed(0)
cfg = Qwen2Config(vocab_size=200, hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
m = Qwen2ForCausalLM(cfg).eval(); ids = torch.randint(5, 200, (1, 11))
r = G.capture_generation(m, ids, max_new_tokens=9, eos_token_id=1)
L1, T, d = r["hidden"].shape
assert (L1, d) == (cfg.num_hidden_layers + 1, 32) and T == len(r["logprob"]) == len(r["entropy"]) == len(r["token_ids"]) == 9
with torch.no_grad():
    fw = m(ids, output_hidden_states=True).hidden_states
    ids2 = torch.cat([ids, torch.tensor([[int(r["token_ids"][0])]])], 1); fw2 = m(ids2, output_hidden_states=True).hidden_states
for l in range(L1):
    assert np.allclose(r["hidden"][l, 0], fw[l][0, -1].numpy(), atol=1e-5), f"layer {l}: step 0"
    assert np.allclose(r["hidden"][l, 1], fw2[l][0, -1].numpy(), atol=1e-4), f"layer {l}: step 1"
assert np.all(r["logprob"] <= 0) and np.all(r["entropy"] >= 0)
print("capture test OK: shapes and per-layer states match a plain forward pass")
