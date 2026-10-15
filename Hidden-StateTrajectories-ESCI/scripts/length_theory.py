"""Numerical check of Proposition 1 (length leakage of max summaries) and Proposition 2 (label-noise attenuation).
Run: python scripts/length_theory.py   (writes figs/theory_check.json)"""
import numpy as np, json
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
rng = np.random.default_rng(11)
def d_for_auc(a): return np.sqrt(2) * norm.ppf(a)
out = {"p1": [], "p2": []}
# ---- Proposition 1: AUROC of the max of iid noise tokens equals E[T_f/(T_f+T_r)]
for a in [0.5, 0.6, 0.7, 0.8]:
    N = 400000
    y = (rng.random(N) < 0.5).astype(int)                      # balanced here: AUROC is prevalence-free
    T = np.clip(np.exp(rng.normal(4.0, 0.6, N) - y * d_for_auc(a) * 0.6).astype(int), 8, 256)
    Tf, Tr = T[y == 0], T[y == 1]                              # failed (long), reliable (short)
    idx = rng.permutation(min(len(Tf), len(Tr)))[:100000]
    lam = np.mean(Tf[idx] / (Tf[idx] + Tr[idx]))               # theory: lambda = E[Tf/(Tf+Tr)] with independent draws
    auc_len = roc_auc_score(y, -T) if False else roc_auc_score(1 - y, T)   # AUROC of length for 'failed'
    # Monte Carlo of the actual max summary, one noise channel (2000 pairs sampled)
    ia, ib = rng.integers(0, len(Tf), 40000), rng.integers(0, len(Tr), 40000)
    mf = np.array([rng.normal(size=t).max() for t in Tf[ia]]); mr = np.array([rng.normal(size=t).max() for t in Tr[ib]])
    mc = np.mean(mf > mr)
    # 12-channel average of maxima (the score a probe would approach if weights were equal)
    m12f = np.array([rng.normal(size=(12, t)).max(1).mean() for t in Tf[ia[:8000]]]); m12r = np.array([rng.normal(size=(12, t)).max(1).mean() for t in Tr[ib[:8000]]])
    out["p1"].append({"length_auc": a, "lambda_theory": float(lam), "mc_one_channel": float(mc), "mc_12_channel_mean": float(np.mean(m12f > m12r)), "auc_len_pairs": float(np.mean(Tf[ia] > Tr[ib]) + .5 * np.mean(Tf[ia] == Tr[ib]))})
# ---- Proposition 2: AUROC_obs - 1/2 = (PPV - FN) (AUROC_true - 1/2)
for A in [0.6, 0.78, 0.9]:
    for ppv, fn in [(0.875, 0.022), (0.69, 0.04), (0.95, 0.01)]:
        n = 600000; prev = 0.125; d = d_for_auc(A)
        y = (rng.random(n) < prev).astype(int); s = rng.normal(size=n) + d * y
        # draw observed labels so that P(true pos | obs pos)=ppv and P(true pos | obs neg)=fn, errors independent of s given y
        # obs-pos probability among true pos = a, among true neg = b: solve for (a,b)
        P, Nn = prev, 1 - prev
        # ppv = P a/(P a + Nn b) ; fn = P(1-a)/(P(1-a)+Nn(1-b))
        # solve numerically
        from scipy.optimize import fsolve
        f = lambda v: [P*v[0]/(P*v[0]+Nn*v[1]) - ppv, P*(1-v[0])/(P*(1-v[0])+Nn*(1-v[1])) - fn]
        a_, b_ = fsolve(f, [0.9, 0.02])
        yo = np.where(y == 1, rng.random(n) < a_, rng.random(n) < b_).astype(int)
        out["p2"].append({"A": A, "ppv": ppv, "fn": fn, "mc": float(roc_auc_score(yo, s)), "formula": 0.5 + (ppv - fn) * (A - 0.5)})
json.dump(out, open("figs/theory_check.json", "w"), indent=1)
for r in out["p1"]: print(r)
for r in out["p2"]: print(r)
