"""How independent verifier errors bias the measured AUROC (binormal detector, errors independent of the score)."""
import numpy as np, json
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
rng = np.random.default_rng(11)
def d_for_auc(a): return np.sqrt(2) * norm.ppf(a)

def observed_auc(a_true, ppv, fn_rate, prev=0.125, n=400000):
    """Detector scores depend on TRUE label. Observed label = verifier output.
    ppv   = P(true reliable | observed reliable); fn_rate = P(true reliable | observed failure)."""
    y = rng.random(n) < prev
    s = rng.normal(0, 1, n) + y * d_for_auc(a_true)
    # Bayes-style: choose sens, fpr so that the observed PPV and NPV-error match.
    # observed positive rate q solves  prev = q*ppv + (1-q)*fn_rate
    q = (prev - fn_rate) / (ppv - fn_rate)
    sens = q * ppv / prev; fpr = q * (1 - ppv) / (1 - prev)
    yo = np.where(y, rng.random(n) < sens, rng.random(n) < fpr)
    return roc_auc_score(yo, s), sens, fpr, q

grid = np.linspace(0.55, 0.95, 9)
out = {"true_auc": grid.tolist(), "curves": {}}
scen = {"perfect labels (PPV 100%, FN 0%)": (1.0, 0.0), "PPV 95.7%, FN 1%": (0.957, 0.01),
        "pilot point estimate (PPV 87.5%, FN 2.2%)": (0.875, 0.022), "pilot CI worst case (PPV 69%, FN 4%)": (0.69, 0.04)}
for name, (ppv, fn) in scen.items():
    out["curves"][name] = [observed_auc(a, ppv, fn)[0] for a in grid]
    print(name, [round(x, 3) for x in out["curves"][name]])
# what the pilot's labels imply for a detector whose true AUROC is 0.78
for name, (ppv, fn) in scen.items():
    o, sens, fpr, q = observed_auc(0.78, ppv, fn); print(f"{name}: true 0.78 -> observed {o:.3f}  (sens {sens:.2f}, fpr {fpr:.3f}, obs-pos rate {q:.3f})")
# Spread at the pilot size: 200 questions, 25 true reliable, labels with the point-estimate error rates
def one_study(a, ppv, fn, n=200, npos=25):
    q = (npos / n - fn) / (ppv - fn); sens = q * ppv / (npos / n); fpr = q * (1 - ppv) / (1 - npos / n)
    y = np.zeros(n, bool); y[:npos] = True; s = rng.normal(0, 1, n) + y * d_for_auc(a)
    yo = np.where(y, rng.random(n) < sens, rng.random(n) < fpr)
    if yo.min() == yo.max(): return np.nan
    return roc_auc_score(yo, s)
sims = np.array([one_study(0.78, 0.875, 0.022) for _ in range(4000)]); sims = sims[~np.isnan(sims)]
out["pilot_size"] = {"mean_obs": float(sims.mean()), "q": [float(np.quantile(sims, .025)), float(np.quantile(sims, .975))]}
print("pilot-size, true .78, point-estimate noise:", out["pilot_size"])
json.dump(out, open("figs/label_noise.json", "w"), indent=1)
