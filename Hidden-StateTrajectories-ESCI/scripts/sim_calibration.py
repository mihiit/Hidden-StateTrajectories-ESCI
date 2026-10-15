#!/usr/bin/env python3
"""Does the paired-comparison procedure of the paper behave as claimed? A calibration simulation.

The confirmatory analysis scores two detectors by repeated stratified CV, averages each example's held-out scores over
repeats, and tests the AUROC difference with a paired DeLong test and a bootstrap over questions. Out-of-fold scores are
NOT independent draws of one classifier (training sets overlap), so such tests are known to be able to err.  Here we
measure how often each inference procedure rejects / excludes zero when the true gap is zero (type-I error), how often it
detects a true gap of about 0.05 (power), and -- the quantity behind the paper's headline -- how often the pre-specified
rule 'upper 95% bound < 0.05 => a gain of 0.05 is ruled out' fires when the truth is 0 (it should usually) and when the
truth IS a gain of 0.05 (it should almost never).

Design mirrors the study: n=900, 22.1% reliable, two correlated detectors (shared noise component), d=60 features each,
one fixed linear-Gaussian signal per detector, same classifier/grid/CV as rtm_protocol.  The only simplification is d (60,
not 342/1536) to make hundreds of replicates affordable.

    python scripts/sim_calibration.py --calibrate                       # find the signal strength giving a 0.05 gap
    python scripts/sim_calibration.py --reps 300 --scenario null --out sim_null.json
    python scripts/sim_calibration.py --reps 300 --scenario gap05 --m2 <value> --out sim_gap05.json
"""
import argparse, json, os, sys, time
import numpy as np
from joblib import Parallel, delayed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rl_common as C  # noqa: E402
import rtm_protocol as P  # noqa: E402

N, PREV, D, SHARED = 900, 0.221, 60, 0.5


def draw(rng, n, m1, m2, u1, u2, y=None):
    if y is None: y = np.zeros(n, int); y[rng.choice(n, int(round(PREV * n)), replace=False)] = 1
    G = rng.normal(size=(n, D))
    X1 = np.sqrt(SHARED) * G + np.sqrt(1 - SHARED) * rng.normal(size=(n, D)) + np.outer(y, m1 * u1)
    X2 = np.sqrt(SHARED) * G + np.sqrt(1 - SHARED) * rng.normal(size=(n, D)) + np.outer(y, m2 * u2)
    return X1, X2, y


def units(rng):
    u1, u2 = rng.normal(size=D), rng.normal(size=D); return u1 / np.linalg.norm(u1), u2 / np.linalg.norm(u2)


def one_rep(seed, m1, m2, a):
    rng = np.random.default_rng(seed); u1, u2 = units(rng); X1, X2, y = draw(rng, N, m1, m2, u1, u2)
    S1, F1 = C.cv_scores(X1, y, folds=5, repeats=a.repeats, seed=seed); S2, F2 = C.cv_scores(X2, y, folds=5, repeats=a.repeats, seed=seed)
    fa1, fa2 = C.fold_aucs(y, S1, F1), C.fold_aucs(y, S2, F2)
    c = C.contrast(y, S1, S2, a.boot, rng, 5, N * 0.8, N * 0.2, fold_a=fa1, fold_b=fa2)
    # population AUROC of the two detectors trained on a training-fold-sized sample (the target of the comparison)
    tr = rng.choice(N, 720, replace=False)
    Xa, Xb, yt = draw(rng, 20000, m1, m2, u1, u2)
    ma = P.make_model().fit(X1[tr], y[tr]); mb = P.make_model().fit(X2[tr], y[tr])
    c["pop_gap"] = float(C.fast_auc(yt, ma.decision_function(Xa)) - C.fast_auc(yt, mb.decision_function(Xb)))
    return c


def calibrate(a):
    rng = np.random.default_rng(1); out = {}
    for m2 in a.grid:
        gaps = []
        for r in range(a.reps):
            u1, u2 = units(rng); X1, X2, y = draw(rng, N, 1.0, m2, u1, u2); tr = rng.choice(N, 720, replace=False)
            Xa, Xb, yt = draw(rng, 20000, 1.0, m2, u1, u2)
            ga = C.fast_auc(yt, P.make_model().fit(X1[tr], y[tr]).decision_function(Xa)); gb = C.fast_auc(yt, P.make_model().fit(X2[tr], y[tr]).decision_function(Xb))
            gaps.append((ga, gb))
        g = np.array(gaps); out[m2] = (g[:, 0].mean(), g[:, 1].mean(), (g[:, 1] - g[:, 0]).mean()); print(f"m2={m2}: AUROC1 {out[m2][0]:.3f}  AUROC2 {out[m2][1]:.3f}  gap(2-1) {out[m2][2]:+.3f}", flush=True)


def summarise(reps, margin=0.05):
    # detector 1 = 'a' of the contrast; contrast diff = AUROC(1) - AUROC(2).  For the gap scenario detector 1 is the WEAKER one,
    # so the quantity of interest is AUROC(2) - AUROC(1) = -diff.  We report in terms of the contrast exactly as run.
    n = len(reps); r = {}
    ex0 = lambda ci: ci[0] > 0 or ci[1] < 0
    r["n_reps"] = n; r["mean_pop_gap_(1-2)"] = float(np.mean([x["pop_gap"] for x in reps])); r["mean_estimated_diff_pooled"] = float(np.mean([x["diff_pooled"] for x in reps]))
    r["reject_delong_p<0.05"] = float(np.mean([x["delong_p"] < .05 for x in reps]))
    r["ci_excludes_0_pooled_bootstrap"] = float(np.mean([ex0(x["ci_pooled"]) for x in reps]))
    r["ci_excludes_0_repmean_bootstrap"] = float(np.mean([ex0(x["ci_repmean"]) for x in reps]))
    r["ci_excludes_0_partition_bootstrap"] = float(np.mean([ex0(x["ci_partition"]) for x in reps]))
    r["reject_nadeau_bengio_p<0.05"] = float(np.mean([x["nadeau_bengio"]["p"] < .05 for x in reps]))
    pg = np.array([x["pop_gap"] for x in reps])
    cov = lambda key: float(np.mean([(x[key][0] <= g <= x[key][1]) for x, g in zip(reps, pg)]))
    r["coverage_of_population_gap_pooled_bootstrap"] = cov("ci_pooled"); r["coverage_repmean_bootstrap"] = cov("ci_repmean"); r["coverage_partition_bootstrap"] = cov("ci_partition")
    r["coverage_nadeau_bengio"] = float(np.mean([(x["nadeau_bengio"]["ci95"][0] <= g <= x["nadeau_bengio"]["ci95"][1]) for x, g in zip(reps, pg)]))
    # the paper's rule, applied to the contrast as AUROC(better-or-equal) - AUROC(other): 'upper bound < margin' rules out a gain of >= margin
    # here: 'gain' = AUROC(1) - AUROC(2) (detector 1 plays the role of the trajectory model A).
    r["rule_gain_ruled_out_pooled"] = float(np.mean([x["ci_pooled"][1] < margin for x in reps]))
    r["rule_gain_ruled_out_partition"] = float(np.mean([x["ci_partition"][1] < margin for x in reps]))
    r["rule_gain_ruled_out_nadeau_bengio"] = float(np.mean([x["nadeau_bengio"]["ci95"][1] < margin for x in reps]))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true"); ap.add_argument("--grid", type=float, nargs="*", default=[1.0, 1.15, 1.3, 1.45, 1.6])
    ap.add_argument("--reps", type=int, default=300); ap.add_argument("--repeats", type=int, default=5); ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--scenario", choices=["null", "gap05", "gap05_rev"], default="null"); ap.add_argument("--m1", type=float, default=1.0); ap.add_argument("--m2", type=float)
    ap.add_argument("--n-jobs", type=int, default=2); ap.add_argument("--out"); ap.add_argument("--seed0", type=int, default=1000)
    a = ap.parse_args()
    if a.calibrate: return calibrate(a)
    m1, m2 = a.m1, (a.m1 if a.scenario == "null" else a.m2)
    if a.scenario == "gap05_rev": m1, m2 = a.m2, a.m1          # detector 1 (the 'trajectory model') is the BETTER one: true gain of about 0.05
    t0 = time.time()
    reps = Parallel(n_jobs=a.n_jobs, verbose=0)(delayed(one_rep)(a.seed0 + i, m1, m2, a) for i in range(a.reps))
    s = summarise(reps); s["scenario"] = a.scenario; s["m1"] = m1; s["m2"] = m2; s["settings"] = dict(n=N, prevalence=PREV, d=D, repeats=a.repeats, boot=a.boot, shared_noise=SHARED); s["minutes"] = (time.time() - t0) / 60
    print(json.dumps(s, indent=1))
    if a.out: json.dump(dict(summary=s, reps=reps), open(a.out, "w"))


if __name__ == "__main__":
    main()
