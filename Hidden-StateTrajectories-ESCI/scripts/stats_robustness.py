#!/usr/bin/env python3
"""Robustness of the confirmatory inference (reviewer point: 'clarify the repeated CV and paired-test procedure').

What the pre-specified analysis (rtm_protocol.py) does, exactly:
  1. For each family, 10 x stratified 5-fold CV (RepeatedStratifiedKFold, random_state=17). The penalty C is chosen by an
     inner 3-fold CV on the TRAINING fold only. Every example receives one held-out score per repeat.
  2. The example's score is the mean of its 10 held-out scores; AUROC is computed once on these 900 mean scores.
  3. 95% CI: percentile bootstrap over questions (2000), mean scores held fixed. Paired contrasts: DeLong test and paired
     bootstrap on the same mean scores. Holm over the contrasts.
Known limits of that procedure (all reported, none hidden): the bootstrap ignores partition variation and the overlap of
training sets across folds; DeLong treats out-of-fold scores as independent draws of one fixed classifier.

This script (a) reproduces step 1-3 exactly and checks the numbers against confirmatory_results.json, and
(b) re-tests the H1 / H2 / H3 contrasts with three further procedures:
    repmean    - mean over repeats of per-repeat AUROC differences, bootstrap over questions
    partition  - bootstrap that also resamples which repeat is used (partition variation enters the interval)
    nadeau_bengio - corrected resampled t-test on the 50 fold-level AUROC differences (accounts for training overlap)
and reports whether the H1 decision ('the upper bound is below 0.05') survives every one of them.
Multiplicity: H1 and H2 are single pre-specified tests; H3 is Holm-corrected over its three contrasts (as pre-specified);
the five-contrast Holm of the original run is also shown.

    python scripts/stats_robustness.py --features cache/features.npz --labels annotation/labels.csv \
        --frozen annotation/FROZEN.json --reference confirmatory_results.json --out results_v2/ --n-jobs 4
"""
import argparse, json, os, sys
import numpy as np
from scipy.stats import mannwhitneyu
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rl_common as C  # noqa: E402
import rtm_protocol as P  # noqa: E402

MARGIN = 0.05
REFNAME = {"F_final_state": "F_final_state"}


def load_cache(path, labels_csv):
    z = np.load(path, allow_pickle=False); ids, y = C.load_labels(labels_csv)
    if list(z["ids"]) != ids: sys.exit("feature cache and labels file do not match (ids/order)")
    if not np.array_equal(z["y"], y): print("NOTE: labels differ from those the cache was built with: this is a label-sensitivity run, treat as exploratory")
    return {k: z[k] for k in z.files if k not in ("ids", "y")}, y


def run(B, y, a):
    rng = np.random.default_rng(a.seed); n = len(y)
    n_te = n / a.folds; n_tr = n - n_te
    S, Fo, FA = {}, {}, {}
    out = {"n": int(n), "n_reliable": int(y.sum()), "settings": dict(folds=a.folds, repeats=a.repeats, seed=a.seed, boot=a.boot), "detectors": {}}
    for k in C.PRIMARY:
        X = C.assemble(B, k)
        S[k], Fo[k] = C.cached_cv(a.cache, k, X, y, None, "plain", a.folds, a.repeats, a.seed, a.n_jobs)
        FA[k] = C.fold_aucs(y, S[k], Fo[k]); m = S[k].mean(0)
        auc = C.fast_auc(y, m); lo, hi = C.boot_ci(y, m, a.boot, rng)
        per_rep = np.array([C.fast_auc(y, S[k][r]) for r in range(a.repeats)])
        out["detectors"][k] = dict(auroc_pooled=float(auc), ci95=[lo, hi], p_vs_chance=float(mannwhitneyu(m[y == 1], m[y == 0], alternative="greater").pvalue),
                                   auroc_per_repeat_mean=float(per_rep.mean()), auroc_per_repeat_sd=float(per_rep.std(ddof=1)),
                                   auroc_per_repeat_range=[float(per_rep.min()), float(per_rep.max())],
                                   fold_auroc_mean=float(FA[k].mean()), fold_auroc_sd=float(FA[k].std(ddof=1)), n_features=int(X.shape[1]))
        d = out["detectors"][k]
        print(f"{k:15s} pooled {auc:.3f} [{lo:.3f},{hi:.3f}] | per-repeat {d['auroc_per_repeat_mean']:.3f} (sd {d['auroc_per_repeat_sd']:.3f}) | fold mean {d['fold_auroc_mean']:.3f} (sd {d['fold_auroc_sd']:.3f})", flush=True)
    ref = "A_full_RTM"; out["contrasts"] = {}
    for k in C.PRIMARY:
        if k == ref: continue
        c = C.contrast(y, S[ref], S[k], a.boot, rng, a.folds, n_tr, n_te, fold_a=FA[ref], fold_b=FA[k])
        out["contrasts"][k] = c
    # multiplicity as pre-specified
    h3 = ["Llast_B4style", "L_layer_axis", "T_token_axis"]
    def fam_holm(key):
        names = h3; ps = [out["contrasts"][k][key] if key == "delong_p" else out["contrasts"][k]["nadeau_bengio"]["p"] for k in names]
        return dict(zip(names, map(float, C.holm(ps))))
    allp = [out["contrasts"][k]["delong_p"] for k in out["contrasts"]]
    out["holm"] = {"H3_delong_3_tests": fam_holm("delong_p"), "H3_nadeau_bengio_3_tests": fam_holm("nb"),
                   "all5_delong_as_originally_run": dict(zip(out["contrasts"], map(float, C.holm(allp))))}
    # decision for H1 under every inference variant
    h1 = out["contrasts"]["F_final_state"]
    variants = {"pooled_bootstrap(pre-specified)": (h1["diff_pooled"], h1["ci_pooled"]), "repmean_bootstrap": (h1["diff_repmean"], h1["ci_repmean"]),
                "partition_bootstrap": (h1["diff_repmean"], h1["ci_partition"]), "nadeau_bengio": (h1["nadeau_bengio"]["diff"], h1["nadeau_bengio"]["ci95"])}
    out["H1_decision"] = {}
    for v, (d, ci) in variants.items():
        verdict = ("supported" if ci[0] > 0 and d >= MARGIN else "gain ruled out (upper bound < %.2f)" % MARGIN if ci[1] < MARGIN else "inconclusive")
        out["H1_decision"][v] = dict(diff=float(d), ci95=[float(ci[0]), float(ci[1])], verdict=verdict)
    out["H1_survives_all_variants"] = all(v["verdict"].startswith("gain ruled out") for v in out["H1_decision"].values())
    out["oof_means"] = {k: S[k].mean(0).tolist() for k in S} if a.save_scores else None
    return out


def compare_reference(out, ref_json, tol):
    ref = json.load(open(ref_json, encoding="utf-8")); rows = []
    for k, v in ref["detectors"].items():
        mine = out["detectors"][k]["auroc_pooled"]; rows.append((k, v["auroc"], mine, abs(v["auroc"] - mine)))
    worst = max(r[3] for r in rows)
    print("\nreproduction of the confirmatory run (published AUROC vs recomputed):")
    for k, a, b, d in rows: print(f"  {k:15s} {a:.3f} vs {b:.3f}  |diff| {d:.4f}")
    ok = worst <= tol
    print("REPRODUCTION", "OK" if ok else f"MISMATCH (max |diff| {worst:.4f} > {tol}); do not trust the extra analyses until this is explained")
    out["reproduction"] = dict(max_abs_diff=float(worst), tolerance=tol, ok=bool(ok), rows=[dict(detector=k, published=a, recomputed=b) for k, a, b, d in rows])
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True); ap.add_argument("--labels", required=True); ap.add_argument("--frozen")
    ap.add_argument("--allow-unfrozen", action="store_true"); ap.add_argument("--reference", help="confirmatory_results.json to reproduce")
    ap.add_argument("--tol", type=float, default=0.003); ap.add_argument("--out", default="results_v2")
    ap.add_argument("--folds", type=int, default=5); ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--boot", type=int, default=2000); ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--n-jobs", type=int, default=1); ap.add_argument("--save-scores", action="store_true")
    a = ap.parse_args()
    if a.frozen: P.check_frozen(a.labels, a.frozen)
    elif not a.allow_unfrozen: ap.error("--frozen FROZEN.json is required (or --allow-unfrozen for exploratory use)")
    B, y = load_cache(a.features, a.labels); a.cache = os.path.join(a.out, "cache")
    out = run(B, y, a)
    if a.reference: compare_reference(out, a.reference, a.tol)
    print("\nH1 decision under each inference procedure:")
    for v, r in out["H1_decision"].items(): print(f"  {v:34s} diff {r['diff']:+.3f}  CI [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]  -> {r['verdict']}")
    print("H1 'gain ruled out' survives every procedure:", out["H1_survives_all_variants"])
    os.makedirs(a.out, exist_ok=True); json.dump(out, open(os.path.join(a.out, "stats_robustness.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
