#!/usr/bin/env python3
"""Length-adjusted evaluation on the real hidden states (the paper's most important missing experiment).

Question: does the H1 conclusion (trajectory features add nothing to a tuned static probe) and the H2 conclusion
(trajectory features carry signal beyond length) survive when response length is removed from the problem?
Proposition 1 says max-summaries of content-free features leak length; the length-only control used in the paper was
linear in log-length, so a non-monotone length effect or a length component shared by the families could remain.

Four complementary adjustments, all scored by the SAME pipeline as the confirmatory run (rl_common.cv_scores):
  resid    every feature is regressed on a cubic polynomial of standardised log-length, coefficients fitted on the
           TRAINING fold only; the classifier sees the residuals.  (families *_r)
  concat   the same length covariates are given to the classifier next to the features (F+len, A+len): checks that
           adding features to a flexible length baseline helps.
  strata   AUROC over reliable/failed pairs of the SAME length bin only (pair-weighted, bins = length quintiles, merged
           until every bin has >= 10 reliable and >= 10 failed).  Applied to the unadjusted out-of-fold scores, so it
           needs no refitting and cannot be moved by anything that is constant within a bin.
  nomax    A without the max-summaries (mean and last only), the summary that Proposition 1 shows to be length-dependent.
Control: len_flex = the cubic length polynomial alone (replaces the linear log-length control, which can miss a
non-monotone length effect).

DECISION RULES (fixed in analysis_plan_v2.md BEFORE this script is run on the real data; margin 0.05 as in H1):
  L1  H1 robust to length  : upper 95% bound of AUROC(A_r) - AUROC(F_r) < 0.05 AND of the within-stratum A - F < 0.05.
      If instead a lower bound > 0 and the point estimate >= 0.05 under either adjustment, the H1 conclusion is overturned.
  L2  Signal beyond length : lower 95% bound of AUROC(A_r) > 0.5 and of AUROC(A_r) - AUROC(len_flex) > 0
      and lower bound of the within-stratum AUROC(A) > 0.5.
  Anything else is reported as inconclusive. Results are reported in full whichever way they fall.

    python scripts/length_adjusted_eval.py --features cache/features.npz --labels annotation/labels.csv \
        --frozen annotation/FROZEN.json --out results_v2/ --n-jobs 4
"""
import argparse, json, os, sys
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rl_common as C  # noqa: E402
import rtm_protocol as P  # noqa: E402
from stats_robustness import load_cache  # noqa: E402

MARGIN = 0.05
BASE = ["F_final_state", "Llast_B4style", "L_layer_axis", "T_token_axis", "A_full_RTM"]


def pboot(y, ma, mb, strata, B, rng, f):
    """Paired bootstrap of f(a) - f(b) for score vectors ma, mb (questions resampled, strata fixed)."""
    n = len(y); d = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max(): continue
        d.append(f(y[i], ma[i], strata[i]) - f(y[i], mb[i], strata[i]))
    return np.quantile(d, [.025, .975]).tolist()


def pooled_f(y, s, st): return C.fast_auc(y, s)


def strat_f(y, s, st): return C.strat_auc(y, s, st)


def run(B, y, a):
    rng = np.random.default_rng(a.seed); ntok = B["ntok"].ravel(); Z = C.length_covariates(ntok, 3)
    strata, edges = C.length_strata(ntok, y, a.strata, a.min_pos)
    out = {"n": int(len(y)), "n_reliable": int(y.sum()), "length": dict(median_tokens=float(np.median(ntok)), strata_edges=[float(e) for e in edges],
           strata_sizes=[int((strata == s).sum()) for s in np.unique(strata)], strata_reliable=[int(((strata == s) & (y == 1)).sum()) for s in np.unique(strata)],
           spearman_len_label=float(spearmanr(ntok, y).correlation)), "detectors": {}, "contrasts": {}, "stratified": {}}
    sc = {}
    def add(name, X, mode):
        S, _ = C.cached_cv(a.cache, name, X, y, Z, mode, a.folds, a.repeats, a.seed, a.n_jobs)
        m = S.mean(0); sc[name] = m; lo, hi = C.boot_ci(y, m, a.boot, rng)
        out["detectors"][name] = dict(auroc=float(C.fast_auc(y, m)), ci95=[lo, hi], spearman_score_vs_loglen=float(spearmanr(m, np.log(ntok)).correlation),
                                      within_stratum_auroc=float(C.strat_auc(y, m, strata)), n_features=int(X.shape[1]))
        d = out["detectors"][name]
        print(f"{name:18s} AUROC {d['auroc']:.3f} [{lo:.3f},{hi:.3f}]  within-length-stratum {d['within_stratum_auroc']:.3f}  rho(score,log len) {d['spearman_score_vs_loglen']:+.2f}", flush=True)
    for k in BASE + ["len_only"]: add(k, C.assemble(B, k), "plain")                       # unadjusted (reproduces confirmatory)
    for k in BASE: add(k + "_resid", C.assemble(B, k), "resid")                          # residualised on cubic log-length
    add("len_flex", Z, "plain")                                                          # flexible length-only control
    add("F+len", C.assemble(B, "F_final_state"), "concat"); add("A+len", C.assemble(B, "A_full_RTM"), "concat")
    add("A_nomax", C.assemble(B, "A_nomax"), "plain"); add("A_nomax_resid", C.assemble(B, "A_nomax"), "resid")

    def contr(name, a_, b_):
        for kind, f in (("pooled", pooled_f), ("within_stratum", strat_f)):
            d = f(y, sc[a_], strata) - f(y, sc[b_], strata); ci = pboot(y, sc[a_], sc[b_], strata, a.boot, rng, f)
            out["contrasts"].setdefault(name, {})[kind] = dict(diff=float(d), ci95=[float(ci[0]), float(ci[1])])
        p = P.delong_paired(y, sc[a_], sc[b_])[1]; out["contrasts"][name]["delong_p"] = float(p)
        c = out["contrasts"][name]
        print(f"{name:28s} pooled {c['pooled']['diff']:+.3f} [{c['pooled']['ci95'][0]:+.3f},{c['pooled']['ci95'][1]:+.3f}]  within-stratum {c['within_stratum']['diff']:+.3f} [{c['within_stratum']['ci95'][0]:+.3f},{c['within_stratum']['ci95'][1]:+.3f}]")
    print()
    contr("H1: A - F (unadjusted)", "A_full_RTM", "F_final_state")
    contr("H1': A_r - F_r (residualised)", "A_full_RTM_resid", "F_final_state_resid")
    contr("H1'': A+len - F+len", "A+len", "F+len")
    contr("A_r - L_r", "A_full_RTM_resid", "L_layer_axis_resid"); contr("A_r - T_r", "A_full_RTM_resid", "T_token_axis_resid")
    contr("H2: A - len_only", "A_full_RTM", "len_only"); contr("H2': A_r - len_flex", "A_full_RTM_resid", "len_flex")
    contr("F_r - len_flex", "F_final_state_resid", "len_flex"); contr("A - A_nomax", "A_full_RTM", "A_nomax")
    contr("F - F_r (length-linked share of F)", "F_final_state", "F_final_state_resid"); contr("A - A_r (length-linked share of A)", "A_full_RTM", "A_full_RTM_resid")

    # per-stratum table (descriptive)
    tab = []
    for s in np.unique(strata):
        m = strata == s; row = dict(stratum=int(s), n=int(m.sum()), reliable=int(y[m].sum()), token_range=[int(ntok[m].min()), int(ntok[m].max())])
        for k in ("F_final_state", "A_full_RTM", "F_final_state_resid", "A_full_RTM_resid"): row[k] = float(C.fast_auc(y[m], sc[k][m]))
        tab.append(row)
    out["stratified"]["per_stratum_auroc"] = tab

    # pre-specified decisions
    c1, c2, c3 = out["contrasts"]["H1: A - F (unadjusted)"], out["contrasts"]["H1': A_r - F_r (residualised)"], out["contrasts"]["H2': A_r - len_flex"]
    up = max(c2["pooled"]["ci95"][1], c2["within_stratum"]["ci95"][1]); lo = min(c2["pooled"]["ci95"][0], c2["within_stratum"]["ci95"][0])
    if up < MARGIN: L1 = "H1 conclusion robust to length (upper bounds < %.2f)" % MARGIN
    elif c2["pooled"]["ci95"][0] > 0 and c2["pooled"]["diff"] >= MARGIN or c2["within_stratum"]["ci95"][0] > 0 and c2["within_stratum"]["diff"] >= MARGIN: L1 = "H1 conclusion OVERTURNED by length adjustment"
    else: L1 = "inconclusive"
    Ar = out["detectors"]["A_full_RTM_resid"]
    L2 = ("signal beyond length confirmed" if Ar["ci95"][0] > 0.5 and c3["pooled"]["ci95"][0] > 0 and
          C.strat_auc(y, sc["A_full_RTM"], strata) > 0.5 and pboot_lo(y, sc["A_full_RTM"], strata, a.boot, rng) > 0.5 else "not confirmed / inconclusive")
    out["decisions"] = dict(L1=L1, L2=L2)
    print("\nL1:", L1); print("L2:", L2)
    return out


def pboot_lo(y, s, strata, B, rng):
    n = len(y); d = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max(): continue
        d.append(C.strat_auc(y[i], s[i], strata[i]))
    return float(np.quantile(d, .025))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True); ap.add_argument("--labels", required=True); ap.add_argument("--frozen")
    ap.add_argument("--allow-unfrozen", action="store_true"); ap.add_argument("--out", default="results_v2")
    ap.add_argument("--folds", type=int, default=5); ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--boot", type=int, default=2000); ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--strata", type=int, default=5); ap.add_argument("--min-pos", type=int, default=10); ap.add_argument("--n-jobs", type=int, default=1)
    a = ap.parse_args()
    if a.frozen: P.check_frozen(a.labels, a.frozen)
    elif not a.allow_unfrozen: ap.error("--frozen FROZEN.json is required (or --allow-unfrozen for exploratory use)")
    B, y = load_cache(a.features, a.labels); a.cache = os.path.join(a.out, "cache")
    out = run(B, y, a)
    os.makedirs(a.out, exist_ok=True); json.dump(out, open(os.path.join(a.out, "length_adjusted.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
