#!/usr/bin/env python3
"""End-to-end test of the v2 analysis chain on synthetic hidden states (numbers carry no scientific meaning; what is
checked is that the machinery behaves as designed):

  T1  blocks/assemble rebuild rtm_protocol.families exactly.
  T2  cv_scores(...).mean(0) equals rtm_protocol.oof_scores exactly (same splits, model, grid) -> the v2 numbers are the
      confirmatory numbers, not a re-implementation.
  T3  fast_auc equals sklearn roc_auc_score; strat_auc with one stratum equals AUROC.
  T4  NULL-LENGTH world: labels depend only on length, features carry no label information.  The unadjusted full-trajectory
      detector must beat chance (length leakage, Proposition 1) while the residualised detector and the within-stratum AUROC
      must not.
  T5  SIGNAL world: labels depend on a content feature independent of length.  The residualised static probe and the
      within-stratum AUROC must stay clearly above chance (the adjustment must not destroy real signal).
  T6  featurize.py + stats_robustness.py + length_adjusted_eval.py run end-to-end on files and produce JSON.
"""
import json, os, subprocess, sys, tempfile
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); SC = os.path.dirname(HERE)
sys.path.insert(0, SC)
import rl_common as C  # noqa: E402
import rtm_protocol as P  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402


def make_world(root, world, n=240, seed=0, L=6, d=12):
    rng = np.random.default_rng(seed); os.makedirs(root, exist_ok=True); rows = []
    # reliable answers are shorter; length-label AUROC about 0.75
    y = (rng.random(n) < 0.25).astype(int)
    T = np.clip(np.exp(rng.normal(3.6, 0.5, n) - 0.6 * y).astype(int), 6, 120)
    if world == "signal":                       # label information is in the content, length is independent of it
        T = np.clip(np.exp(rng.normal(3.6, 0.5, n)).astype(int), 6, 120)
    direction = rng.normal(size=d); direction /= np.linalg.norm(direction)
    for i in range(n):
        h = rng.normal(size=(L, T[i], d))
        if world == "signal": h[-1, -1] += 2.5 * y[i] * direction
        np.savez_compressed(os.path.join(root, f"q{i:05d}.npz"), hidden=h.astype(np.float16))
        rows.append((f"q{i:05d}", int(y[i])))
    with open(os.path.join(root, "labels.csv"), "w") as f:
        f.write("id,label\n"); [f.write(f"{a},{b}\n") for a, b in rows]
    return os.path.join(root, "labels.csv")


def main():
    tmp = tempfile.mkdtemp(prefix="rl_test_")
    # ---- T1, T2, T3 on a small world
    lab = make_world(os.path.join(tmp, "null"), "null_length", n=160, seed=1)
    ids, y = C.load_labels(lab); hs = [np.load(os.path.join(tmp, "null", i + ".npz"))["hidden"] for i in ids]
    Bl = [C.blocks(h) for h in hs]; B = {k: np.vstack([b[k] for b in Bl]) for k in Bl[0]}
    for j in (0, 7, 33):
        fam = P.families(hs[j])
        for k in C.PRIMARY: assert np.allclose(C.assemble({kk: v[j:j + 1] for kk, v in B.items()}, k)[0], fam[k]), (j, k)
    print("T1 OK: blocks/assemble == rtm_protocol.families")
    X = C.assemble(B, "L_layer_axis"); S, Fo = C.cv_scores(X, y, folds=5, repeats=2, seed=17)
    ref = P.oof_scores(X, y, 5, 2, 17)
    assert np.allclose(S.mean(0), ref, atol=1e-9), np.abs(S.mean(0) - ref).max()
    print("T2 OK: cv_scores mean == rtm_protocol.oof_scores (max |diff| %.1e)" % np.abs(S.mean(0) - ref).max())
    s = np.random.default_rng(3).normal(size=len(y)); s[:5] = 1.0
    assert abs(C.fast_auc(y, s) - roc_auc_score(y, s)) < 1e-12 and abs(C.strat_auc(y, s, np.zeros(len(y), int)) - roc_auc_score(y, s)) < 1e-12
    print("T3 OK: fast_auc/strat_auc == sklearn")

    # ---- T4 / T5: run the real scripts on files
    res = {}
    for world, seed in (("null_length", 11), ("signal", 12)):
        root = os.path.join(tmp, world); lab = make_world(root, world, n=300, seed=seed)
        feat = os.path.join(tmp, world + "_feat.npz"); outd = os.path.join(tmp, world + "_out")
        run = lambda *a: subprocess.run([sys.executable, os.path.join(SC, a[0])] + list(a[1:]), check=True, capture_output=True, text=True)
        run("featurize.py", "--feats-dir", root, "--labels", lab, "--out", feat, "--n-jobs", "2")
        r1 = run("stats_robustness.py", "--features", feat, "--labels", lab, "--allow-unfrozen", "--out", outd, "--repeats", "2", "--boot", "150", "--n-jobs", "2")
        r2 = run("length_adjusted_eval.py", "--features", feat, "--labels", lab, "--allow-unfrozen", "--out", outd, "--repeats", "2", "--boot", "150", "--n-jobs", "2")
        sr = json.load(open(os.path.join(outd, "stats_robustness.json"))); la = json.load(open(os.path.join(outd, "length_adjusted.json")))
        res[world] = (sr, la); print(f"\n[{world}] stats_robustness:\n{r1.stdout}\n[{world}] length_adjusted:\n{r2.stdout}")
    sr, la = res["null_length"]; d = la["detectors"]
    assert d["A_full_RTM"]["auroc"] > 0.60, "null-length world: unadjusted trajectory features should show length leakage"
    assert d["A_full_RTM_resid"]["auroc"] < 0.62 and d["A_full_RTM_resid"]["ci95"][0] < 0.5 + 0.02 or d["A_full_RTM_resid"]["auroc"] < 0.58
    assert d["A_full_RTM"]["within_stratum_auroc"] < d["A_full_RTM"]["auroc"] - 0.03, "within-stratum AUROC must remove the length leakage"
    print("T4 OK: length leakage shows unadjusted (%.3f) and disappears after residualisation (%.3f) / within strata (%.3f)" %
          (d["A_full_RTM"]["auroc"], d["A_full_RTM_resid"]["auroc"], d["A_full_RTM"]["within_stratum_auroc"]))
    sr, la = res["signal"]; d = la["detectors"]
    assert d["F_final_state_resid"]["auroc"] > 0.80 and d["F_final_state"]["within_stratum_auroc"] > 0.80, "adjustments must not destroy real content signal"
    print("T5 OK: real signal survives (F_r %.3f, F within-stratum %.3f)" % (d["F_final_state_resid"]["auroc"], d["F_final_state"]["within_stratum_auroc"]))
    assert "H1_decision" in sr and "decisions" in la
    print("T6 OK: scripts run end-to-end and write JSON")
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
