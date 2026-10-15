#!/usr/bin/env python3
"""
Controlled evaluation protocol for hidden-state hallucination detectors (Reliability//Lab).

Addresses reviewer points 1, 2 and 5:
  * every detector is scored out-of-fold on ALL labelled examples (repeated stratified K-fold),
    so AUROC uses all reliable examples (25 of 200 today) rather than the 7-8 in one 30% test fold;
  * the static probe gets the same standardisation + L2 regularisation + inner-CV tuning as RTM;
  * feature families are compared under one classifier, one preprocessing, one split and one tuning budget;
  * all differences are paired (same examples), with bootstrap CIs, DeLong p-values and Holm correction.

Input  (--feats-dir): one <id>.npz per generation containing  hidden  with shape (layers, tokens, dim)
                      (hidden states of the generated tokens, one row per transformer layer).
       (--labels)   : CSV with columns  id,label   (label 1 = reliable answer, 0 = failure), FINAL, detector-blind labels.
Output (--out)      : results.json and results.csv.

    python rtm_protocol.py --feats-dir acts/ --labels annotation/labels.csv --frozen annotation/FROZEN.json --out results/
    python rtm_protocol.py --selftest            # synthetic sanity check, no data needed
"""
import argparse, json, os, sys, warnings
import numpy as np
from scipy.stats import norm, mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
WINDOW = 5
CGRID = np.logspace(-4, 1, 6)

# ----------------------------------------------------------------------------- features
def _summ(x):                      # x: (L, T) -> mean, max, last per layer, concatenated
    return np.concatenate([x.mean(1), x.max(1), x[:, -1]])

def raw_arrays(h):
    """h: (L, T, d). Returns V (L,T-1), D (L-1,T), theta (L-1,T), var (L,T-1)."""
    h = h.astype(np.float64)
    V = np.linalg.norm(h[:, 1:] - h[:, :-1], axis=2)
    D = np.linalg.norm(h[1:] - h[:-1], axis=2)
    num = (h[1:] * h[:-1]).sum(2)
    den = np.linalg.norm(h[1:], axis=2) * np.linalg.norm(h[:-1], axis=2) + 1e-12
    th = np.arccos(np.clip(num / den, -1, 1))
    Tm = V.shape[1]
    var = np.stack([[V[l, max(0, t - WINDOW + 1):t + 1].var() for t in range(Tm)] for l in range(V.shape[0])])
    return V, D, th, var

def families(h):
    V, D, th, var = raw_arrays(h)
    fam = {
        "F_final_state": h[-1, -1].astype(np.float64),                    # static probe (B2-style)
        "Llast_B4style": np.concatenate([D[:, -1], th[:, -1]]),           # cross-layer deltas at final token only (B4-style)
        "L_layer_axis": np.concatenate([_summ(D), _summ(th)]),            # layer-axis features over all tokens
        "T_token_axis": np.concatenate([_summ(V), _summ(var)]),           # token-axis features over all layers
    }
    fam["A_full_RTM"] = np.concatenate([fam["L_layer_axis"], fam["T_token_axis"]])
    fam["len_only"] = np.array([np.log(h.shape[1])])                      # length confound control
    return fam

# ----------------------------------------------------------------------------- models / CV
def make_model():
    pipe = make_pipeline(StandardScaler(), LogisticRegression(penalty="l2", class_weight="balanced", max_iter=2000))
    return GridSearchCV(pipe, {"logisticregression__C": CGRID}, scoring="roc_auc",
                        cv=StratifiedKFold(3, shuffle=True, random_state=0), n_jobs=1, refit=True)

def oof_scores(X, y, n_splits, n_repeats, seed):
    """Mean out-of-fold decision score per example over repeats; tuning happens inside each training fold only."""
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    S, K = np.zeros(len(y)), np.zeros(len(y))
    for tr, te in rskf.split(X, y):
        m = make_model().fit(X[tr], y[tr])
        S[te] += m.decision_function(X[te]); K[te] += 1
    return S / K

# ----------------------------------------------------------------------------- statistics
def auc(y, s): return roc_auc_score(y, s)

def boot_ci(y, s, B, rng):
    n = len(y); out = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max(): continue
        out.append(roc_auc_score(y[i], s[i]))
    return np.quantile(out, [.025, .975]).tolist()

def delong_paired(y, s1, s2):
    pos, neg = y == 1, y == 0
    def comps(s):
        p, q = s[pos], s[neg]
        psi = (p[:, None] > q[None, :]) + .5 * (p[:, None] == q[None, :])
        return psi.mean(1), psi.mean(0), psi.mean()
    a10, a01, A = comps(s1); b10, b01, B = comps(s2)
    S10, S01 = np.cov(np.vstack([a10, b10])), np.cov(np.vstack([a01, b01]))
    S = S10 / pos.sum() + S01 / neg.sum()
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    z = (A - B) / np.sqrt(var) if var > 0 else 0.0
    return A - B, 2 * (1 - norm.cdf(abs(z)))

def paired_boot(y, s1, s2, B, rng):
    n = len(y); d = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max(): continue
        d.append(roc_auc_score(y[i], s1[i]) - roc_auc_score(y[i], s2[i]))
    return np.quantile(d, [.025, .975]).tolist()

def holm(p):
    p = np.asarray(p); o = np.argsort(p); out = np.empty(len(p)); run = 0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i]); out[i] = min(1, run)
    return out

# ----------------------------------------------------------------------------- driver
def run(feats, y, args, outdir):
    rng = np.random.default_rng(args.seed)
    names = list(feats[0].keys()); res = {"n": int(len(y)), "n_reliable": int(y.sum()), "detectors": {}}
    scores = {}
    for k in names:
        X = np.vstack([f[k] for f in feats])
        scores[k] = oof_scores(X, y, args.folds, args.repeats, args.seed)
        a = auc(y, scores[k]); lo, hi = boot_ci(y, scores[k], args.boot, rng)
        p_chance = mannwhitneyu(scores[k][y == 1], scores[k][y == 0], alternative="greater").pvalue
        res["detectors"][k] = dict(auroc=a, ci95=[lo, hi], p_vs_chance=p_chance, n_features=int(X.shape[1]))
        print(f"{k:16s} AUROC {a:.3f}  [{lo:.3f}, {hi:.3f}]  p(vs 0.5)={p_chance:.4f}  d={X.shape[1]}", flush=True)
    ref = args.primary
    others = [k for k in names if k != ref]
    rows = []
    for k in others:
        d, p = delong_paired(y, scores[ref], scores[k]); lo, hi = paired_boot(y, scores[ref], scores[k], args.boot, rng)
        rows.append(dict(comparison=f"{ref} - {k}", diff=d, ci95=[lo, hi], p_raw=p))
    ph = holm([r["p_raw"] for r in rows])
    for r, q in zip(rows, ph): r["p_holm"] = float(q)
    res["paired_vs_" + ref] = rows
    os.makedirs(outdir, exist_ok=True)
    json.dump(res, open(os.path.join(outdir, "results.json"), "w", encoding="utf-8"), indent=1)
    with open(os.path.join(outdir, "results.csv"), "w", encoding="utf-8") as f:
        f.write("detector,auroc,ci_lo,ci_hi,p_vs_chance,n_features\n")
        for k, v in res["detectors"].items(): f.write(f"{k},{v['auroc']:.4f},{v['ci95'][0]:.4f},{v['ci95'][1]:.4f},{v['p_vs_chance']:.5f},{v['n_features']}\n")
    np.save(os.path.join(outdir, "oof_scores.npy"), np.vstack([scores[k] for k in names]))
    return res

def check_frozen(labels_csv, frozen_json):
    """Refuse to run if the label file differs from the hash recorded when the labels were frozen."""
    import hashlib
    meta = json.load(open(frozen_json, encoding="utf-8")); h = hashlib.sha256(open(labels_csv, "rb").read()).hexdigest()
    if h != meta["sha256"]: sys.exit(f"labels file hash {h[:12]} does not match FROZEN.json {meta['sha256'][:12]}: labels changed after freezing")
    print(f"labels verified against FROZEN.json (n={meta['n']}, reliable={meta['n_reliable']}, kappa={meta['kappa_before_adjudication']:.3f})")
    return meta

def load(feats_dir, labels_csv):
    import csv
    rows = list(csv.DictReader(open(labels_csv, encoding="utf-8-sig")))
    feats, y = [], []
    for r in rows:
        h = np.load(os.path.join(feats_dir, r["id"] + ".npz"))["hidden"]
        feats.append(families(h)); y.append(int(r["label"]))
    return feats, np.array(y)

def selftest(args):
    rng = np.random.default_rng(0); feats, y = [], []
    for i in range(120):
        lab = int(i % 8 == 0); T = int(rng.integers(12, 40)); L, d = 8, 32
        h = np.cumsum(rng.normal(size=(L, T, d)) * (0.6 if lab else 1.0), axis=1) * .3 + rng.normal(size=(L, 1, d))
        feats.append(families(h)); y.append(lab)
    y = np.array(y); args.repeats, args.boot = 3, 200
    r = run(feats, y, args, args.out)
    assert 0.0 <= r["detectors"]["A_full_RTM"]["auroc"] <= 1.0 and len(r["paired_vs_" + args.primary]) == 5
    print("selftest OK (synthetic data: numbers carry no scientific meaning)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats-dir"); ap.add_argument("--labels"); ap.add_argument("--out", default="results")
    ap.add_argument("--folds", type=int, default=5); ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--boot", type=int, default=2000); ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--primary", default="A_full_RTM"); ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--frozen", help="FROZEN.json from annotation_tools.py freeze; labels are verified against it")
    ap.add_argument("--allow-unfrozen", action="store_true", help="run on labels that were not frozen (exploratory only)")
    a = ap.parse_args()
    if a.selftest: selftest(a); sys.exit(0)
    if not (a.feats_dir and a.labels): ap.error("--feats-dir and --labels are required (or use --selftest)")
    if a.frozen: check_frozen(a.labels, a.frozen)
    elif not a.allow_unfrozen: ap.error("--frozen FROZEN.json is required for confirmatory runs (or pass --allow-unfrozen for exploratory use)")
    else: print("WARNING: labels not frozen; results are exploratory")
    f, y = load(a.feats_dir, a.labels); run(f, y, a, a.out)
