#!/usr/bin/env python3
"""Shared machinery for the v2 analyses (robust statistics, length-adjusted evaluation).

Design rule: everything that decides a score (features, classifier, grid, folds, seed) is imported from or identical to
rtm_protocol.py, so that the primary numbers of the confirmatory run are reproduced exactly and every extra analysis is
scored by the same pipeline.  Nothing here looks at labels except through the cross-validation loop.
"""
import hashlib, json, os, sys
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import norm, rankdata, t as tdist
from sklearn.model_selection import RepeatedStratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rtm_protocol as P  # noqa: E402

PRIMARY = ["F_final_state", "Llast_B4style", "L_layer_axis", "T_token_axis", "A_full_RTM", "len_only"]


# ----------------------------------------------------------------------------- features
def blocks(h):
    """h: (layers, tokens, d). Returns the building blocks of every feature family (1-D arrays) plus token count.
    Summary order matches rtm_protocol._summ: mean, max, last."""
    V, D, th, var = P.raw_arrays(h)
    b = {"F": h[-1, -1].astype(np.float64), "ntok": np.array([float(h.shape[1])])}
    for name, arr in (("D", D), ("th", th), ("V", V), ("var", var)):
        b[name + "_mean"], b[name + "_max"], b[name + "_last"] = arr.mean(1), arr.max(1), arr[:, -1]
    return b


def assemble(B, fam):
    """B: dict block-name -> (n, k) array. Returns the (n, d) design matrix of a feature family."""
    c = lambda *ks: np.hstack([B[k] for k in ks])
    L = ("D_mean", "D_max", "D_last", "th_mean", "th_max", "th_last")
    T = ("V_mean", "V_max", "V_last", "var_mean", "var_max", "var_last")
    if fam == "F_final_state": return B["F"]
    if fam == "Llast_B4style": return c("D_last", "th_last")
    if fam == "L_layer_axis": return c(*L)
    if fam == "T_token_axis": return c(*T)
    if fam == "A_full_RTM": return c(*L, *T)
    if fam == "A_nomax": return c("D_mean", "D_last", "th_mean", "th_last", "V_mean", "V_last", "var_mean", "var_last")
    if fam == "len_only": return np.log(B["ntok"])
    raise KeyError(fam)


def length_covariates(ntok, degree=3):
    """Label-free length covariates: polynomial of standardised log token count (flexible, not only linear)."""
    z = np.log(np.asarray(ntok, float).ravel()); z = (z - z.mean()) / (z.std() + 1e-12)
    return np.column_stack([z ** k for k in range(1, degree + 1)])


# ----------------------------------------------------------------------------- repeated-CV engine
def _fit_split(X, y, Z, mode, tr, te):
    Xtr, Xte = X[tr], X[te]
    if mode == "resid":     # remove the length-explained part; regression fitted on the TRAINING fold only
        Ztr = np.column_stack([np.ones(len(tr)), Z[tr]])
        coef = np.linalg.lstsq(Ztr, Xtr, rcond=None)[0]
        Xtr = Xtr - Ztr @ coef; Xte = Xte - np.column_stack([np.ones(len(te)), Z[te]]) @ coef
    elif mode == "concat":  # length covariates handed to the classifier as extra columns
        Xtr = np.column_stack([Xtr, Z[tr]]); Xte = np.column_stack([Xte, Z[te]])
    elif mode != "plain":
        raise ValueError(mode)
    m = P.make_model().fit(Xtr, y[tr])
    return te, m.decision_function(Xte)


def cv_scores(X, y, Z=None, mode="plain", folds=5, repeats=10, seed=17, n_jobs=1):
    """Same splitter, model and grid as rtm_protocol.oof_scores. Returns S (repeats, n) out-of-fold scores and
    Fold (repeats, n) fold index of every example in every repeat.  S.mean(0) equals rtm_protocol.oof_scores."""
    n = len(y)
    splits = list(RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed).split(X, y))
    res = Parallel(n_jobs=n_jobs)(delayed(_fit_split)(X, y, Z, mode, tr, te) for tr, te in splits)
    S = np.zeros((repeats, n)); Fold = np.zeros((repeats, n), int)
    for i, (te, sc) in enumerate(res):
        r, f = divmod(i, folds); S[r, te] = sc; Fold[r, te] = f
    return S, Fold


def cached_cv(cache_dir, name, X, y, Z=None, mode="plain", folds=5, repeats=10, seed=17, n_jobs=1):
    """cv_scores with an on-disk cache keyed by everything that determines the result."""
    key = hashlib.sha256(json.dumps([name, mode, folds, repeats, seed, list(X.shape), float(np.round(X.sum(), 6)),
                                     y.tolist()]).encode()).hexdigest()[:16]
    path = os.path.join(cache_dir, f"oof_{name}.npz") if cache_dir else None
    if path and os.path.exists(path):
        z = np.load(path)
        if str(z["key"]) == key: return z["S"], z["Fold"]
    S, Fold = cv_scores(X, y, Z, mode, folds, repeats, seed, n_jobs)
    if path:
        os.makedirs(cache_dir, exist_ok=True); np.savez_compressed(path, S=S, Fold=Fold, key=key)
    return S, Fold


# ----------------------------------------------------------------------------- statistics
def fast_auc(y, s):
    """Mann-Whitney AUROC with ties = 1/2 (identical to sklearn.roc_auc_score)."""
    pos = y == 1; n1 = pos.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return np.nan
    r = rankdata(s)
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def strat_auc(y, s, strata):
    """AUROC over positive-negative pairs that share a stratum (pair-count weighted): cannot be driven by anything
    that is constant within a stratum, e.g. response length when the strata are length bins."""
    num = den = 0.0
    for b in np.unique(strata):
        m = strata == b; n1 = int((y[m] == 1).sum()); n0 = int(m.sum() - n1)
        if n1 == 0 or n0 == 0: continue
        num += fast_auc(y[m], s[m]) * n1 * n0; den += n1 * n0
    return num / den if den else np.nan


def length_strata(ntok, y, n_bins=5, min_pos=10):
    """Quantile bins of token count; adjacent bins are merged until every bin holds >= min_pos reliable and
    >= min_pos failed answers.  Returns integer stratum ids and the final bin edges."""
    ntok = np.asarray(ntok, float).ravel()
    edges = np.unique(np.quantile(ntok, np.linspace(0, 1, n_bins + 1)[1:-1]))
    b = np.digitize(ntok, edges, right=True)
    while True:
        ids = np.unique(b)
        cnt = {i: (int(((b == i) & (y == 1)).sum()), int(((b == i) & (y == 0)).sum())) for i in ids}
        bad = [i for i in ids if min(cnt[i]) < min_pos]
        if not bad or len(ids) == 1: break
        i = bad[0]; j = ids[np.where(ids == i)[0][0] + 1] if i != ids[-1] else ids[np.where(ids == i)[0][0] - 1]
        b = np.where(b == i, j, b)
    return np.unique(b, return_inverse=True)[1], edges


def holm(p): return P.holm(p)


def wilson(k, n, z=1.959964):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def fold_aucs(y, S, Fold):
    """AUROC of each (repeat, fold) test set: array (repeats, folds)."""
    R, n = S.shape; K = int(Fold.max()) + 1
    return np.array([[fast_auc(y[Fold[r] == f], S[r][Fold[r] == f]) for f in range(K)] for r in range(R)])


def nadeau_bengio(da, n_train, n_test):
    """Corrected resampled t-test (Nadeau & Bengio 2003) for the mean of J paired performance differences from
    repeated CV.  da: all fold-level differences (any shape).  The factor (1/J + n_test/n_train) replaces 1/J to
    account for the overlap of training sets that makes the naive variance too small."""
    d = np.asarray(da, float).ravel(); J = len(d)
    var = (1.0 / J + n_test / n_train) * d.var(ddof=1)
    se = np.sqrt(var); tt = d.mean() / se if se > 0 else 0.0
    p = 2 * tdist.sf(abs(tt), J - 1); q = tdist.ppf(0.975, J - 1)
    return dict(diff=float(d.mean()), ci95=[float(d.mean() - q * se), float(d.mean() + q * se)], t=float(tt), p=float(p), J=int(J), se=float(se))


def paired_boot(y, Sa, Sb, B, rng, mode):
    """Bootstrap over questions for AUROC(a) - AUROC(b) on out-of-fold score matrices (repeats, n).
    mode='pooled'    : AUROC of the repeat-averaged score (the pre-specified estimator; ignores partition variation)
    mode='repmean'   : mean over repeats of the per-repeat AUROC difference
    mode='partition' : each replicate also draws one repeat at random, so partition variation enters the interval"""
    R, n = Sa.shape; d = []
    Ma, Mb = Sa.mean(0), Sb.mean(0)
    for _ in range(B):
        i = rng.integers(0, n, n); yi = y[i]
        if yi.min() == yi.max(): continue
        if mode == "pooled": d.append(fast_auc(yi, Ma[i]) - fast_auc(yi, Mb[i]))
        elif mode == "repmean": d.append(np.mean([fast_auc(yi, Sa[r][i]) - fast_auc(yi, Sb[r][i]) for r in range(R)]))
        elif mode == "partition":
            r = rng.integers(0, R); d.append(fast_auc(yi, Sa[r][i]) - fast_auc(yi, Sb[r][i]))
        else: raise ValueError(mode)
    return np.quantile(d, [.025, .975]).tolist()


def boot_ci(y, s, B, rng):
    n = len(y); out = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max(): continue
        out.append(fast_auc(y[i], s[i]))
    return np.quantile(out, [.025, .975]).tolist()


def contrast(y, Sa, Sb, args_boot, rng, folds, n_train, n_test, Fa=None, Fb=None, fold_a=None, fold_b=None):
    """All inference variants for one paired contrast a - b. Returns a dict (see README of stats_robustness)."""
    Ma, Mb = Sa.mean(0), Sb.mean(0)
    d, p = P.delong_paired(y, Ma, Mb)
    out = dict(diff_pooled=float(d), delong_p=float(p),
               ci_pooled=paired_boot(y, Sa, Sb, args_boot, rng, "pooled"),
               ci_repmean=paired_boot(y, Sa, Sb, args_boot, rng, "repmean"),
               ci_partition=paired_boot(y, Sa, Sb, args_boot, rng, "partition"))
    out["diff_repmean"] = float(np.mean([fast_auc(y, Sa[r]) - fast_auc(y, Sb[r]) for r in range(Sa.shape[0])]))
    if fold_a is not None: out["nadeau_bengio"] = nadeau_bengio(fold_a - fold_b, n_train, n_test)
    return out


def load_labels(labels_csv):
    import csv
    rows = list(csv.DictReader(open(labels_csv, encoding="utf-8-sig")))
    return [r["id"] for r in rows], np.array([int(r["label"]) for r in rows])
