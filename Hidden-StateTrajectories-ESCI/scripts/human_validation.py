#!/usr/bin/env python3
"""Human validation of the model-annotated labels (reviewer objection: 'the labels came from LLMs, not people').

Design.  The 900 frozen labels are split into three strata and humans label a sample, blind to the LLM labels:
    S1  LLM label = reliable (1), not adjudicated            sample n1  from N1 = 194
    S0  LLM label = failure (0), not adjudicated             sample n0  from N0 = 687
    SA  the 19 items the two LLM annotators disagreed on      census (all 19)
Sampling the two label classes separately gives DIRECT estimates of the two quantities Proposition 2 needs:
    pi  = P(truly reliable | labelled reliable)     rho = P(truly reliable | labelled failed)     gamma = pi - rho
(gamma is the factor by which independent label error shrinks AUROC - 1/2 and every AUROC difference).  Estimates are
population-weighted (stratum sizes), intervals are stratified bootstrap.  '?' answers are excluded from the point estimate and
bounded by treating them all as reliable / all as failed.

    python scripts/human_validation.py make-sample --labels-dir labels_frozen --out human_validation --n1 80 --n0 80 --seed 2026
        -> sample_key.csv (KEEP PRIVATE), human_sheet_H1.csv, human_sheet_H2.csv, GUIDELINES.md, INSTRUCTIONS.md
    python scripts/human_validation.py analyze --key human_validation/sample_key.csv --labels labels_frozen/labels.csv \
        --sheet human_validation/human_sheet_H1_filled.csv --sheet human_validation/human_sheet_H2_filled.csv \
        [--adjudicated human_validation/human_adjudication.csv] --h1-ci -0.059 0.016 --out human_validation/
    python scripts/human_validation.py selftest
"""
import argparse, csv, hashlib, json, os, shutil, sys, tempfile
import numpy as np


def read_csv(p): return list(csv.DictReader(open(p, encoding="utf-8-sig", newline="")))


def write_csv(p, rows, cols):
    with open(p, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)


# ----------------------------------------------------------------------------- make-sample
def make_sample(a):
    ld = a.labels_dir; lab = {r["id"]: int(r["label"]) for r in read_csv(os.path.join(ld, "labels.csv"))}
    content = {r["id"]: r for r in read_csv(os.path.join(ld, "annotator_A.csv"))}
    adj = {r["id"] for r in read_csv(os.path.join(ld, "disagreements.csv"))}
    assert set(lab) == set(content) and adj <= set(lab), "labels / annotator sheet / disagreements ids do not line up"
    rng = np.random.default_rng(a.seed)
    s1 = sorted(i for i, l in lab.items() if l == 1 and i not in adj); s0 = sorted(i for i, l in lab.items() if l == 0 and i not in adj)
    pick1 = list(rng.choice(s1, min(a.n1, len(s1)), replace=False)); pick0 = list(rng.choice(s0, min(a.n0, len(s0)), replace=False))
    key = [dict(id=i, stratum="S1", llm_label=1) for i in pick1] + [dict(id=i, stratum="S0", llm_label=0) for i in pick0] + \
          [dict(id=i, stratum="SA", llm_label=lab[i]) for i in sorted(adj)]
    os.makedirs(a.out, exist_ok=True)
    write_csv(os.path.join(a.out, "sample_key.csv"), key, ["id", "stratum", "llm_label"])
    ids = [k["id"] for k in key]
    for name, sd in (("H1", a.seed + 1), ("H2", a.seed + 2)):                       # independent random order per annotator
        order = list(np.random.default_rng(sd).permutation(ids))
        rows = [dict(id=i, question=content[i]["question"], gold_answer=content[i]["gold_answer"], model_output=content[i]["model_output"], label="", notes="") for i in order]
        write_csv(os.path.join(a.out, f"human_sheet_{name}.csv"), rows, ["id", "question", "gold_answer", "model_output", "label", "notes"])
    shutil.copy(os.path.join(ld, "GUIDELINES.md"), os.path.join(a.out, "GUIDELINES.md"))
    open(os.path.join(a.out, "INSTRUCTIONS.md"), "w", encoding="utf-8").write(INSTRUCTIONS.format(n=len(ids), sha=hashlib.sha256(open(os.path.join(ld, "GUIDELINES.md"), "rb").read()).hexdigest()[:16]))
    print(f"sample: S1 {len(pick1)}/{len(s1)}, S0 {len(pick0)}/{len(s0)}, SA {len(adj)} (census) = {len(ids)} items; sheets written to {a.out}")


INSTRUCTIONS = """# Instructions for the human annotators ({n} items)

You are checking labels that were produced by language-model annotators. You do not see their labels and must not look for them.

1. Open ONLY your own sheet (`human_sheet_H1.csv` or `human_sheet_H2.csv`). Do not open the other person's sheet or `sample_key.csv`.
2. Read `GUIDELINES.md` first (SHA-256 prefix {sha}). Apply exactly those rules; they are the rules the model annotators used.
3. For each row read `question`, `gold_answer`, `model_output` and fill `label`: `1` = reliable (correct under the guidelines), `0` = failure, `?` = genuinely unsure. Use `notes` for anything unusual.
4. Work independently of the other annotator; do not discuss individual items until both sheets are finished.
5. Save as **CSV UTF-8** under the name `human_sheet_H1_filled.csv` / `human_sheet_H2_filled.csv`. Do not reorder, add or delete rows.
6. About 30-40 seconds per item; split over two sittings. Do not run any detector or look at any detector output while labelling.
"""


# ----------------------------------------------------------------------------- analysis
def parse_label(x):
    x = (x or "").strip()
    return {"1": 1, "0": 0, "?": None}.get(x, "bad")


def load_sheets(paths, key_ids):
    out = []
    for p in paths:
        d = {}
        for r in read_csv(p):
            v = parse_label(r["label"])
            if v == "bad": sys.exit(f"{p}: item {r['id']} has label {r['label']!r}; allowed values are 1, 0, ?")
            d[r["id"]] = v
        miss = [i for i in key_ids if i not in d or (d[i] is None and False)]
        if miss: sys.exit(f"{p}: {len(miss)} sampled items are missing, first {miss[0]}")
        blank = [r["id"] for r in read_csv(p) if (r["label"] or "").strip() == ""]
        if blank: sys.exit(f"{p}: {len(blank)} items are unlabelled (blank), first {blank[0]}; use ? for unsure")
        out.append(d)
    return out


def cohen_kappa(x, y):
    x, y = np.asarray(x), np.asarray(y); po = (x == y).mean()
    pe = sum((x == v).mean() * (y == v).mean() for v in (0, 1)); return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def estimate(key, human, Npop, fill=None):
    """key: list of dict(id, stratum, llm_label); human: id -> 1/0/None; fill: value used for None (None = exclude)."""
    def val(i):
        v = human[i]; return fill if v is None else v
    def rate(rows):
        v = [val(r["id"]) for r in rows]; v = [x for x in v if x is not None]
        return (float(np.mean(v)) if v else np.nan, len(v))
    S1 = [r for r in key if r["stratum"] == "S1"]; S0 = [r for r in key if r["stratum"] == "S0"]; SA = [r for r in key if r["stratum"] == "SA"]
    A1 = [r for r in SA if r["llm_label"] == 1]; A0 = [r for r in SA if r["llm_label"] == 0]
    p1, _ = rate(S1); r0, _ = rate(S0); pa1, _ = rate(A1); ra0, _ = rate(A0)
    N1, N0, nA1, nA0 = Npop["S1"], Npop["S0"], len(A1), len(A0)
    pi = (N1 * p1 + nA1 * (0 if np.isnan(pa1) else pa1)) / (N1 + nA1)
    rho = (N0 * r0 + nA0 * (0 if np.isnan(ra0) else ra0)) / (N0 + nA0)
    return float(pi), float(rho)


def implied(pi, rho, q):
    """Joint table of (LLM label, human label) implied by pi, rho and the LLM-reliable share q."""
    t = {(1, 1): pi * q, (1, 0): (1 - pi) * q, (0, 1): rho * (1 - q), (0, 0): (1 - rho) * (1 - q)}
    po = t[(1, 1)] + t[(0, 0)]; l1 = q; h1 = t[(1, 1)] + t[(0, 1)]; pe = l1 * h1 + (1 - l1) * (1 - h1)
    return po, (po - pe) / (1 - pe)


def analyze(a):
    key = read_csv(a.key); lab = {r["id"]: int(r["label"]) for r in read_csv(a.labels)}
    for r in key: r["llm_label"] = int(r["llm_label"])
    assert all(lab[r["id"]] == r["llm_label"] for r in key), "sample_key.csv disagrees with labels.csv (wrong or modified file)"
    ids = [r["id"] for r in key]; sheets = load_sheets(a.sheet, ids)
    nS1 = sum(1 for i, l in lab.items() if l == 1) - sum(1 for r in key if r["stratum"] == "SA" and r["llm_label"] == 1)
    nS0 = sum(1 for i, l in lab.items() if l == 0) - sum(1 for r in key if r["stratum"] == "SA" and r["llm_label"] == 0)
    Npop = {"S1": nS1, "S0": nS0}; q = sum(lab.values()) / len(lab)
    res = {"n_items": len(ids), "n_annotators": len(sheets), "population": dict(N=len(lab), reliable=int(sum(lab.values())), S1=nS1, S0=nS0, SA=sum(1 for r in key if r["stratum"] == "SA")),
           "sample": {s: sum(1 for r in key if r["stratum"] == s) for s in ("S1", "S0", "SA")}}
    # human-human agreement
    if len(sheets) == 2:
        both = [i for i in ids if sheets[0][i] is not None and sheets[1][i] is not None]
        res["human_human"] = dict(n_both_defined=len(both), raw_agreement=float(np.mean([sheets[0][i] == sheets[1][i] for i in both])),
                                  cohen_kappa=cohen_kappa([sheets[0][i] for i in both], [sheets[1][i] for i in both]),
                                  n_unsure_H1=sum(sheets[0][i] is None for i in ids), n_unsure_H2=sum(sheets[1][i] is None for i in ids))
        adj = {r["id"]: parse_label(r["label"]) for r in read_csv(a.adjudicated)} if a.adjudicated and os.path.exists(a.adjudicated) else {}
        human, unresolved = {}, []
        for i in ids:
            x, y = sheets[0][i], sheets[1][i]
            if x == y: human[i] = x
            elif i in adj and adj[i] in (0, 1): human[i] = adj[i]
            else: human[i] = None; unresolved.append(i)
        res["human_human"]["disagreements"] = [i for i in ids if sheets[0][i] != sheets[1][i]]; res["human_human"]["unresolved_after_adjudication"] = len(unresolved)
        if unresolved and not adj:
            pd = os.path.join(a.out, "human_adjudication_TEMPLATE.csv"); os.makedirs(a.out, exist_ok=True)
            write_csv(pd, [dict(id=i, label="") for i in res["human_human"]["disagreements"]], ["id", "label"])
            print(f"{len(unresolved)} items need adjudication; template written to {pd} (fill label with 1/0, a third person or both humans together using GUIDELINES.md, then rerun with --adjudicated)")
    else:
        human = dict(sheets[0])
    # point estimates and bounds
    pi, rho = estimate(key, human, Npop); lo_pi, lo_rho = estimate(key, human, Npop, fill=0); hi_pi, hi_rho = estimate(key, human, Npop, fill=1)
    res["n_unsure_consensus"] = sum(1 for i in ids if human[i] is None)
    res["estimates"] = dict(pi=pi, rho=rho, gamma=pi - rho, bounds_if_unsure_all_failed=dict(pi=lo_pi, rho=lo_rho, gamma=lo_pi - lo_rho),
                            bounds_if_unsure_all_reliable=dict(pi=hi_pi, rho=hi_rho, gamma=hi_pi - hi_rho))
    # exact (Clopper-Pearson) intervals for the raw stratum rates: the percentile bootstrap is weak when only 0-3 errors are seen
    from scipy.stats import beta
    def cp(k, n): return [float(beta.ppf(.025, k, n - k + 1)) if k > 0 else 0.0, float(beta.ppf(.975, k + 1, n - k)) if k < n else 1.0]
    ex = {}
    for st, name in (("S1", "share_human_reliable_among_sampled_LLM_reliable"), ("S0", "share_human_reliable_among_sampled_LLM_failed")):
        v = [human[r["id"]] for r in key if r["stratum"] == st and human[r["id"]] is not None]; k = int(sum(v))
        ex[name] = dict(k=k, n=len(v), rate=k / len(v) if v else None, clopper_pearson_95=cp(k, len(v)) if v else None)
    res["stratum_rates_exact"] = ex
    # stratified bootstrap
    rng = np.random.default_rng(a.seed); S1 = [r for r in key if r["stratum"] == "S1"]; S0 = [r for r in key if r["stratum"] == "S0"]; SA = [r for r in key if r["stratum"] == "SA"]
    bs = []
    for _ in range(a.boot):
        k2 = [S1[j] for j in rng.integers(0, len(S1), len(S1))] + [S0[j] for j in rng.integers(0, len(S0), len(S0))] + SA
        bs.append(estimate(k2, human, Npop))
    bs = np.array(bs); g = bs[:, 0] - bs[:, 1]
    q1 = lambda v: [float(np.nanquantile(v, .025)), float(np.nanquantile(v, .975))]
    res["estimates"]["ci95"] = dict(pi=q1(bs[:, 0]), rho=q1(bs[:, 1]), gamma=q1(g))
    po, kap = implied(pi, rho, q); res["llm_vs_human_population_weighted"] = dict(raw_agreement=float(po), cohen_kappa=float(kap))
    # consequence for the paper's H1 conclusion (Corollary of Proposition 2)
    L, U = a.h1_ci; glo = res["estimates"]["ci95"]["gamma"][0]
    res["corollary_H1"] = dict(observed_ci=[L, U], gamma_lower_95=glo, true_diff_ci_if_gamma_is_lower_bound=[L / glo, U / glo] if glo > 0 else None,
                               gain_005_still_ruled_out=bool(glo > 0 and U / glo < 0.05), gamma_needed_for_0_05_gain=float(U / 0.05) if U > 0 else None)
    # derived label files for the sensitivity analyses (run rtm_protocol / stats_robustness with --allow-unfrozen)
    os.makedirs(a.out, exist_ok=True)
    corr = dict(lab)
    for i in ids:
        if human[i] is not None: corr[i] = human[i]
    write_csv(os.path.join(a.out, "labels_human_corrected.csv"), [dict(id=i, label=v) for i, v in corr.items()], ["id", "label"])
    flip = dict(lab)
    for r in key:
        if r["stratum"] == "SA": flip[r["id"]] = 1 - flip[r["id"]]
    write_csv(os.path.join(a.out, "labels_adjudication_flipped.csv"), [dict(id=i, label=v) for i, v in flip.items()], ["id", "label"])
    res["derived_label_files"] = dict(human_corrected_n_changed=int(sum(corr[i] != lab[i] for i in lab)), adjudication_flipped_n_changed=int(sum(flip[i] != lab[i] for i in lab)))
    json.dump(res, open(os.path.join(a.out, "human_validation_results.json"), "w", encoding="utf-8"), indent=1)
    e = res["estimates"]
    print(json.dumps(res, indent=1)); print(f"\nSUMMARY  pi={e['pi']:.3f} {e['ci95']['pi']}  rho={e['rho']:.3f} {e['ci95']['rho']}  gamma={e['gamma']:.3f} {e['ci95']['gamma']}")
    return res


# ----------------------------------------------------------------------------- selftest
def selftest(a):
    """Simulate humans who see the truth (with optional noise) on top of the REAL frozen label ids; check the estimators recover
    known pi and rho, that the 95% intervals cover them at about the nominal rate, and that blinding/sheets are sane."""
    ld = a.labels_dir; tmp = tempfile.mkdtemp(prefix="hv_"); lab = {r["id"]: int(r["label"]) for r in read_csv(os.path.join(ld, "labels.csv"))}
    ns = argparse.Namespace(labels_dir=ld, out=tmp, n1=80, n0=80, seed=2026); make_sample(ns)
    sheet = read_csv(os.path.join(tmp, "human_sheet_H1.csv")); key = read_csv(os.path.join(tmp, "sample_key.csv"))
    assert len(sheet) == len(key) and all(r["label"] == "" for r in sheet), "sheet must be blank"
    assert [r["id"] for r in sheet] != [r["id"] for r in key] and [r["id"] for r in sheet] != [r["id"] for r in read_csv(os.path.join(tmp, "human_sheet_H2.csv"))], "orders must differ"
    assert set(r["id"] for r in sheet) == set(r["id"] for r in key) and len(set(r["id"] for r in sheet)) == len(sheet)
    txt = open(os.path.join(tmp, "human_sheet_H1.csv"), encoding="utf-8-sig").read(); assert "llm_label" not in txt and "stratum" not in txt, "LLM label must not leak"
    print("sheet checks OK")
    pi0, rho0, flip = 0.90, 0.03, 0.02; cover = np.zeros(2); gcov = 0; est = []; R = a.reps
    for rep in range(R):
        rng = np.random.default_rng(100 + rep); truth = {i: int(rng.random() < (pi0 if l == 1 else rho0)) for i, l in lab.items()}
        ns = argparse.Namespace(labels_dir=ld, out=tmp, n1=80, n0=80, seed=500 + rep); make_sample(ns)
        for h in ("H1", "H2"):
            rows = read_csv(os.path.join(tmp, f"human_sheet_{h}.csv"))
            for r in rows: t = truth[r["id"]]; r["label"] = str(1 - t if rng.random() < flip else t)
            write_csv(os.path.join(tmp, f"human_sheet_{h}_filled.csv"), rows, ["id", "question", "gold_answer", "model_output", "label", "notes"])
        adj = argparse.Namespace(key=os.path.join(tmp, "sample_key.csv"), labels=os.path.join(ld, "labels.csv"), sheet=[os.path.join(tmp, "human_sheet_H1_filled.csv"), os.path.join(tmp, "human_sheet_H2_filled.csv")],
                                 adjudicated=None, h1_ci=[-0.059, 0.016], out=tmp, boot=400, seed=1)
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()): r = analyze(adj)
        e = r["estimates"]; est.append((e["pi"], e["rho"]))
        # target: P(true=1|LLM=1), P(true=1|LLM=0) in this finite population
        p_true = np.mean([truth[i] for i, l in lab.items() if l == 1]); r_true = np.mean([truth[i] for i, l in lab.items() if l == 0])
        cover += [e["ci95"]["pi"][0] <= p_true <= e["ci95"]["pi"][1], e["ci95"]["rho"][0] <= r_true <= e["ci95"]["rho"][1]]
        gcov += e["ci95"]["gamma"][0] <= (p_true - r_true) <= e["ci95"]["gamma"][1]
    est = np.array(est); print(f"{R} simulated validations: mean pi_hat {est[:,0].mean():.3f} (truth {pi0}), mean rho_hat {est[:,1].mean():.3f} (truth {rho0}); two humans each flip {flip} of items, consensus removes most of that noise")
    print(f"coverage of finite-population pi, rho, gamma by 95% bootstrap CIs (truth = finite-population value): {cover/R}, {gcov/R}")
    assert abs(est[:, 0].mean() - pi0) < 0.03 and abs(est[:, 1].mean() - rho0) < 0.015, "estimators must recover the planted pi and rho"
    assert cover.min() / R >= 0.85 and gcov / R >= 0.85, "bootstrap intervals badly under-cover"
    print("selftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make-sample"); m.add_argument("--labels-dir", required=True); m.add_argument("--out", required=True)
    m.add_argument("--n1", type=int, default=80); m.add_argument("--n0", type=int, default=80); m.add_argument("--seed", type=int, default=2026)
    an = sub.add_parser("analyze"); an.add_argument("--key", required=True); an.add_argument("--labels", required=True); an.add_argument("--sheet", action="append", required=True)
    an.add_argument("--adjudicated"); an.add_argument("--h1-ci", type=float, nargs=2, default=[-0.059, 0.016]); an.add_argument("--out", required=True)
    an.add_argument("--boot", type=int, default=4000); an.add_argument("--seed", type=int, default=7)
    st = sub.add_parser("selftest"); st.add_argument("--labels-dir", default="labels_frozen"); st.add_argument("--reps", type=int, default=60)
    a = ap.parse_args()
    {"make-sample": make_sample, "analyze": analyze, "selftest": selftest}[a.cmd](a)
