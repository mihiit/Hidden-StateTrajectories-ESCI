import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
OUT = "/home/claude/work/figs/"
R = json.load(open(OUT + "sim_results.json"))
plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.8, "axes.labelsize": 9.5,
                     "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
                     "savefig.dpi": 300, "figure.dpi": 150})
C = {"B0": "#7f7f7f", "B2": "#0072B2", "B4": "#009E73", "RTM": "#D55E00"}
W1, W2 = 3.45, 7.1   # IEEE column / two-column width (in)
seeds = [17, 42, 123]
D = {"B0": [.2644, .6514, .5584], "B2": [.5962, .5072, .6104], "B4": [.3810, .7212, .5863], "RTM": [.7812, .7452, .8015]}
LAB = {"B0": "B0 seeded prior", "B2": "B2 static probe", "B4": "B4 ICR-inspired", "RTM": "B5 RTM (ours)"}
def tag(ax, s): ax.text(-0.02, 1.06, s, transform=ax.transAxes, fontsize=10.5, fontweight="bold", va="bottom", ha="right")

# ---------- Fig 1: headline result, column width, two stacked panels (sits under the abstract) ----------
nq = R["null_cur"]; lo_n, hi_n = nq["q025"], nq["q975"]
fig, (a, b) = plt.subplots(2, 1, figsize=(W1, 2.75), gridspec_kw={"height_ratios": [1.5, 1]})
a.axhspan(lo_n, hi_n, color="#d9d9d9", alpha=.8, lw=0, zorder=0); a.axhline(.5, color="k", ls=":", lw=.9, zorder=1)
for i, k in enumerate(["B0", "B2", "B4", "RTM"]):
    x = np.arange(3) + (i - 1.5) * .17
    a.plot(x, D[k], "o", color=C[k], ms=5.5 if k == "RTM" else 4.2, mec="white", mew=.7, zorder=3, label=LAB[k])
    for xx, yy in zip(x, D[k]): a.plot([xx, xx], [.5, yy], color=C[k], lw=1.3 if k == "RTM" else .7, alpha=.85 if k == "RTM" else .5, zorder=2)
a.set_xticks(range(3)); a.set_xticklabels([f"split {s}" for s in seeds], fontsize=7.8); a.tick_params(axis="y", labelsize=7.8); b.tick_params(labelsize=7.6); a.set_xlim(-.55, 2.55); a.set_ylim(.2, .9); a.set_ylabel("Test AUROC", fontsize=8)
from matplotlib.patches import Patch
h, l = a.get_legend_handles_labels(); h.append(Patch(fc="#d9d9d9")); l.append("95% range, no signal")
a.legend(h, l, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(.5, 1.0), handletextpad=.15, columnspacing=.7, fontsize=6.3)
a.text(0.012, 0.95, "(a)", transform=a.transAxes, fontsize=9, fontweight="bold", va="top")
comp = [("RTM − B0", .243, .002, .491, "B0"), ("RTM − B2", .191, -.087, .453, "B2"), ("RTM − B4", .215, -.082, .474, "B4")]
for i, (n, d, lo, hi, k) in enumerate(comp):
    b.plot([lo, hi], [i, i], color=C[k], lw=2.2, solid_capstyle="butt"); b.plot(d, i, "o", color="white", mec=C[k], mew=1.8, ms=5, zorder=3)
    b.text(.985, i, f"p$_{{Holm}}$ = {[.144,.252,.252][i]:.3f}", fontsize=6.8, ha="right", va="center", transform=b.get_yaxis_transform())
b.axvline(0, color="k", lw=.9); b.set_yticks(range(3)); b.set_yticklabels([c[0] for c in comp]); b.invert_yaxis()
b.set_xlabel("Paired AUROC difference (95% CI), split 123", fontsize=7.8, labelpad=1); b.set_xlim(-.15, .75); b.set_ylim(2.6, -.6)
b.text(0.012, 0.97, "(b)", transform=b.transAxes, fontsize=9, fontweight="bold", va="top")
fig.tight_layout(h_pad=.5, pad=.3); fig.savefig(OUT + "fig_pilot_hero.png"); plt.close()

# ---------- Fig 2: sampling spread vs number of reliable examples (column width, stacked) ----------
fig, a = plt.subplots(1, 1, figsize=(W1, 2.2))
nul = np.load(OUT + "null_cur_fold.npy"); alt = np.load(OUT + "alt_cur_fold.npy")
bins = np.linspace(0, 1, 36)
a.hist(nul, bins=bins, density=True, color="#bdbdbd", alpha=.9, label="no signal (true AUROC 0.50)")
a.hist(alt, bins=bins, density=True, color=C["RTM"], alpha=.45, label="true AUROC 0.78")
for v in D["RTM"]: a.axvline(v, color=C["RTM"], lw=1.3)
for v in D["B0"]: a.plot(v, 0.14, "v", color="#333", ms=6)
a.set_xlabel("AUROC on one test fold (8 reliable, 52 failed)"); a.set_ylabel("Density"); a.set_xlim(0, 1); a.set_ylim(0, 6.3)
a.legend(frameon=False, loc="upper left", fontsize=7.8, bbox_to_anchor=(0, 1.0))
a.text(.02, .60, "vertical lines: RTM results\n▼  B0 results", transform=a.transAxes, ha="left", va="top", fontsize=7.8)

fig.tight_layout(pad=.3); fig.savefig(OUT + "fig2_sampling.png"); plt.close()

# ---------- Fig 3: power curves (column width, stacked) ----------
P = R["power"]; Ns = P["Ns"]
fig, (a, b) = plt.subplots(2, 1, figsize=(W1, 4.5), sharex=True)
cols = {0.1: "#56B4E9", 0.15: "#0072B2", 0.2: "#001f4d"}
for ax, al, ttl in [(a, "0.0500", "single primary comparison, α = 0.05"), (b, "0.0167", "three comparisons, α = 0.05/3")]:
    for d in [0.1, 0.15, 0.2]:
        y = P["curves"][f"d{d}_a{al}"]
        ax.plot(Ns, y, "o-", color=cols[d], lw=1.8, ms=4.2, label=f"true gap = {d:.2f}")
    ax.axhline(.8, color="k", ls="--", lw=.8); ax.text(1630, .81, "80% power", fontsize=7.6, ha="right", va="bottom")
    ax.axvspan(350, 500, color="#f0e0d0", alpha=.8, lw=0, zorder=0)
    ax.set_title(ttl, fontsize=8.6, pad=3); ax.set_xlim(150, 1650); ax.set_ylim(0, 1.04); ax.set_ylabel("Power (paired DeLong)")
a.legend(frameon=False, loc="lower right", fontsize=7.8); b.text(425, .03, "350–500\nrule of\nthumb", fontsize=7.2, ha="center", color="#8a5a2b")
b.set_xlabel("Total questions N (12.5% reliable)"); tag(a, "(a)"); tag(b, "(b)")
fig.tight_layout(h_pad=1.0); fig.savefig(OUT + "fig3_power.png"); plt.close()

# ---------- Fig 4: label-correction sensitivity (dumbbell) ----------
rows = [("s17", {"B0": (.349, .264), "B2": (.594, .596), "B4": (.549, .381), "RTM": (.789, .781)}),
        ("s42", {"B0": (.704, .651), "B2": (.522, .507), "B4": (.474, .721), "RTM": (.827, .745)})]
fig, ax = plt.subplots(figsize=(W1, 3.1)); y = 0; yt = []; yl = []
for s, dd in rows:
    for k in ["RTM", "B4", "B2", "B0"]:
        pre, post = dd[k]
        ax.plot([pre, post], [y, y], color=C[k], lw=2.2, alpha=.55, zorder=1)
        ax.plot(pre, y, "o", mfc="white", mec=C[k], mew=1.6, ms=6.5, zorder=3); ax.plot(post, y, "o", color=C[k], ms=6.5, zorder=3)
        yt.append(y); yl.append(f"{LAB[k].split(' ')[0] if k!='RTM' else 'B5'}  {s}"); y += 1
    y += .6
ax.axvline(.5, color="k", ls=":", lw=.8); ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=8.2); ax.invert_yaxis()
ax.set_xlabel("Test AUROC"); ax.set_xlim(.2, .9)
ax.plot([], [], "o", mfc="white", mec="#333", mew=1.4, label="before correction"); ax.plot([], [], "o", color="#333", label="after correction")
ax.legend(frameon=False, loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(.5, 1.0))
fig.tight_layout(); fig.savefig(OUT + "fig4_label_sensitivity.png"); plt.close()

# ---------- Fig 5: verifier audit ----------
def wilson(k, n, z=1.959964):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d; return c - h, c + h
lo, hi = wilson(3, 24); lo2, hi2 = wilson(4, 179)
fig, ax = plt.subplots(figsize=(W1, 1.95))
ax.errorbar([12.5], [0], xerr=[[12.5 - lo * 100], [hi * 100 - 12.5]], fmt="o", color=C["RTM"], capsize=4, ms=7, lw=2)
ax.errorbar([4 / 179 * 100], [1], xerr=[[4 / 179 * 100 - lo2 * 100], [hi2 * 100 - 4 / 179 * 100]], fmt="o", mfc="white", color="#666", capsize=4, ms=7, lw=1.6, ls="none")
ax.text(12.5, -.32, "12.5%  [4.3, 31.0]", ha="center", fontsize=8, color=C["RTM"])
ax.text(7.5, 1.0, "≥2.2%  [0.9, 5.6]\n(lower bound only)", ha="left", fontsize=7.6, color="#555", va="center")
ax.set_yticks([0, 1]); ax.set_yticklabels(["“reliable” labels\n(full census, 3/24)", "“failure” labels\n(partial audit, 4/179)"], fontsize=8)
ax.set_ylim(1.75, -.7); ax.set_xlim(0, 36); ax.set_xlabel("Verifier error rate (%), Wilson 95% CI")
fig.tight_layout(); fig.savefig(OUT + "fig5_audit.png"); plt.close()

# ---------- Fig 6: revised evaluation protocol (vertical, column width) ----------
fig, ax = plt.subplots(figsize=(W1, 2.75)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 100)
steps = [("1  Generate", "HaluEval QA, Qwen2.5-1.5B, hidden states", "#eef2f7", "#334"),
         ("2  Label, blind", "two annotators, no detector scores", "#fdf0e6", "#a63"),
         ("3  Freeze data", "labels and test split locked first", "#fdf0e6", "#a63"),
         ("4  Features", "five families, same preprocessing", "#eef2f7", "#334"),
         ("5  Tune in CV", "nested CV inside training folds only", "#eef2f7", "#334"),
         ("6  Compare", "paired DeLong, bootstrap, Holm; CIs", "#eef2f7", "#334")]
ax.set_ylim(13, 102)
bh, gp, x0, bw = 11.0, 3.6, 2, 82
for i, (t, d, fc, ec) in enumerate(steps):
    y = 100 - (i + 1) * bh - i * gp
    ax.add_patch(FancyBboxPatch((x0, y), bw, bh, boxstyle="round,pad=0.25", fc=fc, ec=ec, lw=1.0))
    ax.text(x0 + 2.5, y + bh * .72, t, ha="left", va="center", fontsize=8.0, fontweight="bold")
    ax.text(x0 + 2.5, y + bh * .27, d, ha="left", va="center", fontsize=7.2)
    if i < len(steps) - 1: ax.add_patch(FancyArrowPatch((x0 + bw / 2, y - .3), (x0 + bw / 2, y - gp + .5), arrowstyle="-|>", mutation_scale=8, color="#333"))
def brace(i0, i1, label, col, bg):
    ytop = 100 - i0 * (bh + gp) + .6; ybot = 100 - (i1 + 1) * bh - i1 * gp - .6
    ax.add_patch(Rectangle((x0 + bw + 2.5, ybot), 2.4, ytop - ybot, fc=bg, ec="none"))
    ax.text(x0 + bw + 8.2, (ytop + ybot) / 2, label, rotation=90, ha="center", va="center", fontsize=7.2, color=col)
brace(1, 2, "blind zone", "#7a3d12", "#f6d9c3"); brace(3, 5, "matched setup", "#1f3a5f", "#dce6f2")
fig.savefig(OUT + "fig6_protocol.png", bbox_inches="tight", pad_inches=.03); plt.close()

# ---------- Fig 7: feature families on the layer x token grid ----------
Lg, Tg = 6, 8
fig, axs = plt.subplots(1, 4, figsize=(W1 + .15, 1.85))
def base(ax, title):
    ax.imshow(np.ones((Lg, Tg, 3)) * .94, origin="lower", aspect="auto", interpolation="nearest", extent=(0, Tg, 0, Lg))
    for x in range(Tg + 1): ax.plot([x, x], [0, Lg], color="white", lw=.8)
    for y in range(Lg + 1): ax.plot([0, Tg], [y, y], color="white", lw=.8)
    ax.set_xlim(0, Tg); ax.set_ylim(0, Lg); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_title(title, fontsize=7.8, pad=2)
def arr(ax, x0, y0, x1, y1, c): ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=6, color=c, lw=1.3, shrinkA=0, shrinkB=0))
base(axs[0], "F: final\nstate"); axs[0].add_patch(Rectangle((Tg - 1, Lg - 1), 1, 1, fc=C["B2"], ec="none"))
base(axs[1], "L: layer-axis\n(Δ, angle)")
for t in range(Tg):
    for l in range(Lg - 1): arr(axs[1], t + .5, l + .5, t + .5, l + 1.5, C["B4"]) if t % 2 == 0 else None
base(axs[2], "T: token-axis\n(speed, var.)")
for l in range(Lg):
    for x0 in (.3, 4.3): arr(axs[2], x0, l + .5, x0 + 3.3, l + .5, "#E69F00")
base(axs[3], "A: L + T\n(RTM)")
axs[3].imshow(np.ones((Lg, Tg, 3)) * np.array(matplotlib.colors.to_rgb(C["RTM"])) , origin="lower", aspect="auto", extent=(0, Tg, 0, Lg), alpha=.85)
for x in range(Tg + 1): axs[3].plot([x, x], [0, Lg], color="white", lw=.8)
for y in range(Lg + 1): axs[3].plot([0, Tg], [y, y], color="white", lw=.8)
axs[0].set_ylabel("layer →", fontsize=7.6); fig.text(.5, .01, "generated token →", ha="center", fontsize=7.6)
fig.tight_layout(rect=(0, .07, 1, 1), w_pad=.4); fig.savefig(OUT + "fig7_families.png"); plt.close()
print("done")

# ---------- equation images ----------
import matplotlib as mpl
mpl.rcParams["mathtext.fontset"] = "stix"
eqs = {
 "eq1": r"$V_{l,t}=\Vert H_{l,t}-H_{l,t-1}\Vert_{2}$",
 "eq2": r"$D_{l,t}=\Vert H_{l,t}-H_{l-1,t}\Vert_{2}$",
 "eq4": r"$\mathrm{AUROC}_{\mathrm{obs}}-\dfrac{1}{2}=(\pi-\rho)\left(\mathrm{AUROC}-\dfrac{1}{2}\right)$",
 "eq5": r"$\mathrm{AUROC}(M)=\lambda=\mathbb{E}\left[\dfrac{T_f}{T_f+T_r}\right]$",
 "eq3": r"$\theta_{l,t}=\arccos\left(\dfrac{H_{l,t}\cdot H_{l-1,t}}{\Vert H_{l,t}\Vert\,\Vert H_{l-1,t}\Vert}\right)$",
}
for k, t in eqs.items():
    fig = plt.figure(figsize=(3, .8)); fig.text(0, .5, t, fontsize=11, va="center")
    fig.savefig(OUT + k + ".png", bbox_inches="tight", pad_inches=.02, transparent=False, facecolor="white"); plt.close()
print("eq done")

# ---------- Fig 8: label-noise attenuation ----------
LN = json.load(open(OUT + "label_noise.json")); g = np.array(LN["true_auc"])
fig, ax = plt.subplots(figsize=(W1, 2.9))
ax.plot([.55, .95], [.55, .95], color="k", ls=":", lw=.9)
styles = [("perfect labels (PPV 100%, FN 0%)", "#555", "-"), ("PPV 95.7%, FN 1%", "#56B4E9", "-"),
          ("pilot point estimate (PPV 87.5%, FN 2.2%)", C["RTM"], "-"), ("pilot CI worst case (PPV 69%, FN 4%)", "#8c1d00", "--")]
labs = {"perfect labels (PPV 100%, FN 0%)": "no label error", "PPV 95.7%, FN 1%": "PPV 95.7%, FN 1%",
        "pilot point estimate (PPV 87.5%, FN 2.2%)": "pilot estimate", "pilot CI worst case (PPV 69%, FN 4%)": "worst case in CI"}
for k, c, ls in styles: ax.plot(g, LN["curves"][k], ls, color=c, lw=1.9 if "pilot point" in k else 1.3, label=labs[k])
ax.annotate("true 0.78 reads as 0.74", xy=(.78, .741), xytext=(.66, .84), fontsize=7.8, color=C["RTM"], arrowprops=dict(arrowstyle="-|>", color=C["RTM"], lw=1.1))
ax.set_xlabel("True AUROC of the detector"); ax.set_ylabel("AUROC measured against verifier labels"); ax.set_xlim(.55, .95); ax.set_ylim(.5, .95)
ax.legend(frameon=False, fontsize=7.4, loc="lower right")
fig.tight_layout(); fig.savefig(OUT + "fig8_label_noise.png"); plt.close()
print("fig8 done")

# ---------- Fig 9: length can masquerade as trajectory signal ----------
LC = json.load(open(OUT + "length_confound.json")); g = np.array(LC["length_auc"])
fig, ax = plt.subplots(figsize=(W1, 2.7))
ax.axhline(.5, color="k", ls=":", lw=.9)
for key, c, ls, lab in [("len_only", "#555555", "-", "length only"), ("features", C["RTM"], "-", "trajectory features with no content"), ("features_resid", C["B2"], "--", "same features, length regressed out")]:
    v = np.array(LC[key]); m = v.mean(1); lo, hi = np.quantile(v, .025, 1), np.quantile(v, .975, 1)
    ax.plot(g, m, ls, color=c, lw=1.8, marker="o", ms=3.8, label=lab)
    if key == "features": ax.fill_between(g, lo, hi, color=c, alpha=.15, lw=0)
TH = json.load(open(OUT + "theory_check.json"))["p1"]
ax.plot([r["length_auc"] for r in TH], [max(r["lambda_theory"], 1 - r["lambda_theory"]) for r in TH], ":", color="#222222", lw=1.6, marker="s", ms=3.4, label="theory: one max channel (Prop. 1)")
ax.set_xlabel("AUROC of response length alone against the label"); ax.set_ylabel("Out-of-fold AUROC"); ax.set_xlim(.48, .82); ax.set_ylim(.4, 1.02)
ax.legend(frameon=False, fontsize=7.2, loc="upper left")
fig.tight_layout(); fig.savefig(OUT + "fig9_length.png"); plt.close()
print("fig9 done")
