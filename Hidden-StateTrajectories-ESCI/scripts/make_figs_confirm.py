import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = "/home/claude/work/figs/"
R = json.load(open(OUT + "confirmatory_results.json"))
plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.8, "savefig.dpi": 300})
W1 = 3.45
COL = {"F": "#0072B2", "Llast": "#009E73", "L": "#56B4E9", "T": "#CC79A7", "A": "#D55E00", "len": "#7f7f7f"}
keys = [("F_final_state", "F", "F  final state (static)"), ("L_layer_axis", "L", "L  layer axis"),
        ("A_full_RTM", "A", "A  full RTM"), ("Llast_B4style", "Llast", "L$_{last}$  final token"),
        ("T_token_axis", "T", "T  token axis"), ("len_only", "len", "len  length only")]
# ---- Fig 1: confirmatory result
fig, (a, b) = plt.subplots(2, 1, figsize=(W1, 3.0), gridspec_kw={"height_ratios": [1.35, 1]})
for i, (k, c, lab) in enumerate(keys):
    d = R["detectors"][k]; lo, hi = d["ci"]
    a.plot([lo, hi], [i, i], color=COL[c], lw=2.2, solid_capstyle="butt")
    a.plot(d["auroc"], i, "o", color=COL[c], mec="white", mew=.8, ms=5.5 if c == "A" else 4.5, zorder=3)
    a.text(.985, i, f"{d['auroc']:.3f}", fontsize=7, va="center", ha="right", transform=a.get_yaxis_transform())
a.axvline(.5, color="k", ls=":", lw=.9); a.set_yticks(range(6)); a.set_yticklabels([k[2] for k in keys], fontsize=7.4)
a.invert_yaxis(); a.set_xlim(.45, .86); a.tick_params(axis="x", labelsize=7.6); a.set_ylim(5.6, -.6)
a.set_xlabel("Out-of-fold AUROC (95% CI), 199 reliable / 900", fontsize=7.8, labelpad=1)
a.text(.012, .97, "(a)", transform=a.transAxes, fontsize=9, fontweight="bold", va="top")
pk = [("F_final_state", "A − F"), ("L_layer_axis", "A − L"), ("Llast_B4style", "A − L$_{last}$"), ("T_token_axis", "A − T"), ("len_only", "A − len")]
for i, (k, lab) in enumerate(pk):
    d = R["paired_A_minus"][k]; lo, hi = d["ci"]; sig = d["p_holm"] < .05
    c = "#D55E00" if sig else "#555555"
    b.plot([lo, hi], [i, i], color=c, lw=2.2, solid_capstyle="butt")
    b.plot(d["diff"], i, "o", color="white" if not sig else c, mec=c, mew=1.6, ms=4.8, zorder=3)
    ph = d["p_holm"]; s = f"p$_{{Holm}}$ = {ph:.2f}" if ph >= .001 else "p$_{Holm}$ < 0.001"
    b.text(.985, i, s, fontsize=6.6, va="center", ha="right", transform=b.get_yaxis_transform())
b.axvline(0, color="k", lw=.9); b.axvline(.05, color="k", ls="--", lw=.7)
b.set_yticks(range(5)); b.set_yticklabels([p[1] for p in pk], fontsize=7.4); b.invert_yaxis()
b.set_xlim(-.1, .44); b.set_ylim(4.6, -.6); b.tick_params(axis="x", labelsize=7.6)
b.set_xlabel("Paired AUROC difference (95% CI); dashed = 0.05", fontsize=7.8, labelpad=1)
b.text(.012, .97, "(b)", transform=b.transAxes, fontsize=9, fontweight="bold", va="top")
fig.tight_layout(h_pad=.6, pad=.3); fig.savefig(OUT + "fig1_hero.png"); plt.close()
# ---- reversal figure
fig, ax = plt.subplots(figsize=(W1, 1.9))
rows = [("Pilot (200 q, 25 reliable)", .571, .776, "static B2 (untuned)", "RTM"),
        ("Confirmatory (900 q, 199 reliable)", .753, .732, "static F (tuned)", "RTM (A)")]
for i, (lab, s, r, ls, lr) in enumerate(rows):
    ax.plot([s, r], [i, i], color="#999999", lw=1.4, zorder=1)
    ax.plot(s, i, "o", color="#0072B2", ms=6, mec="white", zorder=3); ax.plot(r, i, "o", color="#D55E00", ms=6, mec="white", zorder=3)
    lo_, hi_ = sorted([(s, "#0072B2"), (r, "#D55E00")])
    ax.text(lo_[0] - .008, i - .25, f"{lo_[0]:.3f}", ha="right", fontsize=7, color=lo_[1])
    ax.text(hi_[0] + .008, i - .25, f"{hi_[0]:.3f}", ha="left", fontsize=7, color=hi_[1])
ax.axvline(.5, color="k", ls=":", lw=.9)
ax.set_yticks([0, 1]); ax.set_yticklabels([r[0] for r in rows], fontsize=7.4); ax.set_ylim(1.7, -.7); ax.set_xlim(.45, .85)
ax.set_xlabel("AUROC", fontsize=8, labelpad=1); ax.tick_params(axis="x", labelsize=7.6)
from matplotlib.lines import Line2D
ax.legend([Line2D([], [], marker="o", ls="", color="#0072B2"), Line2D([], [], marker="o", ls="", color="#D55E00")],
          ["static probe", "trajectory (RTM)"], frameon=False, fontsize=7, loc="lower right", ncol=1)
fig.tight_layout(pad=.3); fig.savefig(OUT + "fig_reversal.png"); plt.close()
import json as j
from PIL import Image
import glob, os
j.dump({os.path.basename(f)[:-4]: list(Image.open(f).size) for f in glob.glob(OUT + "*.png")}, open(OUT + "dims.json", "w"))
