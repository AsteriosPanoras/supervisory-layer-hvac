"""[make_fig_profiles_col3.py = col2 without panel (a), whose numbers Table 2 shows directly above; the two regime panels become (a) and (b), 1.9 in high; output fig_profiles_col3.pdf]
[make_fig_profiles_col2.py = make_fig_profiles_col.py 2.05 in high instead of 2.4 (v49_31, 8-page limit); output fig_profiles_col2.pdf]
[make_fig_profiles_col.py = make_fig_profiles.py drawn at the 5.5-in column of the official template, so the letters print at
the sizes set here instead of 90 % of them (v49_22); output fig_profiles_col.pdf]
Figure 'profile source' for v49_16.
Panel (a) is panel (a) of the co-author's fig_main_2021; panels (b) and (c) are the two panels of his fig_regimes; panel (b) of
fig_main_2021 (gain versus profile changes) is left out because Table 2 gives the changes.  Style constants, numbers and labels are
copied from his make_figures.py (v47 Overleaf zip, read in full on 11 Nov); the numbers are checked against the paper text below.
Changes of style for the merge: each policy keeps one colour in all panels (PS5 blue, PD5 orange, MG aqua, the others grey);
panel (a) labels policies by code (Table 2, directly above, names them); shorter panel titles (class counts go to the caption).
usage: python -I make_fig_profiles.py MAIN_TEX OUTDIR"""
import sys, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TEX, OUT = sys.argv[1], sys.argv[2]
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY, INK, INK2, GRID = "#8a8984", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5, "axes.labelsize": 8.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight",
})
def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.6); ax.set_axisbelow(True)

# ---- numbers (a co-author's make_figures.py, from the v43 tables) ----
pol  = ["G30 (tuned fixed)", "PS5 (select once)", "TRIG2 (persistence)", "PD5 (daily)", "MG (one-time gate)"]
gain = [3.237, 3.686, 3.963, 4.132, 4.467]
lo   = [2.461, 2.560, 2.871, 3.017, 3.316]
hi   = [4.111, 4.835, 5.198, 5.384, 5.713]
col  = [GRAY, BLUE, GRAY, ORANGE, AQUA]
cls = ["Cold", "Mild", "Hot"]
d23 = {"Select once (PS5)": [4.17, 1.59, 2.42], "Daily (PD5)": [4.32, 6.07, 2.70]}
d21 = {"Select once (PS5)": [4.96, 1.81, 2.71], "Daily (PD5)": [4.63, 4.40, 2.44],
       "One-time gate (MG)": [5.08, 4.41, 2.89]}
cmap = {"Select once (PS5)": BLUE, "Daily (PD5)": ORANGE, "One-time gate (MG)": AQUA}

# ---- check against the paper: Table 2 (rounded to 2 decimals) and the regime paragraph ----
import os
# col2: since v49_28 the 2021 class gains to two decimals are in Appendix E, so the check reads the main text and the appendix together
tex = " ".join((open(TEX).read() + open(os.path.join(os.path.dirname(TEX), "appendices.tex")).read()).split())
from decimal import Decimal, ROUND_HALF_UP
r2 = lambda v: str(Decimal(str(v)).quantize(Decimal("0.01"), ROUND_HALF_UP))  # the table rounds half up (4.835 -> 4.84)
for g, l, h in zip(gain, lo, hi):
    s = f"$+{r2(g)}$" if g != 4.467 else "$\\mathbf{+4.47}$"
    assert s in tex and f"[+{r2(l)}, +{r2(h)}]" in tex, (g, l, h)
for v in (d23["Daily (PD5)"][1], d23["Select once (PS5)"][1], d21["Daily (PD5)"][1], d21["Select once (PS5)"][1],
          *d21["One-time gate (MG)"], d21["Daily (PD5)"][0], d21["Select once (PS5)"][0], d23["Daily (PD5)"][0], d23["Select once (PS5)"][0],
          d21["Daily (PD5)"][2], d21["Select once (PS5)"][2], d23["Daily (PD5)"][2], d23["Select once (PS5)"][2]):
    assert f"{v:.2f}\\%" in tex or f"{v:.2f}," in tex or f"{v:.2f} " in tex, v

fig = plt.figure(figsize=(5.82, 1.9))
gs = fig.add_gridspec(1, 2, width_ratios=[0.86, 1.14], wspace=0.08)
b = fig.add_subplot(gs[0]); c = fig.add_subplot(gs[1], sharey=b)

# panel (a) of col2 removed: Table 2 gives the same gains and intervals

# (b), (c) gains by initial class (a co-author's panels)
for ax, data, title in [(b, d23, "(a) 2023 weather"),
                        (c, d21, "(b) 2021 weather")]:
    k = len(data); w = 0.8 / 3; x = np.arange(3)
    for j, (name, v) in enumerate(data.items()):
        pos = x + (j - (k - 1) / 2) * (w + 0.02)
        ax.bar(pos, v, w, color=cmap[name], edgecolor="white", lw=1.0, label=name)
        for p, val in zip(pos, v):
            ax.text(p, val + 0.12, f"{val:.1f}", ha="center", fontsize=6.0, color=INK2)
    ax.set_xticks(x, cls); ax.axhline(0, color=INK2, lw=0.6); grid(ax)
    ax.set_title(title, fontsize=8.5, loc="left")
b.set_ylabel("Gain vs. B (%)"); b.set_ylim(0, 7)
plt.setp(c.get_yticklabels(), visible=False)
h, l = c.get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.55, 1.12), fontsize=7.5, handlelength=1.0, columnspacing=0.9, handletextpad=0.4)
fig.savefig(f"{OUT}/fig_profiles_col3.pdf"); fig.savefig(f"{OUT}/fig_profiles_col3.png", dpi=200); plt.close(fig)
print("ok: numbers match the paper text")
