"""Figures from the results reported in SB26 v43 (Tables 1, 2, 11, 12, 18; Sec. 4.2)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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

# ---------- Figure 1: 2021 main cohort (n = 98) ----------
pol  = ["G30 (tuned fixed)", "PS5 (select once)", "TRIG2 (persistence)", "PD5 (daily)", "MG (one-time gate)"]
gain = [3.237, 3.686, 3.963, 4.132, 4.467]
lo   = [2.461, 2.560, 2.871, 3.017, 3.316]
hi   = [4.111, 4.835, 5.198, 5.384, 5.713]
chg  = [0, 0, 2.33, 4.72, 2.19]
col  = [GRAY, GRAY, GRAY, GRAY, AQUA]

fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 2.5), gridspec_kw={"width_ratios": [1.15, 1]})
y = np.arange(len(pol))
for i in y:
    a.plot([lo[i], hi[i]], [i, i], color=col[i], lw=1.6, solid_capstyle="round")
    a.plot(gain[i], i, "o", ms=6, color=col[i], mec="white", mew=1.2, zorder=3)
    a.text(hi[i] + 0.08, i, f"{gain[i]:+.2f}", va="center", fontsize=7.5, color=INK2)
a.set_yticks(y, pol); a.set_xlim(2.2, 6.4); a.axvline(0, color=INK2, lw=0.6)
a.set_xlabel("Gain vs. fixed baseline B (%)"); grid(a, "x")
a.set_title("(a) Gain vs. B with 95% CI", fontsize=8.5, loc="left")

dyn = [1, 2, 3, 4]
for i in dyn:
    b.plot(chg[i], gain[i], "o", ms=7, color=col[i], mec="white", mew=1.2, zorder=3)
lab = {1: ("PS5", (6, -3)), 2: ("TRIG2", (6, -9)), 3: ("PD5", (-8, 6)), 4: ("MG", (6, 3))}
for i, (t, off) in lab.items():
    b.annotate(t, (chg[i], gain[i]), xytext=off, textcoords="offset points", fontsize=7.5, color=INK,
               ha="right" if off[0] < 0 else "left")
b.set_xlim(-0.4, 5.4); b.set_ylim(3.4, 4.7)
b.set_xlabel("Realised post-initial profile changes / episode"); b.set_ylabel("Gain vs. B (%)")
grid(b, "both"); b.set_title("(b) Gain vs. reassignment frequency", fontsize=8.5, loc="left")
fig.tight_layout(w_pad=2.0)
fig.savefig("figures/fig_main_2021.pdf"); fig.savefig("figures/fig_main_2021.png", dpi=200); plt.close(fig)

# ---------- Figure 2: gains by initial weather regime ----------
cls = ["Cold", "Mild", "Hot"]
d23 = {"Select once (PS5)": [4.17, 1.59, 2.42], "Daily (PD5)": [4.32, 6.07, 2.70]}
d21 = {"Select once (PS5)": [4.96, 1.81, 2.71], "Daily (PD5)": [4.63, 4.40, 2.44],
       "One-time gate (MG)": [5.08, 4.41, 2.89]}
cmap = {"Select once (PS5)": BLUE, "Daily (PD5)": ORANGE, "One-time gate (MG)": AQUA}
fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.4), sharey=True)
for ax, data, title in [(axs[0], d23, "(a) 2023 weather, n = 70 (23/24/23)"),
                        (axs[1], d21, "(b) 2021 weather, n = 98 (30/42/26)")]:
    k = len(data); w = 0.8 / 3; x = np.arange(3)
    for j, (name, v) in enumerate(data.items()):
        pos = x + (j - (k - 1) / 2) * (w + 0.02)
        ax.bar(pos, v, w, color=cmap[name], edgecolor="white", lw=1.0, label=name)
        for p, val in zip(pos, v):
            ax.text(p, val + 0.12, f"{val:.1f}", ha="center", fontsize=6.5, color=INK2)
    ax.set_xticks(x, cls); ax.axhline(0, color=INK2, lw=0.6); grid(ax)
    ax.set_title(title, fontsize=8.5, loc="left")
axs[0].set_ylabel("Gain vs. B (%)"); axs[0].set_ylim(0, 7)
h, l = axs[1].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.07), fontsize=7.5)
fig.tight_layout()
fig.savefig("figures/fig_regimes.pdf"); fig.savefig("figures/fig_regimes.png", dpi=200); plt.close(fig)

# ---------- Figure 3: primary contrast, sensitivities, secondary contrasts ----------
rows = [  # label, est, lo, hi, group
    ("MG $-$ TRIG2, main (n = 98) [primary]", 0.504, 0.337, 0.704, "p"),
    ("  reference-reweighted (n = 98)",        0.504, 0.333, 0.706, "s"),
    ("  previously evaluated variants (n = 50)", 0.425, 0.227, 0.671, "s"),
    ("  newly generated variants (n = 48)",    0.598, 0.306, 0.921, "s"),
    ("  design-class reconstruction (n = 98)", 0.398, 0.183, 0.613, "s"),
    ("  main + extension pooled (n = 118)",    0.407, 0.210, 0.633, "s"),
    ("  optional extension only (n = 20)",    -0.004, None, None, "s"),
    ("MG $-$ PD5 (secondary)",  0.335, 0.200, 0.487, "c"),
    ("MG $-$ PS5 (secondary)",  0.781, 0.441, 1.236, "c"),
    ("MG $-$ G30 (secondary)",  1.230, 0.185, 2.325, "c"),
    ("PD5 $-$ PS5 (secondary)", 0.446, 0.070, 0.937, "c"),
]
gcol = {"p": AQUA, "s": GRAY, "c": BLUE}
fig, ax = plt.subplots(figsize=(6.8, 3.0))
ypos = []; yy = 0
for i, r in enumerate(rows):
    if i in (1, 7): yy += 0.6
    ypos.append(yy); yy += 1
ypos = -np.array(ypos)
for (lab_, e, l, h, g), yv in zip(rows, ypos):
    if l is not None:
        ax.plot([l, h], [yv, yv], color=gcol[g], lw=1.6, solid_capstyle="round")
    if l is not None:
        ax.plot(e, yv, "D" if g == "p" else "o", ms=6 if g == "p" else 5, color=gcol[g],
                mec="white", mew=1.0, zorder=3)
    else:
        ax.plot(e, yv, "o", ms=5.5, mfc="white", mec=INK2, mew=1.2, zorder=4)
    txt = f"{e:+.3f}" + ("" if l is not None else " (no CI)")
    ax.text(1.02, yv, txt, va="center", fontsize=7.2, color=INK2,
            transform=ax.get_yaxis_transform())
ax.set_yticks(ypos, [r[0] for r in rows]); ax.axvline(0, color=INK2, lw=0.7, ls="--", zorder=1)
ax.set_xlim(-0.3, 2.5); ax.set_xlabel("Difference in baseline-referenced gain (percentage points)")
grid(ax, "x")
fig.tight_layout()
fig.savefig("figures/fig_contrasts.pdf"); fig.savefig("figures/fig_contrasts.png", dpi=200); plt.close(fig)
print("ok")
