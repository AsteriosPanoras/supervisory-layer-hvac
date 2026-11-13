"""p272b: redraw of the p272 learning-curve figure from P272_LEARNING_CURVE.json (nothing recomputed).  Each 'day' runs from 06:00 to 06:00 and
all windows start on a Sunday, so day 1 holds the first night's pre-start (Monday 02:00-05:00) before the first opening: every arm pays
that energy before any comfort return, and its cumulative gain on day 1 (about -14 %) dwarfed the rest of panel (a).  Panel (a) now starts
on day 2; the caption states day 1.  usage: python -I p272b_learning_curve_plot.py <p272_out_dir>"""
import sys, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = sys.argv[1]; R = json.load(open(f"{D}/P272_LEARNING_CURVE.json"))
G, LO, HI, W = R["cumulative_gain_pct"], R["ci95_low"], R["ci95_high"], R["gain_by_week_pct"]
BLUE, AQUA, GRAY, INK2, GRID = "#2a78d6", "#1baf7a", "#8a8984", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "axes.labelsize": 8.5, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight"})
fig, (a, b) = plt.subplots(1, 2, figsize=(5.6, 2.35), gridspec_kw={"width_ratios": [1.35, 1]})
x = np.arange(2, 31); s = slice(1, 30)
for d0 in (8, 15, 22, 29):
    a.axvspan(d0 - 0.5, d0 + 0.5, color=GRID, alpha=0.7, lw=0)
for d0 in (7, 14, 21, 28):
    a.axvspan(d0 - 0.5, d0 + 0.5, color=GRID, alpha=0.7, lw=0)
for p, col, lab in (("SUP", AQUA, "learned start + pressure reset"), ("ROS", BLUE, "learned start"), ("ROSFIX", GRAY, "fixed start")):
    a.fill_between(x, np.array(LO[p])[s], np.array(HI[p])[s], color=col, alpha=0.15, lw=0); a.plot(x, np.array(G[p])[s], color=col, lw=1.4, label=lab)
a.axhline(0, color=INK2, lw=0.6); a.set_xlim(1.5, 30.5); a.set_xticks([2, 7, 14, 21, 28]); a.set_xlabel("day of the episode (weekends shaded)")
a.set_ylabel("cumulative gain vs. G30 (%)"); a.set_title("(a) cumulative gain from day 2", fontsize=8.5, loc="left")
xx = np.arange(4); w_ = 0.27
for j, (p, col) in enumerate((("ROSFIX", GRAY), ("ROS", BLUE), ("SUP", AQUA))):
    b.bar(xx + (j - 1) * w_, W[p][:4], width=w_, color=col)
b.axhline(0, color=INK2, lw=0.6); b.set_xticks(xx, ["1", "2", "3", "4"]); b.set_xlabel("week"); b.set_ylabel("gain vs. G30 (%)")
b.set_title("(b) gain of each week", fontsize=8.5, loc="left")
fig.tight_layout(w_pad=1.5); h, l = a.get_legend_handles_labels()
fig.legend(h, l, fontsize=7, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.99), handlelength=1.8, columnspacing=1.6); fig.savefig(f"{D}/fig_learning_curve_col2.pdf"); fig.savefig(f"{D}/fig_learning_curve_col2.png", dpi=200); plt.close(fig)
print("day-1 cumulative gain:", {p: G[p][0] for p in G})
