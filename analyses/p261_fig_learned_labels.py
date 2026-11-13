"""p261: Figure 'what the layer learned' (Appendix J), identical to figure 3 of paper_v48/v48/make_figures_v48.py except the labels in the paper's common language
('tuned static' -> 'G30', 'pressure overrides' -> 'pressure reset', 'scene' -> 'episode').  Data: out/PAPER_ANALYSES_2.json (p253).  usage: python -I p261_fig_learned_labels.py <out_dir>"""
import json, os, sys
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

H = os.path.dirname(os.path.abspath(__file__)); OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
A = json.load(open(f"{H}/out/PAPER_ANALYSES_2.json"))

# ---------- Figure 3: what the layer learned (a) and where it acted (b) ----------
fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 2.45), gridspec_kw={"width_ratios": [1.15, 1]})
sc = A["A_learned_tau"]["SUP"]["per_scene"]
arch = sorted({k.split("|")[0] for k in sc}, key=lambda x: np.mean([v["tau_final_h"] for kk, v in sc.items() if kk.startswith(x)]))
for i, ar in enumerate(arch):
    t = [v["tau_final_h"] for k, v in sc.items() if k.startswith(ar)]
    a.plot([i] * len(t), t, "o", ms=3.5, color=BLUE, alpha=0.55, mec="none")
    a.plot(i, np.mean(t), "_", ms=11, mew=2.0, color=INK)
a.set_xticks(range(len(arch)), [x.replace("building_", "") for x in arch], rotation=60, fontsize=6.5)
a.set_ylabel(r"learned $\tau$ (h)"); a.set_ylim(0, 2.4); grid(a)
a.set_title("(a) Learned recovery constant per architecture", fontsize=8.5, loc="left")
cm = {"C": BLUE, "M": ORANGE, "H": AQUA}
for r in A["B_activity_vs_gain"]["per_scene"]:
    inact = r["los_extensions"] == 0 and r["tr_override_share"] == 0.0
    b.plot(100 * r["tr_override_share"], r["gain_SUP_vs_G30_pct"], "o", ms=4.2, color="white" if inact else cm[r["class"]],
           mec=INK2 if inact else "white", mew=0.8, zorder=3)
for c, n in (("C", "cold start"), ("M", "mild start"), ("H", "hot start")):
    b.plot([], [], "o", ms=4.2, color=cm[c], mec="white", label=n)
b.plot([], [], "o", ms=4.2, color="white", mec=INK2, label="never acted")
b.axhline(0, color=INK2, lw=0.6); b.set_xlabel("pressure reset (% of occupied steps)"); b.set_ylabel("gain vs. G30 (%)")
b.legend(fontsize=6.5, loc="upper right", handletextpad=0.2); grid(b, "both")
b.set_title("(b) Gain per episode vs. activity", fontsize=8.5, loc="left")
fig.tight_layout(w_pad=1.5); fig.savefig(f"{OUT}/fig_learned.pdf"); fig.savefig(f"{OUT}/fig_learned.png", dpi=200); plt.close(fig)
print("fig_learned written")
