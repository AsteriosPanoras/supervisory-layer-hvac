"""p272: how fast the learning source pays on an unknown building.  Post hoc, descriptive, frozen blind runs; nothing is run.
Arms on the blind set (all start every episode knowing nothing about the building): G30 (comparator), ROSFIX (fixed start: the start rule
with the time constant frozen at 1 h), ROS (learned start), SUP (learned start + pressure reset).  For day d = 1..30 (288 steps each; all
blind windows start on a Sunday): the cumulative pooled gain over the 10 composite architectures, G(d) = 100 (1 - sum_{t <= d} L_arm /
sum_{t <= d} L_G30), with L = |reward| per step; the gain of each week alone; and the learned-minus-fixed difference.  Intervals: the 10
composite architectures resampled 3,000 times (seed 1), all episodes of an architecture kept.  Gate: G(30) must equal the printed
results (fixed 0.31, learned 1.25, layer 2.82 %).  Loading and filters as p254/p267.  usage: python -I p272_learning_curve.py <out_dir>"""
import sys, os, csv, glob, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
H = os.path.dirname(os.path.abspath(__file__)); N = os.path.dirname(H); OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
MAN = [f"{N}/k07x3run/K07X3_RUN_MANIFEST.csv", f"{N}/k07x3run/K07X3_RUN_MANIFEST_S2.csv", f"{N}/k07x3h/K07X3H_RUN_MANIFEST.csv"]
man = [r for f in MAN for r in csv.DictReader(open(f))]; key2 = {(r["architecture"], r["weather"], r["policy_key"]): r for r in man}
grp = {r["architecture"]: r["group"] for r in man}
ARMS = ("G30", "ROSFIX", "ROS", "SUP"); R, start = {}, {}
for d in ("results_k07x3", "results_k07x3h"):
    for f in glob.glob(f"{N}/{d}/**/*.summary.json", recursive=True):
        s = json.load(open(f)); r = key2.get((s["building"], s["weather"], s["policy_key"]))
        if not r or r["policy"] not in ARMS or s["status"] != "OK" or s["steps_done"] != 8640 or s.get("illegal_actions", 0): continue
        k = (r["architecture"], r["weather"], r["policy"])
        if k in R: raise SystemExit(f"duplicate {k}")
        R[k] = np.abs(np.load(f.replace(".summary.json", ".npz"), allow_pickle=True)["rewards"].astype(float)); start[(k[0], k[1])] = s.get("sim_start")
cells = sorted({(a, w) for a, w, p in R if all((a, w, q) in R for q in ARMS)}); assert len(cells) == 36, len(cells)
comp = [c for c in cells if grp[c[0]] == "composite"]; archs = sorted({a for a, w in comp}); assert len(comp) == 30 and len(archs) == 10
import datetime as dt
wd = {dt.datetime.fromisoformat(str(start[c])).strftime("%A") for c in comp}
DAY = {p: np.array([[R[(a, w, p)][288 * i:288 * (i + 1)].sum() for i in range(30)] for a, w in comp]) for p in ARMS}   # episodes x days
idx = {a: [i for i, (x, w) in enumerate(comp) if x == a] for a in archs}
def cum_gain(p, rows):
    num = DAY[p][rows].sum(0).cumsum(); den = DAY["G30"][rows].sum(0).cumsum(); return 100 * (1 - num / den)
def week_gain(p, rows, d0, d1):
    return 100 * (1 - DAY[p][rows][:, d0:d1].sum() / DAY["G30"][rows][:, d0:d1].sum())
allrows = list(range(len(comp)))
G = {p: cum_gain(p, allrows) for p in ARMS[1:]}
for p, v in (("ROSFIX", "0.31"), ("ROS", "1.25"), ("SUP", "2.82")):
    assert f"{G[p][-1]:.2f}" == v, (p, G[p][-1])
rng = np.random.default_rng(1); B = []
for _ in range(3000):
    pick = rng.integers(0, 10, 10); rows = [i for j in pick for i in idx[archs[j]]]
    B.append({p: cum_gain(p, rows) for p in ARMS[1:]})
ci = {p: (np.percentile([b[p] for b in B], 2.5, axis=0), np.percentile([b[p] for b in B], 97.5, axis=0)) for p in ARMS[1:]}
diff = G["ROS"] - G["ROSFIX"]; dci = (np.percentile([b["ROS"] - b["ROSFIX"] for b in B], 2.5, axis=0), np.percentile([b["ROS"] - b["ROSFIX"] for b in B], 97.5, axis=0))
weeks = [(0, 7), (7, 14), (14, 21), (21, 28), (28, 30)]
W = {p: [week_gain(p, allrows, a, b) for a, b in weeks] for p in ARMS[1:]}
first_pos = {p: next((d + 1 for d in range(30) if ci[p][0][d] > 0), None) for p in ARMS[1:]}
first_diff_pos = next((d + 1 for d in range(30) if dci[0][d] > 0), None)
res = {"note": "post hoc, descriptive; blind set, 10 composite architectures (30 episodes); windows start on " + "/".join(sorted(wd)),
       "cumulative_gain_pct": {p: [round(float(x), 3) for x in G[p]] for p in G},
       "ci95_low": {p: [round(float(x), 3) for x in ci[p][0]] for p in G}, "ci95_high": {p: [round(float(x), 3) for x in ci[p][1]] for p in G},
       "learned_minus_fixed_cum_pp": [round(float(x), 3) for x in diff], "learned_minus_fixed_ci95": [[round(float(x), 3) for x in dci[0]], [round(float(x), 3) for x in dci[1]]],
       "gain_by_week_pct": {p: [round(float(x), 3) for x in W[p]] for p in W}, "weeks": ["days 1-7", "8-14", "15-21", "22-28", "29-30"],
       "first_day_lower_ci_above_0": first_pos, "first_day_learned_minus_fixed_lower_ci_above_0": first_diff_pos}
json.dump(res, open(f"{OUT}/P272_LEARNING_CURVE.json", "w"), indent=1)
# figure at column width, the paper's palette
BLUE, ORANGE, AQUA, GRAY, INK2, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "axes.labelsize": 8.5, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight"})
fig, (a, b) = plt.subplots(1, 2, figsize=(5.6, 2.35), gridspec_kw={"width_ratios": [1.35, 1]})
x = np.arange(1, 31)
for p, col, lab in (("SUP", AQUA, "learned start + pressure reset"), ("ROS", BLUE, "learned start"), ("ROSFIX", GRAY, "fixed start")):
    a.fill_between(x, ci[p][0], ci[p][1], color=col, alpha=0.15, lw=0); a.plot(x, G[p], color=col, lw=1.4, label=lab)
for d0 in (1, 8, 15, 22, 29):
    a.axvspan(d0 - 0.5, d0 + 0.5, color=GRID, alpha=0.6, lw=0); a.axvspan(d0 + 5.5, d0 + 6.5, color=GRID, alpha=0.6, lw=0) if d0 + 6 <= 30 else None
a.axhline(0, color=INK2, lw=0.6); a.set_xlim(0.5, 30.5); a.set_xticks([1, 7, 14, 21, 28]); a.set_xlabel("day of the episode"); a.set_ylabel("cumulative gain vs. G30 (%)")
a.legend(fontsize=6.5, loc="upper left"); a.set_title("(a) cumulative gain", fontsize=8.5, loc="left")
wl = ["1", "2", "3", "4"]; xx = np.arange(4); w_ = 0.27
for j, (p, col) in enumerate((("ROSFIX", GRAY), ("ROS", BLUE), ("SUP", AQUA))):
    b.bar(xx + (j - 1) * w_, W[p][:4], width=w_, color=col)
b.axhline(0, color=INK2, lw=0.6); b.set_xticks(xx, wl); b.set_xlabel("week"); b.set_ylabel("gain vs. G30 (%)"); b.set_title("(b) gain of each week", fontsize=8.5, loc="left")
fig.tight_layout(w_pad=1.5); fig.savefig(f"{OUT}/fig_learning_curve_col.pdf"); fig.savefig(f"{OUT}/fig_learning_curve_col.png", dpi=200); plt.close(fig)
print("windows start on:", sorted(wd))
for p in ("ROSFIX", "ROS", "SUP"):
    print(f"{p:7s} cum gain day 7 {G[p][6]:+.2f} [{ci[p][0][6]:+.2f},{ci[p][1][6]:+.2f}] | day 14 {G[p][13]:+.2f} | day 21 {G[p][20]:+.2f} | day 30 {G[p][29]:+.2f} [{ci[p][0][29]:+.2f},{ci[p][1][29]:+.2f}] | by week {[round(v, 2) for v in W[p]]} | first day CI>0: {first_pos[p]}")
print(f"learned - fixed (pp, cumulative): day 7 {diff[6]:+.2f} [{dci[0][6]:+.2f},{dci[1][6]:+.2f}] | day 14 {diff[13]:+.2f} [{dci[0][13]:+.2f},{dci[1][13]:+.2f}] | day 30 {diff[29]:+.2f} [{dci[0][29]:+.2f},{dci[1][29]:+.2f}] | first day CI>0: {first_diff_pos}")
