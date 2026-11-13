"""[p262 = p259 + the 36 episodes of the registered test on ZT Best (RESULT_HEADROOM_ZT.json, p258) in panel (a); nothing else changed]
Figure 'When does the layer help?' (post hoc, descriptive; frozen, already-read runs).  usage: python -I p259_when_figure.py <out_dir>
(a) per-episode gain of the layer over K07-G30 against the comfort share of G30's loss on the 36 blind episodes (from RESULT_HEADROOM_K07.json, p258), the pooled gain
    of each third, the range of the same share on the six official training scenes (K07-G30 runs of the unoccupied-setup stage) and Penguin's mean share on the blind set;
(b) pooled composite gain of the layer (vs G30), of tuning (G30 vs B) and of Penguin's learned precooling (vs Penguin) when the same runs are re-weighted with comfort
    weight w and energy = carbon = (1 - w) / 2; w = 0.6 is the competition.  Checked against WEIGHT_SENSITIVITY.json (w = 0.6, 1/3, 0.2) to 0.01 pp."""
import sys, os, csv, glob, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
H = os.path.dirname(os.path.abspath(__file__)); N = os.path.dirname(H); OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, AQUA, GRAY, INK, INK2, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "axes.labelsize": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight"})
CK = ("normalized_productivity_regret", "normalized_energy_cost", "normalized_carbon_emission")
def comps(f):
    z = np.load(f.replace(".summary.json", ".npz"), allow_pickle=True); ck = [str(x) for x in z["comp_keys"]]
    return np.array([np.abs(z["comps"][:, ck.index(k)].astype(float)).mean() for k in CK])
share = lambda c: 0.6 * c[0] / (0.6 * c[0] + 0.2 * c[1] + 0.2 * c[2])
R = json.load(open(f"{H}/out/RESULT_HEADROOM_K07.json")); HZ = json.load(open(f"{N}/ztsup/RESULT_HEADROOM_ZT.json")); WS = json.load(open(f"{H}/out/WEIGHT_SENSITIVITY.json"))
# blind-set runs (composite and OOD) of B, G30, SUP, Penguin V0 and PC
man = [r for f in (f"{N}/k07x3run/K07X3_RUN_MANIFEST.csv", f"{N}/k07x3run/K07X3_RUN_MANIFEST_S2.csv", f"{N}/k07x3h/K07X3H_RUN_MANIFEST.csv", f"{N}/pgf/PGF_RUN_MANIFEST.csv") for r in csv.DictReader(open(f))]
key2 = {(r["architecture"], r["weather"], r["policy_key"]): r for r in man}; grp = {r["architecture"]: r["group"] for r in man}
U = {}
for d in ("results_k07x3", "results_k07x3h", "results_pgf"):
    for f in glob.glob(f"{N}/{d}/**/*.summary.json", recursive=True):
        s = json.load(open(f)); r = key2.get((s["building"], s["weather"], s["policy_key"]))
        if not r or r["policy"] not in ("B", "G30", "SUP", "V0", "PC") or s["status"] != "OK" or s["steps_done"] != 8640: continue
        k = (r["architecture"], r["weather"], r["policy"])
        if k in U: raise SystemExit(f"duplicate {k}")
        U[k] = comps(f)
cells = sorted({(a, w) for a, w, p in U}); comp_cells = [c for c in cells if grp[c[0]] == "composite"]
peng_share = float(np.mean([share(U[(*c, "V0")]) for c in cells if (*c, "V0") in U])); n_peng = sum(1 for c in cells if (*c, "V0") in U)
# official training scenes: K07-G30 runs of the unoccupied-setup stage (competition tracker kit)
G = "RAMP=0.3548,COLD_SAT=-1.0,KI=0.0068,KP=0.0699,HEAT_U_SCALE=0.8747,PRESTART_H=2,EVENING_STOP_H=0,NIGHT_HEAT_DROP_K=3.5886,MILD_SHIFT_K=0.7701,WATER_SCALE=0.9907,MIN_DAMPER=0.0"
off = {}
for f in glob.glob(f"{N}/k07uns/laptop/work_off/out/**/*.summary.json", recursive=True):
    s = json.load(open(f))
    if s["policy_key"] == f"k07@{G}" and s["status"] == "OK": off[f"{s['building']}|{s['weather']}"] = share(comps(f))
assert len(off) == 6, off
def gain_w(x, y, w):
    wt = np.array([w, (1 - w) / 2, (1 - w) / 2]); cs = [c for c in comp_cells if (*c, x) in U and (*c, y) in U]
    return 100 * (1 - sum(U[(*c, x)] @ wt for c in cs) / sum(U[(*c, y)] @ wt for c in cs))
CON = (("layer on K07-G30", "SUP", "G30", BLUE, "-"), ("tuning K07 (G30 vs. B)", "G30", "B", ORANGE, "--"), ("learned precooling on Penguin", "PC", "V0", AQUA, "-."))
for lab, w in (("competition (0.6 comfort, 0.2 energy, 0.2 carbon)", 0.6), ("equal (1/3 each)", 1 / 3), ("energy-heavy (0.2, 0.4, 0.4)", 0.2)):
    for k, (x, y) in (("layer vs G30", ("SUP", "G30")), ("G30 vs B", ("G30", "B")), ("Penguin+PC vs Penguin", ("PC", "V0"))):
        assert abs(gain_w(x, y, w) - WS[lab][k]) < 0.01, (lab, k, gain_w(x, y, w), WS[lab][k])
ws = np.linspace(0, 1, 101); curves = {lab: [float(gain_w(x, y, w)) for w in ws] for lab, x, y, _, _ in CON}
zero = {lab: (float(ws[int(np.argmax(np.diff(np.sign(v)) != 0))]) if np.any(np.diff(np.sign(v)) != 0) else None) for lab, v in curves.items()}   # first sign change; None = never crosses zero
data = {"note": "post hoc, descriptive; frozen already-read runs", "official_scene_comfort_share": off, "penguin_mean_comfort_share_blind": peng_share, "penguin_episodes": n_peng,
        "headroom_k07": {k: R[k] for k in ("episodes", "spearman_gain_vs_share", "p_one_sided_perm", "thirds_by_share", "mean_share")},
        "weight_curve": {"w_comfort": ws.tolist(), **curves}, "w_where_gain_is_closest_to_zero": zero}
json.dump(data, open(f"{OUT}/WHEN_FIGURE_DATA.json", "w"), indent=1)
# ---- figure ----
fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 2.6), gridspec_kw={"width_ratios": [1.25, 1]})
ep = R["per_episode"]; xs = np.array([100 * e["base_comfort_share"] for e in ep]); ys = np.array([e["gain_pct"] for e in ep]); oo = np.array([e["group"] == "ood" for e in ep])
lo_off, hi_off = 100 * min(off.values()), 100 * max(off.values())
a.axvspan(lo_off, hi_off, color=GRID, alpha=0.8, lw=0, zorder=0)
a.axhline(0, color=INK2, lw=0.6, zorder=1)
a.scatter(xs[~oo], ys[~oo], s=16, color=BLUE, edgecolor="white", linewidth=0.5, zorder=3, label="K07-G30")
a.scatter(xs[oo], ys[oo], s=18, marker="s", facecolor="white", edgecolor=BLUE, linewidth=0.9, zorder=3, label="K07-G30, OOD")
zx = np.array([100 * e["base_comfort_share"] for e in HZ["per_episode"]]); zy = np.array([e["gain_pct"] for e in HZ["per_episode"]])
a.scatter(zx, zy, s=18, marker="^", color=ORANGE, edgecolor="white", linewidth=0.5, zorder=3, label="ZT Best")
zo = np.argsort(zx, kind="stable"); m3 = len(zx) // 3
for i, tt in enumerate(HZ["thirds_by_share"]):
    idx = zo[i * m3:(i + 1) * m3] if i < 2 else zo[2 * m3:]; x0, x1 = zx[idx].min(), zx[idx].max()
    a.plot([x0, x1], [tt["pooled_gain_pct"]] * 2, color=ORANGE, lw=1.3, ls="--", solid_capstyle="round", zorder=4)
order = np.argsort(xs, kind="stable"); n3 = len(xs) // 3
for i, t in enumerate(R["thirds_by_share"]):
    idx = order[i * n3:(i + 1) * n3] if i < 2 else order[2 * n3:]; x0, x1 = xs[idx].min(), xs[idx].max()
    a.plot([x0, x1], [t["pooled_gain_pct"]] * 2, color=INK, lw=1.6, solid_capstyle="round", zorder=4)
    a.text(1.9 if i == 0 else (x0 + x1) / 2, t["pooled_gain_pct"] + 0.45, (f"{t['pooled_gain_pct']:+.2f}%" if abs(t["pooled_gain_pct"]) < 0.1 else f"{t['pooled_gain_pct']:+.1f}%"), ha="center", va="bottom", fontsize=6.8 if i == 0 else 7, color=INK, zorder=6,
           bbox=dict(fc="white", ec="none", pad=0.4, alpha=0.9) if i == 0 else None)  # p262: lowest label on a white ground, clear of the y axis
ytop = max(ys.max(), zy.max(), 1) * 1.10
a.text(hi_off + 0.6, ytop * 0.97, "official training\nscenes (leaderboard)", fontsize=6.5, color=INK2, va="top", ha="left")
yp = min(ys.min(), 0) - 0.9
a.plot([100 * peng_share], [yp], marker="^", ms=6, color=AQUA, mec="white", mew=0.5, zorder=5)
a.text(100 * peng_share + 0.9, yp, "Penguin, mean share", fontsize=6.5, color=INK2, va="center", ha="left")
a.set_ylim(min(ys.min(), zy.min(), 0) - 1.6, ytop); a.set_xlim(-3.5, max(xs.max(), zx.max()) * 1.04)
a.set_xlabel("comfort share of the loss of the controller beneath (%)"); a.set_ylabel("gain of the layer (%)")
a.grid(axis="y", color=GRID, lw=0.6); a.set_axisbelow(True); a.legend(fontsize=6.3, loc="lower right", handletextpad=0.3, borderaxespad=0.2)
a.set_title(f"(a) per blind episode: $\\rho$ = {R['spearman_gain_vs_share']:.2f} (K07-G30), {HZ['spearman_gain_vs_share']:.2f} (ZT Best)", fontsize=8.0, loc="left")
b.axhline(0, color=INK2, lw=0.6); b.axvline(0.6, color=GRAY, lw=0.8, ls=":")
for lab, x, y, col, ls in CON:
    b.plot(ws, curves[lab], color=col, lw=1.5, ls=ls, label=lab)
b.set_xlim(0, 0.8); vis = [v for lab in curves for wv, v in zip(ws, curves[lab]) if wv <= 0.8]; b.set_ylim(min(vis) - 0.8, max(vis) * 1.08)
b.text(0.605, b.get_ylim()[1] * 0.97, "competition", fontsize=6.5, color=INK2, va="top", ha="left")
for (lab, x, y, col, ls), short in zip(CON, ("layer", "tuning", "precooling")):
    b.text(0.70, curves[lab][70] + 0.55, short, fontsize=6.5, color=INK, ha="center", va="bottom")
b.set_xlabel("weight of comfort $w$ (energy = carbon = $(1-w)/2$)"); b.set_ylabel("pooled gain (%)")
b.grid(axis="y", color=GRID, lw=0.6); b.set_axisbelow(True); b.legend(fontsize=6.3, loc="upper left", handlelength=2.2, borderaxespad=0.2)
b.set_title("(b) the same runs, re-weighted", fontsize=8.5, loc="left")
fig.tight_layout(w_pad=1.4); fig.savefig(f"{OUT}/fig_when.pdf"); fig.savefig(f"{OUT}/fig_when.png", dpi=200); plt.close(fig)
print(json.dumps({"official_share_pct": {k: round(100 * v, 2) for k, v in off.items()}, "penguin_mean_share_pct": round(100 * peng_share, 2), "penguin_episodes": n_peng,
                  "gain_at_w": {lab: {f"{w:.2f}": round(curves[lab][int(round(w * 100))], 2) for w in (0.0, 0.2, 0.33, 0.6, 0.8, 1.0)} for lab in curves}, "zero_crossing_w": zero}, indent=1))
