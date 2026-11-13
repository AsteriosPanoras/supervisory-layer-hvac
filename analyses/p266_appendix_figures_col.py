"""p266: the five appendix figures at column width.  Same data and the same computations as before; only size and layout change, so that the text prints at the size of the
main-text figures (included at \\linewidth, ~1:1), plus the legibility defects seen at print size in v49_42:
 - error analysis: two rows, (a) and (b) above and (c) below, because the bar labels overlapped and the title of (b) ran into the
   y label of (c); bin labels typeset (<= -> ≤, >= -> ≥, a..b -> a to b, true minus), same bins and values;
 - learned: the legend of (b) in two columns in the empty upper right (it covered a point); more space between the panel titles;
 - case: the G30 zone line drawn above the layer's line, which hid it where the two coincide (same labels and legend order);
 - decisions: the y label in two lines (it was taller than the axis); contrasts: true minus in "-0.004".
Sources: contrasts = the rows of Figure 3 in the co-author's make_figures.py, parsed, not retyped; learned = p261 (out/PAPER_ANALYSES_2.json);
case, extremes and decisions = p256b (out/PAPER_ERROR_ANALYSIS.json and the frozen blind runs); the recomputed case-study JSON must equal
out/figs_v2/CASE_STUDY_WORST_DAY.json.  usage: python -I p266_appendix_figures_col.py <make_figures.py of the paper> <out_dir>"""
import sys, os, csv, glob, json, ast, re, hashlib, numpy as np, datetime as dt
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
H = os.path.dirname(os.path.abspath(__file__)); N = os.path.dirname(H); MF, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, AQUA, GRAY, INK, INK2, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "axes.labelsize": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight"})
W = 5.6   # figure width before the tight bounding box, as for the main-text figures (printed at \linewidth = 5.5 in)
def grid(ax, axis="y"): ax.grid(axis=axis, color=GRID, lw=0.6); ax.set_axisbelow(True)
def save(fig, name): fig.savefig(f"{OUT}/{name}.pdf"); fig.savefig(f"{OUT}/{name}.png", dpi=200); plt.close(fig)
sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()
MAN = {"inputs": {}}

# ---------- contrasts (Figure 3 of the co-author's make_figures.py; rows parsed) ----------
src = open(MF).read(); MAN["inputs"]["make_figures.py"] = sha(MF)
rows = ast.literal_eval("[" + re.search(r"rows = \[  # label, est, lo, hi, group\n(.*?)\n\]\n", src, re.S).group(1) + "]")
assert len(rows) == 11 and rows[0][1:4] == (0.504, 0.337, 0.704), rows[0]
gcol = {"p": AQUA, "s": GRAY, "c": BLUE}
fig, ax = plt.subplots(figsize=(W, 3.0))
ypos = []; yy = 0
for i, r in enumerate(rows):
    if i in (1, 7): yy += 0.6
    ypos.append(yy); yy += 1
ypos = -np.array(ypos)
for (lab_, e, l, h, g), yv in zip(rows, ypos):
    if l is not None:
        ax.plot([l, h], [yv, yv], color=gcol[g], lw=1.6, solid_capstyle="round")
        ax.plot(e, yv, "D" if g == "p" else "o", ms=6 if g == "p" else 5, color=gcol[g], mec="white", mew=1.0, zorder=3)
    else:
        ax.plot(e, yv, "o", ms=5.5, mfc="white", mec=INK2, mew=1.2, zorder=4)
    ax.text(1.02, yv, f"{e:+.3f}".replace("-", "−") + ("" if l is not None else " (no CI)"), va="center", fontsize=7.2, color=INK2, transform=ax.get_yaxis_transform())
ax.set_yticks(ypos, [r[0] for r in rows]); ax.axvline(0, color=INK2, lw=0.7, ls="--", zorder=1)
ax.set_xlim(-0.3, 2.5); ax.set_xlabel("Difference in baseline-referenced gain (percentage points)"); grid(ax, "x")
fig.tight_layout(); save(fig, "fig_contrasts_col")

# ---------- what the layer learned (p261) ----------
A = json.load(open(f"{H}/out/PAPER_ANALYSES_2.json")); MAN["inputs"]["PAPER_ANALYSES_2.json"] = sha(f"{H}/out/PAPER_ANALYSES_2.json")
fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.45), gridspec_kw={"width_ratios": [1.15, 1]})
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
b.legend(fontsize=6.5, loc="upper right", ncol=2, handletextpad=0.2, columnspacing=0.8); grid(b, "both")
b.set_title("(b) Gain per episode vs. activity", fontsize=8.5, loc="left")
fig.tight_layout(w_pad=2.5); save(fig, "fig_learned_col")

# ---------- case study, extremes and decisions (p256b) ----------
EA = json.load(open(f"{H}/out/PAPER_ERROR_ANALYSIS.json")); MAN["inputs"]["PAPER_ERROR_ANALYSIS.json"] = sha(f"{H}/out/PAPER_ERROR_ANALYSIS.json")
man = [r for f in (f"{N}/k07x3run/K07X3_RUN_MANIFEST.csv", f"{N}/k07x3run/K07X3_RUN_MANIFEST_S2.csv") for r in csv.DictReader(open(f))]; key2 = {(r["architecture"], r["weather"], r["policy_key"]): r for r in man}
w0 = EA["X_weather"]["worst_10_days"][0]; arch, wx = w0["episode"].split("|"); day = w0["date"]
Z, S = {}, {}
for f in glob.glob(f"{N}/results_k07x3/**/*.summary.json", recursive=True):
    s = json.load(open(f)); r = key2.get((s["building"], s["weather"], s["policy_key"]))
    if r and r["architecture"] == arch and r["weather"] == wx and r["policy"] in ("G30", "SUP") and s["steps_done"] == 8640: Z[r["policy"]] = np.load(f.replace(".summary.json", ".npz"), allow_pickle=True); S[r["policy"]] = s
g, u = Z["G30"], Z["SUP"]; ts = np.array([str(x) for x in g["ts"]]); tt = np.array([dt.datetime.fromisoformat(x) for x in ts])
dday = dt.date.fromisoformat(day); win = np.array([(t.date() >= dday - dt.timedelta(days=2)) and (t.date() <= dday) for t in tt])
zm = lambda z: np.nanmean(z["zone_t_k"].astype(float), axis=1) - 273.15
WN = {"energy": ("normalized_energy_cost", 0.2), "carbon": ("normalized_carbon_emission", 0.2), "comfort": ("normalized_productivity_regret", 0.6)}
def comp(z, nm): ck = [str(x) for x in z["comp_keys"]]; k, wt = WN[nm]; return np.abs(wt * z["comps"][:, ck.index(k)].astype(float))
dmask = np.array([t.date() == dday for t in tt]); hours = np.array([t.hour for t in tt])
hourly = {nm: [float((comp(u, nm)[dmask & (hours == h)] - comp(g, nm)[dmask & (hours == h)]).sum()) for h in range(24)] for nm in WN}
lg = S["SUP"]["ros"]["log"]; idx_day = np.where(dmask)[0]
dec = [e for e in lg if e["event"] in ("start", "check", "update") and idx_day[0] - 288 <= e["step"] <= idx_day[-1]]
ovr_day = int(u["tr_ovr"][dmask].any(axis=1).sum()) if "tr_ovr" in u.files else None
case = {"episode": w0["episode"], "date": day, "weekday": dday.strftime("%A"), "day_gain_pct": w0["gain_pct"], "tmean_day": w0["tmean"], "dT_vs_previous_day": w0["dT"],
        "oat_daily_means": {str(d): float(g["oat_c"][np.array([t.date() == d for t in tt])].mean()) for d in sorted({t.date() for t in tt[win]})},
        "decisions": dec, "pressure_override_steps_that_day": ovr_day, "hourly_component_change": hourly,
        "day_component_change_sum": {nm: float(sum(v)) for nm, v in hourly.items()}, "day_loss_G30": float(np.abs(g["rewards"][dmask]).sum()), "day_loss_SUP": float(np.abs(u["rewards"][dmask]).sum())}
json.dump(case, open(f"{OUT}/CASE_STUDY_WORST_DAY.json", "w"), indent=1, default=str)
assert json.load(open(f"{OUT}/CASE_STUDY_WORST_DAY.json")) == json.load(open(f"{H}/out/figs_v2/CASE_STUDY_WORST_DAY.json")), "case data differ from v2"
# case study
fig, (a, b) = plt.subplots(1, 2, figsize=(5.3, 2.5), gridspec_kw={"width_ratios": [1.5, 1]})   # 5.3: the title of (b) widens the box
x = np.arange(win.sum()) / 12.0
l_out, = a.plot(x, g["oat_c"][win], color=GRAY, lw=0.9, label="outdoor")
l_lay, = a.plot(x, zm(u)[win], color=AQUA, lw=1.7, label="zones, with layer")
l_g30, = a.plot(x, zm(g)[win], color=BLUE, lw=0.9, label="zones, G30")
for e in dec:
    if e["event"] == "start":
        k = int(np.where(np.where(win)[0] == e["step"])[0][0]) if e["step"] in np.where(win)[0] else None
        if k is not None: a.axvline(k / 12.0, color=AQUA, lw=0.8, ls="--")
a.set_xlabel("hours from %s 00:00" % (dday - dt.timedelta(days=2)).isoformat()); a.set_ylabel("temperature (°C)")
a.legend(handles=[l_out, l_g30, l_lay], fontsize=6.5, loc="best"); grid(a)
a.set_title("(a) Weekend and Monday %s" % day, fontsize=8.5, loc="left")
hh = np.arange(24)
for j, (nm, col) in enumerate((("energy", ORANGE), ("carbon", GRAY), ("comfort", BLUE))):
    b.bar(hh + (j - 1) * 0.28, np.array(hourly[nm]), color=col, width=0.28, label=nm)
b.axhline(0, color=INK2, lw=0.6); b.set_xlabel("hour of day"); b.set_ylabel("loss change (layer − G30)"); b.legend(fontsize=6.5); grid(b)
b.set_title("(b) Change of the loss components", fontsize=8.5, loc="left")
fig.tight_layout(w_pad=1.5); save(fig, "fig_case_col")
# extremes and hours: two rows
X = EA["X_weather"]; T = EA["T_time"]
def pretty(l):
    l = l.replace("<=", "≤").replace(">=", "≥"); l = re.sub(r"(-?\d+)\.\.(-?\d+)", r"\1 to \2", l)
    return re.sub(r"(?<![\w])-(?=\d)", "−", l).replace(" C", " °C")
fig = plt.figure(figsize=(W, 4.3)); gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])
a, b, c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
def bars(ax, d, title):
    labs = list(d); vals = [d[k][0] for k in labs]; ns = [d[k][1] for k in labs]
    ax.bar(range(len(labs)), vals, color=[AQUA if v >= 0 else ORANGE for v in vals], width=0.7)
    for i, (v, n) in enumerate(zip(vals, ns)): ax.text(i, v + 0.1, f"{v:+.1f}\n(n={n})", ha="center", fontsize=6.5, color=INK2)
    ax.set_xticks(range(len(labs)), [pretty(l) for l in labs], rotation=30, fontsize=7, ha="right", rotation_mode="anchor")
    ax.axhline(0, color=INK2, lw=0.6); grid(ax); ax.set_title(title, fontsize=8.5, loc="left"); ax.set_ylim(min(0, min(vals) * 1.3), max(vals) * 1.5)
bars(a, X["by_daily_mean_oat"], "(a) by daily mean outdoor temp."); a.set_ylabel("gain vs. G30 (%)")
bars(b, X["by_day_to_day_change"], "(b) by change from previous day")
hrs = list(range(24)); dd = T["by_hour"]
for nm, col in (("energy", ORANGE), ("carbon", GRAY), ("comfort", BLUE)):
    c.plot(hrs, [dd[str(h)]["delta_pct_of_total_G30_loss"][nm] if str(h) in dd else dd[h]["delta_pct_of_total_G30_loss"][nm] for h in hrs], color=col, lw=1.2, marker="o", ms=2.5, label=nm)
c.axhline(0, color=INK2, lw=0.6); c.set_xlabel("hour of day"); c.set_ylabel("change (% of total loss)"); c.set_xticks(range(0, 24, 3))
c.legend(fontsize=7, loc="upper right"); grid(c); c.set_title("(c) where it pays and gains", fontsize=8.5, loc="left")
fig.tight_layout(h_pad=1.0, w_pad=1.5); save(fig, "fig_extremes_col")
# decisions
mor = EA["D_mornings"]; cm = {"C": BLUE, "M": ORANGE, "H": AQUA}
fig, ax = plt.subplots(figsize=(3.4, 2.5))
for m in mor: ax.plot(m["surprise_recovery_K"], m["err"], "o", ms=3.2, color=cm[m["class"]], alpha=0.75, mec="none")
ax.axhline(0.5, color=ORANGE, lw=0.7, ls="--"); ax.axhline(0, color=INK2, lw=0.6); ax.axvline(0, color=INK2, lw=0.4)
ax.set_xlabel("outdoor temp. change during recovery (K)"); ax.set_ylabel("deficit at opening,\nactual − predicted (K)"); grid(ax, "both")
for c_, n_ in (("C", "cold start"), ("M", "mild start"), ("H", "hot start")): ax.plot([], [], "o", ms=3.2, color=cm[c_], label=n_)
ax.legend(fontsize=6.5, loc="upper right"); fig.tight_layout(); save(fig, "fig_decisions_col")
MAN["n_points"] = {"decisions": len(mor), "learned_episodes": len(A["B_activity_vs_gain"]["per_scene"]), "contrast_rows": len(rows)}
MAN["outputs"] = {f: sha(f"{OUT}/{f}") for f in sorted(os.listdir(OUT)) if f.endswith(".pdf")}
json.dump(MAN, open(f"{OUT}/P266_MANIFEST.json", "w"), indent=1); print(json.dumps(MAN, indent=1))
