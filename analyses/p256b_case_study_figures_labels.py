"""[p256b = p256 with the figure labels in the paper's common language: 'tuned static' -> 'G30'; nothing else changed]
Case study of the worst blind day and the error-analysis figures (post hoc, descriptive; frozen, already-read runs).  usage: python -I p256_case_study_figures.py <out_dir>
Case: the blind day with the largest loss of the layer vs G30 (from PAPER_ERROR_ANALYSIS.json, worst_10_days[0]).  For that episode: outdoor temperature, mean zone
temperature of both arms, the learned-start decision of that morning (time, deficit, time constant, predicted vs actual deficit at opening), pressure overrides, and the
hourly change of the weighted loss components on that day.  Figures (same style as v47/v48): fig_case.pdf (case study), fig_extremes.pdf (gain by daily mean outdoor
temperature, by day-to-day change, and by hour of day), fig_decisions.pdf (prediction error at opening vs weather change during the recovery)."""
import sys, os, csv, glob, json, numpy as np, datetime as dt
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
H = os.path.dirname(os.path.abspath(__file__)); N = os.path.dirname(H); OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, AQUA, GRAY, INK, INK2, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "axes.labelsize": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight"})
def grid(ax, axis="y"): ax.grid(axis=axis, color=GRID, lw=0.6); ax.set_axisbelow(True)
EA = json.load(open(f"{H}/out/PAPER_ERROR_ANALYSIS.json"))
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
tr = S["SUP"].get("tr") or {}; ovr_day = int(u["tr_ovr"][dmask].any(axis=1).sum()) if "tr_ovr" in u.files else None
case = {"episode": w0["episode"], "date": day, "weekday": dday.strftime("%A"), "day_gain_pct": w0["gain_pct"], "tmean_day": w0["tmean"], "dT_vs_previous_day": w0["dT"],
        "oat_daily_means": {str(d): float(g["oat_c"][np.array([t.date() == d for t in tt])].mean()) for d in sorted({t.date() for t in tt[win]})},
        "decisions": dec, "pressure_override_steps_that_day": ovr_day, "hourly_component_change": hourly,
        "day_component_change_sum": {nm: float(sum(v)) for nm, v in hourly.items()}, "day_loss_G30": float(np.abs(g["rewards"][dmask]).sum()), "day_loss_SUP": float(np.abs(u["rewards"][dmask]).sum())}
json.dump(case, open(f"{OUT}/CASE_STUDY_WORST_DAY.json", "w"), indent=1, default=str)
# ---- Figure: case study ----
fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 2.5), gridspec_kw={"width_ratios": [1.5, 1]})
x = np.arange(win.sum()) / 12.0
a.plot(x, g["oat_c"][win], color=GRAY, lw=0.9, label="outdoor")
a.plot(x, zm(g)[win], color=BLUE, lw=1.1, label="zones, G30")
a.plot(x, zm(u)[win], color=AQUA, lw=1.1, label="zones, with layer")
for e in dec:
    if e["event"] == "start":
        k = int(np.where(np.where(win)[0] == e["step"])[0][0]) if e["step"] in np.where(win)[0] else None
        if k is not None: a.axvline(k / 12.0, color=AQUA, lw=0.8, ls="--")
a.set_xlabel("hours from %s 00:00" % (dday - dt.timedelta(days=2)).isoformat()); a.set_ylabel("temperature (°C)"); a.legend(fontsize=6.5, loc="best"); grid(a)
a.set_title("(a) Weekend and Monday %s" % day, fontsize=8.5, loc="left")
hh = np.arange(24); bot = np.zeros(24)
for j, (nm, col) in enumerate((("energy", ORANGE), ("carbon", GRAY), ("comfort", BLUE))):
    v = np.array(hourly[nm]); b.bar(hh + (j - 1) * 0.28, v, color=col, width=0.28, label=nm)
b.axhline(0, color=INK2, lw=0.6); b.set_xlabel("hour of day"); b.set_ylabel("loss change (layer − G30)"); b.legend(fontsize=6.5); grid(b)
b.set_title("(b) Change of the loss components", fontsize=8.5, loc="left")
fig.tight_layout(w_pad=1.5); fig.savefig(f"{OUT}/fig_case.pdf"); fig.savefig(f"{OUT}/fig_case.png", dpi=200); plt.close(fig)
# ---- Figure: extremes and hours ----
X = EA["X_weather"]; T = EA["T_time"]
fig, (a, b, c) = plt.subplots(1, 3, figsize=(6.8, 2.3), gridspec_kw={"width_ratios": [1.1, 1.1, 1.3]})
def bars(ax, d, title, rot=35):
    labs = list(d); vals = [d[k][0] for k in labs]; ns = [d[k][1] for k in labs]
    ax.bar(range(len(labs)), vals, color=[AQUA if v >= 0 else ORANGE for v in vals], width=0.7)
    for i, (v, n) in enumerate(zip(vals, ns)): ax.text(i, v + 0.1, f"{v:+.1f}\n(n={n})", ha="center", fontsize=5.5, color=INK2)
    ax.set_xticks(range(len(labs)), [l.replace(" C", "°C") for l in labs], rotation=rot, fontsize=6, ha="right"); ax.axhline(0, color=INK2, lw=0.6); grid(ax); ax.set_title(title, fontsize=8, loc="left"); ax.set_ylim(min(0, min(vals) * 1.3), max(vals) * 1.45)
bars(a, X["by_daily_mean_oat"], "(a) by daily mean outdoor temp."); a.set_ylabel("gain vs. G30 (%)")
bars(b, X["by_day_to_day_change"], "(b) by change from previous day")
hrs = list(range(24)); dd = T["by_hour"]
for nm, col in (("energy", ORANGE), ("carbon", GRAY), ("comfort", BLUE)):
    c.plot(hrs, [dd[str(h)]["delta_pct_of_total_G30_loss"][nm] if str(h) in dd else dd[h]["delta_pct_of_total_G30_loss"][nm] for h in hrs], color=col, lw=1.2, marker="o", ms=2.5, label=nm)
c.axhline(0, color=INK2, lw=0.6); c.set_xlabel("hour of day"); c.set_ylabel("change (% of total loss)"); c.legend(fontsize=6); grid(c); c.set_title("(c) where it pays and gains", fontsize=8, loc="left")
fig.tight_layout(w_pad=1.0); fig.savefig(f"{OUT}/fig_extremes.pdf"); fig.savefig(f"{OUT}/fig_extremes.png", dpi=200); plt.close(fig)
# ---- Figure: decisions under weather unknown at decision time ----
mor = EA["D_mornings"]; cm = {"C": BLUE, "M": ORANGE, "H": AQUA}
fig, ax = plt.subplots(figsize=(3.4, 2.4))
for m in mor: ax.plot(m["surprise_recovery_K"], m["err"], "o", ms=3.2, color=cm[m["class"]], alpha=0.75, mec="none")
ax.axhline(0.5, color=ORANGE, lw=0.7, ls="--"); ax.axhline(0, color=INK2, lw=0.6); ax.axvline(0, color=INK2, lw=0.4)
ax.set_xlabel("outdoor temp. change during recovery (K)"); ax.set_ylabel("deficit at opening: actual - predicted (K)"); grid(ax, "both")
for c_, n_ in (("C", "cold start"), ("M", "mild start"), ("H", "hot start")): ax.plot([], [], "o", ms=3.2, color=cm[c_], label=n_)
ax.legend(fontsize=6, loc="upper right"); fig.tight_layout(); fig.savefig(f"{OUT}/fig_decisions.pdf"); fig.savefig(f"{OUT}/fig_decisions.png", dpi=200); plt.close(fig)
print(json.dumps({k: case[k] for k in ("episode", "date", "weekday", "day_gain_pct", "tmean_day", "dT_vs_previous_day", "oat_daily_means", "pressure_override_steps_that_day", "day_component_change_sum")}, indent=1, default=str))
print("decisions:", json.dumps(dec, default=str)[:1500])
