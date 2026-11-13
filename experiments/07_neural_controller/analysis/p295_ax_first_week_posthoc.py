"""p295 (post hoc, exploratory; written 2026-11-12 after the AXSUP blind result): could the two architectures on which the layer lost on
the Residual PPO controller (52209, 52211) have been seen in the first week (steps 0-2015)?  Per architecture (3 windows pooled):
the controller's first-week comfort share (p278 per-episode values, averaged), the layer's first-week gain, its gain over days 8-30 and
over the whole episode.  Two candidate damage-limitation signals are scored against the final sign: (a) first-week comfort share in the
lowest third; (b) negative first-week gain.  Descriptive only: no signal is chosen or tuned here.  usage: python3 p295_ax_first_week_posthoc.py <out.json>"""
import sys, os, csv, glob, json, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); N = os.path.dirname(H); OUT = sys.argv[1]; WK = 2016
man = list(csv.DictReader(open(f"{H}/AXSUP_BLIND_MANIFEST.csv"))); key = {(r["architecture"], r["weather"], r["policy_key"]): r for r in man}
RW = {}
for f in glob.glob(f"{N}/results_axsup_blind/**/*.summary.json", recursive=True) + glob.glob(f"{H}/laptop/work_axb*/out/**/*.summary.json", recursive=True):
    s = json.load(open(f)); r = key.get((s["building"], s["weather"], s["policy_key"]))
    if not r or r["policy"] not in ("AX", "AXS") or s["status"] != "OK" or s["steps_done"] != 8640: continue
    RW[(r["architecture"], r["weather"], r["policy"])] = np.abs(np.load(f.replace(".summary.json", ".npz"), allow_pickle=True)["rewards"])
HA = json.load(open(f"{H}/RESULT_HEADROOM_AX.json")); fw = {}
for e in HA["per_episode"]: fw.setdefault(e["architecture"], []).append(e["first_week_share"])
res = {"note": "post hoc, exploratory; no signal chosen or tuned", "architectures": {}}
for a in sorted(fw):
    ws = sorted({w for aa, w, p in RW if aa == a and (aa, w, "AXS") in RW and (aa, w, "AX") in RW})
    g = lambda sl: 100 * (1 - sum(RW[(a, w, "AXS")][sl].sum() for w in ws) / sum(RW[(a, w, "AX")][sl].sum() for w in ws))
    res["architectures"][a] = {"windows": len(ws), "first_week_comfort_share_pct": 100 * float(np.mean(fw[a])), "gain_week1_pct": g(slice(0, WK)),
                               "gain_days8_30_pct": g(slice(WK, None)), "gain_pct": g(slice(None))}
A = res["architectures"]; shares = sorted(v["first_week_comfort_share_pct"] for v in A.values()); cut = shares[len(shares) // 3 - 1]
for nm, flag in (("a_low_first_week_share", lambda v: v["first_week_comfort_share_pct"] <= cut), ("b_negative_first_week_gain", lambda v: v["gain_week1_pct"] < 0)):
    fl = [k for k, v in A.items() if flag(v)]
    res[nm] = {"flagged": fl, "losers_flagged": [k for k in fl if A[k]["gain_pct"] < 0], "winners_flagged": [k for k in fl if A[k]["gain_pct"] >= 0],
               "gain_given_up_by_winners_flagged_pct_points": sum(A[k]["gain_days8_30_pct"] for k in fl if A[k]["gain_pct"] >= 0)}
json.dump(res, open(OUT, "w"), indent=1)
for k, v in A.items(): print(f"{k:16s} first-week share {v['first_week_comfort_share_pct']:5.1f}% | week-1 gain {v['gain_week1_pct']:+6.2f} | days 8-30 {v['gain_days8_30_pct']:+6.2f} | all {v['gain_pct']:+6.2f}")
for nm in ("a_low_first_week_share", "b_negative_first_week_gain"): print(nm, json.dumps(res[nm]))
