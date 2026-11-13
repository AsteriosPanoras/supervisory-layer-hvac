"""p290 (experiment B, PREREG_EXT12_v1): oracle prior for the learned start on K07-G30.
OR = the layer with TAU0 set to the episode's own final time constant (an upper bound, not deployable); reference = SUP (or SUPR, see
ext12_bc_common.reference).  It runs only if all 36 OR runs and the reference exist.
 - B1: OR beats the reference (pooled gain > 0).
 - B2: the pooled gain over days 1-14 (steps 0-4031) exceeds the gain over days 15-30 (steps 4032-8639), composite architectures.
Descriptive: OR, SUP and ROS against G30.  Written 2026-11-12 before any OR run.  usage: python -I p290_oracle_prior.py <out.json>"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))      # python -I does not add the script's folder
from ext12_bc_common import load, reference, tools
OUT = sys.argv[1]
L, RW, C, RAW, grp = load(); ref, rep = reference(L, RAW)
n_or = sum(1 for k in L if k[2] == "OR")
if ref is None or n_or < 36:
    json.dump({"status": f"NOT RUN: reference {ref}, OR runs {n_or}", **rep}, open(OUT, "w"), indent=1); sys.exit(print("not run", ref, n_or, rep))
cells, CO, OOD, gain, contrast = tools(L, grp, "OR")
res = {"status": "OK", **rep, "OR_vs_ref": contrast("OR", ref), "OR_vs_G30": contrast("OR", "G30"), "ref_vs_G30": contrast(ref, "G30"), "ROS_vs_G30": contrast("ROS", "G30")}
cc = [(a, w) for a, w in cells if a in CO]; E = 4032
early = 100 * (1 - sum(RW[(a, w, "OR")][:E].sum() for a, w in cc) / sum(RW[(a, w, ref)][:E].sum() for a, w in cc))
late = 100 * (1 - sum(RW[(a, w, "OR")][E:].sum() for a, w in cc) / sum(RW[(a, w, ref)][E:].sum() for a, w in cc))
res["gain_days1_14_pct"], res["gain_days15_30_pct"] = early, late
res["predictions"] = {"B1_or_beats_ref": res["OR_vs_ref"]["gain_pct"] > 0, "B2_early_gain_exceeds_late_gain": early > late}
json.dump(res, open(OUT, "w"), indent=1)
v = res["OR_vs_ref"]
print(f"OR vs {ref}: {v['gain_pct']:+.3f} % CI [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] better {v['composite_changed_better']}/{v['composite_changed']} worst {v['worst_architecture_gain_pct']:+.3f} OOD {v['ood_gain_pct']}"
      f" | days 1-14 {early:+.3f} %, days 15-30 {late:+.3f} % | OR vs G30 {res['OR_vs_G30']['gain_pct']:+.3f}, ref vs G30 {res['ref_vs_G30']['gain_pct']:+.3f} | REPRO_OK {rep['REPRO_OK']} | {res['predictions']}")
