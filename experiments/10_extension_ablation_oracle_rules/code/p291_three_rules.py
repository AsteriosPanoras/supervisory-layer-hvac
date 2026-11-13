"""p291 (experiment C, PREREG_EXT12_v1): a third rule in the same template on K07-G30.
URT = learned start + pressure reset + the Guideline 36 unoccupied setup (UNS_COOL_K=305.4); reference = SUP (or SUPR, see
ext12_bc_common.reference).  It runs only if all 36 URT runs and the reference exist.
 - C1: the sign of the pooled gain URT vs reference equals the sign of UR vs ROS on the same scenes (measured before; the setup on top of
   the learned start).
 - C2: URT's energy + carbon loss is below the reference's (composite architectures; the term the setup targets).
Descriptive: URT and the reference against G30; the component changes of URT as % of the reference's loss.
Written 2026-11-12 before any URT run.  usage: python -I p291_three_rules.py <out.json>"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))      # python -I does not add the script's folder
from ext12_bc_common import load, reference, tools, W
OUT = sys.argv[1]
L, RW, C, RAW, grp = load(); ref, rep = reference(L, RAW)
n_u = sum(1 for k in L if k[2] == "URT")
if ref is None or n_u < 36:
    json.dump({"status": f"NOT RUN: reference {ref}, URT runs {n_u}", **rep}, open(OUT, "w"), indent=1); sys.exit(print("not run", ref, n_u, rep))
cells, CO, OOD, gain, contrast = tools(L, grp, "URT")
res = {"status": "OK", **rep, "URT_vs_ref": contrast("URT", ref), "UR_vs_ROS": contrast("UR", "ROS"), "URT_vs_G30": contrast("URT", "G30"), "ref_vs_G30": contrast(ref, "G30")}
cc = [(a, w) for a, w in cells if a in CO]; tot = sum(sum(C[(a, w, ref)].values()) for a, w in cc)
res["component_change_pct_of_ref_loss"] = {nm: 100 * sum(C[(a, w, "URT")][nm] - C[(a, w, ref)][nm] for a, w in cc) / tot for nm in W}
ec = lambda p: sum(C[(a, w, p)]["energy"] + C[(a, w, p)]["carbon"] for a, w in cc)
sgn = lambda x: (x > 0) - (x < 0)
res["predictions"] = {"C1_sign_matches_UR_vs_ROS": sgn(res["URT_vs_ref"]["gain_pct"]) == sgn(res["UR_vs_ROS"]["gain_pct"]), "C2_energy_carbon_below_ref": ec("URT") < ec(ref)}
json.dump(res, open(OUT, "w"), indent=1)
v = res["URT_vs_ref"]
print(f"URT vs {ref}: {v['gain_pct']:+.3f} % CI [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] better {v['composite_changed_better']}/{v['composite_changed']} worst {v['worst_architecture_gain_pct']:+.3f}"
      f" | UR vs ROS {res['UR_vs_ROS']['gain_pct']:+.3f} | URT vs G30 {res['URT_vs_G30']['gain_pct']:+.3f}, ref vs G30 {res['ref_vs_G30']['gain_pct']:+.3f}"
      f" | components {({k: round(x, 3) for k, x in res['component_change_pct_of_ref_loss'].items()})} | REPRO_OK {rep['REPRO_OK']} | {res['predictions']}")
