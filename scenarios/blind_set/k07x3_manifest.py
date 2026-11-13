"""K07-X3: write K07X3_MANIFEST.md (architectures, strips, smoke, run-time estimate) from K07X3_COHORT.json and the smoke summaries.  Reads ONLY status, steps, illegal actions,
memory and wall time from the smoke summaries (no score is read or printed).  usage: python3 k07x3_manifest.py <k07x3_dir>"""
import sys, json, glob, os
import numpy as np
D = sys.argv[1]; coh = json.load(open(os.path.join(D, "K07X3_COHORT.json"))); sm = {}
for f in glob.glob(os.path.join(D, "laptop_run_k07x3/work_k07x3smoke/out/stage1/*.summary.json")):
    s = json.load(open(f)); sm[s["building"]] = {k: s.get(k) for k in ("status", "weather", "steps_done", "illegal_actions", "ru_maxrss_mb", "wall_s")}
z = {a["name"]: a["zones"] for a in coh["architectures"]}
ok = [n for n in sm if sm[n]["status"] == "OK" and sm[n]["steps_done"] == 120]
# Run-time model: 10-day wall = r x (120-step smoke wall), r calibrated on the two measured references, whose smoke ran the same way (same key, 120 steps, 2 parallel, nice 19,
# next to other runs; summaries in the K07-X smoke folder): building_52007 smoke 180.7 s -> 10-day 18 min (r 5.98); building_52105 smoke 168.1 s -> 10-day 26 min (r 9.28).
# Central r = their mean; the range is the two r.  30-day = 3 x 10-day (the brief).  A zones-only model (3.05 ms per zone-step from the same references) is reported as a cross-check.
R_LO, R_HI = 18 * 60 / 180.7, 26 * 60 / 168.1; R_C = (R_LO + R_HI) / 2; C_REF = (26 * 60 / (2880 * 176) + 18 * 60 / (2880 * 124)) / 2
def est(n, steps, r=None): return (sm[n]["wall_s"] * (r or R_C) / 60) * (steps / 2880)
def est_z(n, steps): return steps * C_REF * z[n] / 60
setup, c_smoke = float("nan"), float("nan")
L = ["# K07-X3 fresh blind set: architecture manifest (outcome-blind; official generator; 2026-11-10)", "",
     "| building | seed | class | shape | size (ft) | floors | zones | AHUs | facade U | envelope C | COLD strip (T19) | MILD strip (T19) | HOT strip (T19) | smoke (120 steps, HOT strip) | est. 10 d / 30 d (min, 1 local worker) |",
     "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for a in coh["architectures"]:
    cs = {c["day1_class"]: c for c in coh["cells"] if c["architecture"] == a["name"]}; s = sm.get(a["name"])
    sm_txt = f'{s["status"]}, {s["steps_done"]} steps, {s["illegal_actions"]} illegal, {s["ru_maxrss_mb"]:.0f} MB, {s["wall_s"]:.0f} s' if s else "not run"
    L.append(f'| {a["name"]} | {a["seed"]} | {"composite" if a["group"] == "composite" else "OOD"} | {a["shape"]} | {a["width_ft"]}x{a["height_ft"]} | {a["floors"]} | {a["zones"]} | {a["ahus"]} | {a["facade_U_W_m2K"]} | {a["envelope_C_MJ_m3K"]} | '
             + " | ".join(f'{cs[k]["weather"]} ({cs[k]["day1_T19"]:.2f})' for k in ("C", "M", "H"))
             + (f' | {sm_txt} | {est(a["name"], 2880):.0f} / {est(a["name"], 8640):.0f} ({est(a["name"], 2880, R_LO):.0f}-{est(a["name"], 2880, R_HI):.0f} / {est(a["name"], 8640, R_LO):.0f}-{est(a["name"], 8640, R_HI):.0f}) |' if s else " | not run | - |"))
tot10 = sum(est(n, 2880) for n in z) * 3; tot30 = sum(est(n, 8640) for n in z) * 3; totz10 = sum(est_z(n, 2880) for n in z) * 3
L += ["", f"**Smoke:** key `k07@RAMP=0.25,COLD_SAT=-0.8,HEAT_U_SCALE=0.875,KP=0.075,NIGHT_HEAT_DROP_K=3,MILD_SHIFT_K=0.75`, 120 steps, each building on its HOT strip, laptop stage mechanism "
      f"(run_stage_laptop_k07x3.sh = run_stage_laptop_k07x2.sh with H = laptop_run_k07x3; kernel_runner_v3 KRE-v2, runner run_episode_k07_v9x.py), 2 workers, nice 19, cores 12-13, next to the authors' runs on cores 0-11. "
      f"{len(ok)}/{len(z)} OK.",
      f"**Run time (one local worker, per scene of that building; all 3 strips of a building cost about the same):** 10-day = r x smoke wall, r = {R_C:.2f} (range {R_LO:.2f}-{R_HI:.2f} = the two measured references: "
      f"building_52007 smoke 180.7 s -> 10-day 18 min, building_52105 smoke 168.1 s -> 10-day 26 min); 30-day = 3 x 10-day.  Smoke wall depends on floor area x floors more than on zones "
      f"(building_52207, 76 zones on 4 floors, is the slowest).  Whole set, one policy key, 36 scenes: about {tot10 / 60:.1f} h (10-day) and {tot30 / 60:.1f} h (30-day) of single-worker time; "
      f"divide by the parallel workers.  Cross-check, zones-only model (3.05 ms per zone-step from the same references): {totz10 / 60:.1f} h (10-day)."]
open(os.path.join(D, "K07X3_MANIFEST.md"), "w").write("\n".join(L) + "\n")
json.dump({"smoke": sm, "r_central": R_C, "r_range": [R_LO, R_HI], "c_ref_ms_per_zone_step": 1000 * C_REF,
           "est_min_per_scene": {n: {"10d": round(est(n, 2880), 1), "10d_range": [round(est(n, 2880, R_LO), 1), round(est(n, 2880, R_HI), 1)], "30d": round(est(n, 8640), 1), "zones_model_10d": round(est_z(n, 2880), 1)} for n in z},
           "total_h_one_worker_36_scenes": {"10d": round(tot10 / 60, 2), "30d": round(tot30 / 60, 2), "zones_model_10d": round(totz10 / 60, 2)}}, open(os.path.join(D, "K07X3_SMOKE_AND_RUNTIME.json"), "w"), indent=1)
print("\n".join(L[-2:])); print({n: (sm[n]["status"], sm[n]["illegal_actions"], round(sm[n]["ru_maxrss_mb"]), round(sm[n]["wall_s"])) for n in sorted(sm)})
