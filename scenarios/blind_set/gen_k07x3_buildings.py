"""K07-X3 (fresh blind set): candidate architectures from the OFFICIAL floor generator, the method of gen_k07x_buildings.py / gen_k07x_composite_batch2.py with NEW seeds.
usage: <kit venv python> gen_k07x3_buildings.py <kit_root (private copy k07x3/kit)> <out_dir (k07x3)>.  Outcome-blind; no simulation.
Composite: ONE official batch, seeds 52200-52219 (20 seeds, materials stratified across the batch by the official draw_materials, rng seeded from sha256(repr(seeds)) as in K07-X);
  the 10 composite architectures are the first 10 seeds (seed order) that pass the official 20 m2 acceptance check (checked in k07x3_cohort.py by the official scenario generator).
OOD: the 2 of the 5 non-composite shapes NOT in the K07-X v2 cohort (rectangle, pentagon, u_shape, parallelogram, triangle) with the smallest sha256('K07X3|OOD|shape|<shape>');
  seeds 52300 and 52301 in that order (one batch of 2, official stratification); a failure is re-seeded with 52302, 52303, ... (same shape, recorded).
Seeds 52200-52219 and 52300-52309 are used nowhere else (checked: no building_522xx / building_523xx in the K07-X kit, the local kit or the scenario indexes)."""
import sys, json, hashlib, random, os
KIT, OUT = sys.argv[1], sys.argv[2]; sys.path.insert(0, KIT); os.chdir(KIT)
from smart_buildings.run_floor_generator import build_config, save_config, DEFAULT_RANGES
from smart_buildings.smart_control.utils.config_generator import BUILDING_SHAPES
from smart_buildings.smart_control.utils.materials import draw_materials, facade_u, envelope_capacity
from pathlib import Path
H = lambda *a: hashlib.sha256("|".join(map(str, a)).encode()).hexdigest()
def mats_for(seeds): return draw_materials(random.Random(int.from_bytes(hashlib.sha256(repr(seeds).encode()).digest()[:8], "big")), len(seeds))
V2_SHAPES = {"h_shape", "semicircle", "t_shape", "oval", "trapezoid"}            # OOD shapes of the K07-X v2 cohort (K07X_V2_COHORT.json "ood_shapes")
free = [s for s in BUILDING_SHAPES if s != "composite" and s not in V2_SHAPES]
ood_shapes = sorted(free, key=lambda s: H("K07X3|OOD|shape", s))[:2]
comp_seeds = list(range(52200, 52220)); ood_seeds = [52300, 52301]
rows = []
def row(c, group):
    fp, m = c["floor_plan"], c["materials"]
    return {"name": c["name"], "seed": c["seed"], "group": group, "shape": fp["building_shape"], "style": fp.get("composite_style"), "width_ft": fp["width"], "height_ft": fp["height"], "floors": fp["num_floors"],
            "room_min_ft": fp["min_room_size"], "room_max_ft": fp["max_room_size"], "shape_ratio": (round(fp["shape_ratio"], 3) if fp.get("shape_ratio") is not None else None),
            "coverage": (round(fp["composite_coverage"], 3) if fp.get("composite_coverage") is not None else None), "ahu_floor": c["ahu"]["num_ahus"], "max_zones_per_ahu": c["ahu"].get("max_zones_per_ahu"),
            "occupants_per_zone": c["simulation"]["occupants_per_zone"], "facade_U_W_m2K": round(facade_u(m), 3), "envelope_C_MJ_m3K": round(envelope_capacity(m) / 1e6, 3), "materials": m}
for s, m in zip(comp_seeds, mats_for(comp_seeds)):
    c = build_config(s, f"building_{s:05d}", "composite", m); save_config(c, Path(KIT) / "smart_buildings" / "buildings", save_pdf=False, show=False); rows.append(row(c, "composite"))
for s, sh, m in zip(ood_seeds, ood_shapes, mats_for(ood_seeds)):
    c = build_config(s, f"building_{s:05d}", sh, m); save_config(c, Path(KIT) / "smart_buildings" / "buildings", save_pdf=False, show=False); rows.append(row(c, "ood"))
json.dump({"default_ranges": DEFAULT_RANGES, "shapes_available": BUILDING_SHAPES, "ood_shape_pool": free, "ood_shapes": ood_shapes, "composite_seeds": comp_seeds, "ood_seeds": ood_seeds, "candidates": rows},
          open(Path(OUT) / "K07X3_BUILDINGS_RAW.json", "w"), indent=1, default=str)
for r in rows: print(r["name"], r["group"], r["shape"], f'{r["width_ft"]}x{r["height_ft"]}', "floors", r["floors"], "rooms", r["room_min_ft"], r["room_max_ft"], "AHU>=", r["ahu_floor"], "U", r["facade_U_W_m2K"], "C", r["envelope_C_MJ_m3K"])
