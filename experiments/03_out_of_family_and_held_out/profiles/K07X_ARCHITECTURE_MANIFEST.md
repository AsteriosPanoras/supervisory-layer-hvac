# K07-X architecture feasibility manifest (2026-11-10; outcome-blind; official generator)

Generator: official `run_floor_generator.build_config` with `DEFAULT_RANGES` (the ranges the docstring states the evaluation buildings are drawn from), official materials stratification, official `run_scenario_generator` (acceptance check: no zone below 20 m2). Seeds 52000-52007 (composite, 52005-52007 reserves) and 52100-52109 (one per non-composite footprint). Private kit copy; the frozen local kit is untouched.

**Family relation (honest):** the 6 shipped training buildings are all `composite/architectural`. The 5 composite candidates are **new draws of that same type** — new geometry, topology and envelope, the same generator family as the hidden evaluation buildings, NOT variants of the six parents. The 10 other footprints are **new morphologies** outside the shipped type and outside the hidden evaluation distribution. Everything earlier in this project (H2, D, M2, CAL1 variants) used perturbed variants of the six parents. So these are **2 morphology classes and 15 distinct buildings, not 15 independent families**.

| split | building | seed | type | footprint | size (ft) | floors | rooms (ft) | zones | AHUs | max zones/AHU | occ/zone | facade U (W/m2K) | envelope C (MJ/m3K) | weather strips | K07 smoke (120 steps, arm B) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dev | building_52001 | 52001 | composite (new draw) | composite | 565x334 | 4 | 45-58 | 70 | 7 | 10 | 1 | 1.56 | 0.961 | denver_1, fairbanks_3 | OK, 0 illegal, 1524 MB, 83 s |
| dev | building_52003 | 52003 | composite (new draw) | composite | 690x417 | 2 | 60-74 | 40 | 4 | 10 | 1 | 1.901 | 1.368 | singapore_5, atlanta_4 | OK, 0 illegal, 1489 MB, 88 s |
| dev | building_52005 | 52005 | composite (new draw) | composite | 509x433 | 3 | 46-62 | 60 | 6 | 10 | 1 | 1.215 | 0.697 | phoenix_7, fairbanks_6 | OK, 0 illegal, 1402 MB, 48 s |
| dev | building_52100 | 52100 | new morphology | rectangle | 410x396 | 3 | 45-59 | 66 | 7 | 10 | 1 | 1.269 | 0.999 | singapore_2, phoenix_2 | OK, 0 illegal, 1291 MB, 81 s |
| dev | building_52101 | 52101 | new morphology | trapezoid | 624x400 | 3 | 32-48 | 136 | 14 | 10 | 1 | 1.773 | 1.824 | atlanta_4, denver_1 | OK, 0 illegal, 1686 MB, 134 s |
| dev | building_52102 | 52102 | new morphology | h_shape | 436x410 | 2 | 41-56 | 49 | 5 | 10 | 1 | 2.03 | 0.481 | atlanta_7, denver_4 | OK, 0 illegal, 974 MB, 56 s |
| dev | building_52105 | 52105 | new morphology | oval | 668x421 | 3 | 31-46 | 176 | 18 | 10 | 2 | 1.058 | 0.975 | denver_4, singapore_2 | OK, 0 illegal, 1934 MB, 168 s |
| dev | building_52106 | 52106 | new morphology | u_shape | 569x414 | 2 | 30-45 | 84 | 9 | 10 | 2 | 1.716 | 1.548 | fairbanks_6, singapore_5 | OK, 0 illegal, 1104 MB, 61 s |
| dev | building_52109 | 52109 | new morphology | triangle | 498x355 | 3 | 56-69 | 21 | 5 | 6 | 2 | 1.922 | 0.495 | fairbanks_3, atlanta_7 | OK, 0 illegal, 1123 MB, 36 s |
| heldout | building_52000 | 52000 | composite (new draw) | composite | 487x387 | 4 | 59-71 | 52 | 6 | 10 | 2 | 1.699 | 0.401 | minneapolis_2, minneapolis_5 | OK, 0 illegal, 1786 MB, 117 s |
| heldout | building_52004 | 52004 | composite (new draw) | composite | 535x383 | 4 | 54-70 | 56 | 6 | 10 | 1 | 2.059 | 0.555 | minneapolis_8, minneapolis_7 | OK, 0 illegal, 1680 MB, 88 s |
| heldout | building_52103 | 52103 | new morphology | t_shape | 581x332 | 3 | 50-62 | 36 | 7 | 9 | 1 | 1.626 | 0.777 | minneapolis_3, minneapolis_6 | OK, 0 illegal, 1233 MB, 49 s |
| heldout | building_52104 | 52104 | new morphology | pentagon | 412x377 | 2 | 42-56 | 33 | 7 | 8 | 2 | 1.513 | 1.649 | minneapolis_1, minneapolis_4 | OK, 0 illegal, 846 MB, 40 s |
| heldout | building_52107 | 52107 | new morphology | parallelogram | 513x359 | 2 | 53-66 | 30 | 3 | 10 | 2 | 1.365 | 0.569 | minneapolis_2, minneapolis_5 | OK, 0 illegal, 941 MB, 44 s |
| heldout | building_52108 | 52108 | new morphology | semicircle | 550x396 | 3 | 53-68 | 36 | 7 | 8 | 2 | 1.195 | 0.389 | minneapolis_1, minneapolis_4 | OK, 0 illegal, 1256 MB, 58 s |
| replaced | building_52002 | 52002 | composite (new draw) | composite | 605x423 | 2 | 50-66 | 0 | 0 | 0 | 2 | 1.346 | 1.805 | phoenix_7, fairbanks_6 | not built |

**Feasibility blockers recorded:** building_52002 (composite, dev) failed the official generator's acceptance check (one 17.16 m2 zone below the 20 m2 minimum; the generator says to change the seed). It is replaced by the first pre-declared composite reserve, building_52005, which inherits its split and strips. This is a technical failure, not a performance-based replacement. No other failure: 15/15 architectures build and run K07 (0 illegal actions).
**Split rule (outcome-blind):** held-out = the 2 composite candidates and the 4 footprints with the smallest sha256 keys ('K07X|HO|seed|…', 'K07X|HO|shape|…'). Held-out morphologies (semicircle, t_shape, pentagon, parallelogram) never appear in development.
**Weather:** development uses the 10 official strips of the 5 training stations that the shipped training set does not use (each architecture gets 2 strips of different stations, fixed-offset rule). Held-out uses **minneapolis**, a shipped station used neither by the training set nor by development: one cold-half strip (Jan, Feb, Apr, Oct) and one warm-half strip (May, Jun, Aug, Sep) per architecture.
**Cost (smoke, 2 runs in parallel at nice 19 next to CTX-A, so pessimistic):** large footprints (oval 176 zones, trapezoid 136 zones) are 3-4x slower and need up to 1.9 GB per run. Full 8640-step runs are expected at about 30-90 min on a Kaggle worker, so the laptop must run fewer workers than CTX-A.
**Within-family secondary (3 never-used variants of the six parents, outcome-blind):** b2_48031 (parent train_2, 108 zones, 11 AHUs), b2_40154 (train_7, 37, 4), b2_40155 (train_8, 46, 6), crossed onto the same minneapolis strips as the held-out set.
