"""Post hoc, descriptive: where in the day, and in which term, the layer changed the neural controller's loss on the six official scenes."""
import sys, glob, json, numpy as np
D = sys.argv[1]
SC = [("building_train_1", "denver_7"), ("building_train_1", "fairbanks_1"), ("building_train_1", "phoenix_5"), ("building_train_1", "singapore_8"), ("building_train_2", "fairbanks_2"), ("building_train_4", "phoenix_8")]
KEYS = {"AX": "k07__", "AXS": "k07sup__;ROS-1_TR-1"}
W = {"comfort": ("normalized_productivity_regret", 0.6), "energy": ("normalized_energy_cost", 0.2), "carbon": ("normalized_carbon_emission", 0.2)}
P = {"night 0-5": range(0, 5), "early 5-8": range(5, 8), "day 8-18": range(8, 18), "evening 18-24": range(18, 24)}
def load(b, w, arm):
    f = glob.glob(f"{D}/{b}_{w}_{KEYS[arm]}_8640.summary.json")[0]; z = np.load(f.replace(".summary.json", ".npz"), allow_pickle=True)
    ck = [str(x) for x in z["comp_keys"]]; hr = np.array([int(t[11:13]) for t in z["ts"]])
    return z, {nm: np.abs(wt * z["comps"][:, ck.index(c)].astype(float)) for nm, (c, wt) in W.items()}, hr, json.load(open(f))
for b, w in SC:
    za, ca, hr, _ = load(b, w, "AX"); zs, cs, _, ss = load(b, w, "AXS")
    tot = sum(v.sum() for v in ca.values())
    diff = (za["actions_cont"] != zs["actions_cont"]).any(1) | (za["disc"] != zs["disc"]).any(1)
    first = int(np.argmax(diff)) if diff.any() else -1
    starts = [e for e in (ss.get("ros") or {}).get("log", []) if e.get("event") == "start"]
    shift = zs["prestart_attr_c"] - zs["base_prestart_c"]
    print(f"\n== {b}/{w}: gain {100 * (1 - sum(v.sum() for v in cs.values()) / tot):+.2f} % | actions differ on {diff.mean():.0%} of steps (first at step {first}) | layer starts: {len(starts)} | tau_final {ss['ros'].get('tau_final_h')} h | prestart shift (h) mean when shifted {shift[shift != 0].mean() if (shift != 0).any() else 0:+.2f}")
    print(f"   {'period':14s} " + " ".join(f"{k:>8s}" for k in W) + "   (change, % of AX loss; + = layer lowered it)")
    for nm, hrs in P.items():
        m = np.isin(hr, list(hrs))
        print(f"   {nm:14s} " + " ".join(f"{100 * (ca[k][m].sum() - cs[k][m].sum()) / tot:+8.2f}" for k in W))
    occ = za["occupancy"] > 0
    print(f"   {'occupied':14s} " + " ".join(f"{100 * (ca[k][occ].sum() - cs[k][occ].sum()) / tot:+8.2f}" for k in W))
    print(f"   {'unoccupied':14s} " + " ".join(f"{100 * (ca[k][~occ].sum() - cs[k][~occ].sum()) / tot:+8.2f}" for k in W))
