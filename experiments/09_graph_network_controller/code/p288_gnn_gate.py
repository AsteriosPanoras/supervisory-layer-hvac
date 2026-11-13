"""p288: the launch gate of PREREG_GNNSUP_v1, on the validation scenes only (no blind file is read).
 G1: GN alone (k07@, GNN mode, the given weights) ran on the 4 validation scenes for 2016 steps, with status OK and 0 illegal actions.
 G2: GN's pooled loss on those scenes <= 1.25 x the teacher's pooled loss on the same scenes and steps. The teacher's loss is read from
     its recording runs (recording mode, which F1 shows to be the teacher's own episode).
Loss = mean |reward| over the episode; pooled = sum over scenes.  usage: python -I p288_gnn_gate.py <weights_sha256> <out.json>"""
import sys, os, glob, json, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); WSHA, OUT = sys.argv[1], sys.argv[2]
VAL = [("building_52003", "atlanta_4"), ("building_52104", "minneapolis_1"), ("building_52109", "atlanta_7"), ("building_train_8", "fairbanks_8")]


def runs(pat, mode, wsha=None):
    out = {}
    for f in glob.glob(pat, recursive=True):
        s = json.load(open(f)); g = s.get("gnn") or {}
        if (s["building"], s["weather"]) not in VAL or s["policy_key"] != "k07@" or g.get("mode") != mode or s["steps_requested"] not in ("2016", 2016):
            continue
        if wsha is not None and g.get("weights_sha256") != wsha:
            continue
        k = (s["building"], s["weather"])
        if k in out: raise SystemExit(f"duplicate {k}: {f}")
        z = np.load(f.replace(".summary.json", ".npz"), allow_pickle=True)
        out[k] = {"status": s["status"], "steps": s["steps_done"], "illegal": s.get("illegal_actions", 0), "L": float(np.abs(z["rewards"]).mean()) if s["status"] == "OK" else None}
    return out


T = runs(f"{H}/laptop/work_rec_*/out/**/*.summary.json", "rec")
G = runs(f"{H}/laptop/work_gnval*/out/**/*.summary.json", "gnn", WSHA)
res = {"weights_sha256": WSHA, "teacher": {f"{b}|{w}": v for (b, w), v in T.items()}, "gn": {f"{b}|{w}": v for (b, w), v in G.items()}}
res["G1"] = all(k in G and G[k]["status"] == "OK" and G[k]["steps"] == 2016 and G[k]["illegal"] == 0 for k in VAL)
ok_t = all(k in T and T[k]["status"] == "OK" and T[k]["steps"] == 2016 for k in VAL)
res["teacher_complete"] = ok_t
if res["G1"] and ok_t:
    res["ratio_pooled_gn_over_teacher"] = sum(G[k]["L"] for k in VAL) / sum(T[k]["L"] for k in VAL)
    res["per_scene_ratio"] = {f"{b}|{w}": G[(b, w)]["L"] / T[(b, w)]["L"] for b, w in VAL}
    res["G2"] = bool(res["ratio_pooled_gn_over_teacher"] <= 1.25)
else:
    res["G2"] = False
res["GATE_PASS"] = bool(res["G1"] and res["G2"])
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps({k: res[k] for k in ("G1", "teacher_complete", "G2", "GATE_PASS")}), res.get("ratio_pooled_gn_over_teacher"), res.get("per_scene_ratio"))
