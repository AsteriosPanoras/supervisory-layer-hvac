"""p275: fidelity of the unchanged supervisor on the public Residual PPO controller (runner run_episode_k07_v9supe_ax.py), as p250 did
for ZT Best.  Development scenes 52010 phoenix_7 and 52103 fairbanks_3, 864 steps; no reward is used to decide anything.
 F2: 'k07sup@;ROS=0,TR=0' equals 'k07@' (continuous and discrete actions and rewards, bitwise).
 F3: 'k07sup@;ROS=1,TR=1' runs (status OK, 864 steps, no illegal action) and is identical to 'k07@' up to its first differing action,
     where the supervisor had acted (its first learned start, or its first pressure override, or the start-lead base it sets at step 0);
     the activity is reported (learned starts, pressure-override steps per air handler).
 (F1 of p250 compared a new runner with an earlier one; no earlier run of this controller exists, so there is nothing to compare with.)
usage: python -I p275_ax_fidelity.py <work_dir_out> <result_json>"""
import sys, glob, json, numpy as np
OUT, RES = sys.argv[1], sys.argv[2]
def load(b, w, key):
    sk = key.replace("@", "__").replace(",", "_").replace("=", "-"); f = glob.glob(f"{OUT}/{b}_{w}_{sk}_864.summary.json")
    assert len(f) == 1, (b, w, key, f); s = json.load(open(f[0])); z = np.load(f[0].replace(".summary.json", ".npz"), allow_pickle=True)
    return s, z
R = {"scenes": {}, "ALL_PASS": True}
for b, w in (("building_52010", "phoenix_7"), ("building_52103", "fairbanks_3")):
    (s0, z0), (s2, z2), (s3, z3) = (load(b, w, k) for k in ("k07@", "k07sup@;ROS=0,TR=0", "k07sup@;ROS=1,TR=1"))
    ok = all(s["status"] == "OK" and s["steps_done"] == 864 and s["illegal_actions"] == 0 for s in (s0, s2, s3))
    f2 = all(np.array_equal(z0[k], z2[k]) for k in ("actions_cont", "disc", "rewards"))
    a0, a3 = z0["actions_cont"], z3["actions_cont"]; d0, d3 = z0["disc"], z3["disc"]
    diff = np.where((np.abs(a0 - a3).max(axis=1) > 0) | (d0 != d3).any(axis=1))[0]; first = int(diff[0]) if len(diff) else None
    ros = (s3.get("ros") or {}).get("log", []); starts = [e for e in ros if e.get("event") == "start"]; tr = s3.get("tr") or {}
    ovr = tr.get("override_steps"); tr_first = next((e["step"] for e in tr.get("log", []) if e.get("event") == "override_on"), None)
    pre_equal = first is None or (np.array_equal(a0[:first], a3[:first]) and np.array_equal(z0["rewards"][:first], z3["rewards"][:first]))
    acted_at = sorted(x for x in ([starts[0]["step"]] if starts else []) + ([tr_first] if tr_first is not None else []))
    base_ps = (s3.get("ros") or {}).get("base_prestart"); pi0 = z0["pi_active_c"]; pi3 = z3["pi_active_c"]
    first_pi_diff = int(np.where(pi0 != pi3)[0][0]) if (pi0 != pi3).any() else None
    explained = first is None or (acted_at and first >= acted_at[0]) or (first_pi_diff is not None and first >= first_pi_diff)
    f3 = ok and pre_equal and bool(explained)
    R["scenes"][f"{b}|{w}"] = {"status_ok_no_illegal": ok, "F2_identical": f2, "F3_first_diff_step": first, "F3_identical_before": pre_equal,
                               "supervisor_first_action_step": acted_at[0] if acted_at else None, "first_pi_hours_difference_step": first_pi_diff,
                               "learned_starts": [{k: e.get(k) for k in ("step", "why", "tod_h", "prestart_set", "tau_h")} for e in starts],
                               "pressure_override_steps_per_ahu": ovr, "base_prestart": base_ps, "F3_pass": f3,
                               "ms_per_step": [s0["policy_ms_per_step"], s3["policy_ms_per_step"]], "wall_s": [s0["wall_s"], s3["wall_s"]],
                               "rss_mb": [s0["ru_maxrss_mb"], s3["ru_maxrss_mb"]]}
    R["ALL_PASS"] &= bool(ok and f2 and f3)
json.dump(R, open(RES, "w"), indent=1)
print(json.dumps(R, indent=1)[:3000])
