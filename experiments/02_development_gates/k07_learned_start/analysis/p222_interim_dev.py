"""Interim look at FINISHED DEVELOPMENT runs only (ROS or TR vs G30).  Held-out scenes are filtered out by the development manifest BEFORE any file is opened.
usage: python -I p222_interim_dev.py <ROS|TR> <dev_manifest.csv> <dir> [<dir> ...]   (G30 dev runs come from results_k07x2/dev2k*)"""
import sys, json, glob, csv, os, numpy as np
ARM, MAN = sys.argv[1], sys.argv[2]; DIRS = sys.argv[3:]; N = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
man = [r for r in csv.DictReader(open(MAN)) if r["policy"] in ("G30", ARM)]; dev = {(r["architecture"], r["weather"]) for r in man}; key = {r["policy"]: r["policy_key"] for r in man}
def runs(dirs, pkey):
    out = {}
    for d in dirs:
        for f in glob.glob(f"{d}/**/*.summary.json", recursive=True):
            b = os.path.basename(f)
            if not any(b.startswith(f"{a}_{w}_") for a, w in dev): continue                 # held-out files are never opened
            s = json.load(open(f))
            if s["policy_key"] == pkey and (s["building"], s["weather"]) in dev and s["status"] == "OK" and s["steps_done"] == 8640: out[(s["building"], s["weather"])] = (s, f)
    return out
A = runs(DIRS, key[ARM]); G = runs([f"{N}/results_k07x2/dev2k{i}" for i in range(5)], key["G30"])
print(f"{ARM}: {len(A)} finished development runs of 18")
for c in sorted(A):
    s, f = A[c]; la = float(np.abs(np.load(f.replace(".summary.json", ".npz"), allow_pickle=True)["rewards"]).mean())
    lg = float(np.abs(np.load(G[c][1].replace(".summary.json", ".npz"), allow_pickle=True)["rewards"]).mean()) if c in G else None
    if ARM == "ROS":
        st = [e for e in s["ros"]["log"] if e["event"] == "start" and e["why"] == "need"]; beh = f"extended mornings {len(st)}, mean lead {np.mean([6 - e['tod_h'] for e in st]):.2f} h" if st else "never extended"
        beh += f", tau {s['ros']['tau_final_h']:.2f} h, frozen {s['ros']['frozen']}"
    else:
        t = s["tr"]; beh = f"override share {sum(t['override_steps']) / max(1, sum(t['pi_steps'])):.1%}, AHUs used {sum(1 for x in t['override_steps'] if x)}/{len(t['override_steps'])}, unresponsive zones {len(t['unresponsive_zones'])}"
    print(f"  {c[0][-5:]} {c[1]:13s} {ARM} vs G30 {100 * (la / lg - 1):+.2f} % | {beh}" if lg else f"  {c} no G30 reference")
