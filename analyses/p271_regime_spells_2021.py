"""p271: the premise of the explanatory model of Appendix G on the 2021 weather.  Post hoc and descriptive; nothing is run.
Appendix G: switching on entering a regime pays only if the expected remaining spell is long enough; 'if spells that follow a mild start
are long while departures from cold or hot starts are brief, the condition favours frequent reassignment only after mild starts, as the
gate does'.  This script measures exactly that on the 2021 cohort (H2), from the daily decisions logged by the runs: each run's 'unif'
log holds, per day, the observed mean temperature of the 228 five-minute steps and the class the controller saw.
 0. The class sequence is weather-only: for every episode, all arms that log decisions must show the same 30 classes (else abort).
 1. By the first observed class (C / M / H; E kept apart if present): episodes, class changes, length of the first spell, share of days
    in the first class, and the length of the later spells: after a mild start, the spells it moves into; after a cold or hot start,
    the departures from that class (spells away from it, ended by a return or by day 30).
 2. Day-to-day persistence P_kk per class, pooled over episodes.
Days with an invalid window (no class) keep the previous class, as the controller keeps its profile.
Main episodes = H2-main cells with complete logs (the paper analyses 98 of 100 planned; two failed to initialise).
usage: python -I p271_regime_spells_2021.py <out_dir>"""
import sys, os, csv, glob, json, collections, statistics
H = os.path.dirname(os.path.abspath(__file__)); N = os.path.dirname(H); OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
cells = list(csv.DictReader(open(f"{N}/cells_h2.csv"))); cohort = {(c["building"], c["weather"]): c["cohort"] for c in cells}
design = {(c["building"], c["weather"]): c["cls"] for c in cells}
seqs = collections.defaultdict(dict)
for f in glob.glob(f"{N}/results_h2/**/*.summary.json", recursive=True):
    s = json.load(open(f)); key = (s.get("building"), s.get("weather"))
    if key not in cohort or s.get("status") != "OK" or s.get("steps_done") != 8640 or not s.get("unif"): continue
    days = {e["day"]: e.get("class") for e in s["unif"]}
    if len(days) < 30: continue
    seq, last = [], None
    for d in range(1, 31):
        c = days.get(d) or last; seq.append(c); last = c
    seqs[key][s["policy_key"]] = seq
same = all(len({tuple(v) for v in arms.values()}) == 1 for arms in seqs.values())
assert same, "class sequences differ between arms of an episode"
ep = {k: next(iter(v.values())) for k, v in seqs.items()}
main = {k: v for k, v in ep.items() if cohort[k] == "H2-main"}; ext = {k: v for k, v in ep.items() if cohort[k] == "H2-ext"}
def spells(seq):
    out = []; cur, n = seq[0], 0
    for c in seq:
        if c == cur: n += 1
        else: out.append((cur, n)); cur, n = c, 1
    out.append((cur, n)); return out
def summarise(E):
    by = collections.defaultdict(list)
    for k, seq in E.items(): by[seq[0]].append(seq)
    res = {}
    for c0, group in sorted(by.items()):
        sp = [spells(s) for s in group]
        first = [x[0][1] for x in sp]; changes = [len(x) - 1 for x in sp]; share = [sum(1 for c in s if c == c0) / 30 for s in group]
        later = [n for x in sp for (c, n) in x[1:] if c != c0]                       # spells in other classes
        after_mild = [n for x in sp for (c, n) in x[1:] if c0 == "M" and c != "M"]   # spells a mild start moves into
        res[c0] = {"episodes": len(group), "class_changes_mean": statistics.mean(changes), "first_spell_mean_days": statistics.mean(first),
                   "first_spell_median_days": statistics.median(first), "share_days_in_first_class": statistics.mean(share),
                   "later_spells_in_other_classes_mean_days": statistics.mean(later) if later else None,
                   "later_spells_in_other_classes_median_days": statistics.median(later) if later else None,
                   "n_later_spells": len(later), "episodes_never_leaving": sum(1 for x in sp if len(x) == 1),
                   "design_class_agreement": None}
    trans = collections.Counter(); stay = collections.Counter()
    for seq in E.values():
        for a, b in zip(seq, seq[1:]):
            trans[a] += 1; stay[a] += (a == b)
    res["persistence_P_kk"] = {k: stay[k] / trans[k] for k in sorted(trans)}
    return res
R = {"note": "post hoc, descriptive; 2021 cohort (H2); classes as observed by the controller (unif log); invalid days keep the previous class",
     "episodes_main": len(main), "episodes_extension": len(ext), "arms_per_episode_checked_identical": True,
     "main": summarise(main), "extension": summarise(ext),
     "first_class_vs_design_class_main": {f"{a}->{b}": n for (a, b), n in collections.Counter((design[k], v[0]) for k, v in main.items()).items()}}
json.dump(R, open(f"{OUT}/P271_REGIME_SPELLS_2021.json", "w"), indent=1, default=str)
print(f"main episodes {len(main)}, extension {len(ext)}; class sequences identical across arms: {same}")
for part in ("main", "extension"):
    print(f"\n== {part}")
    for c0, v in R[part].items():
        if c0 == "persistence_P_kk": print("   P_kk:", {k: round(x, 3) for k, x in v.items()}); continue
        print(f"   start {c0}: n={v['episodes']}, changes {v['class_changes_mean']:.2f}, first spell mean {v['first_spell_mean_days']:.1f} d (median {v['first_spell_median_days']}), "
              f"days in first class {100 * v['share_days_in_first_class']:.0f}%, later spells in other classes mean {v['later_spells_in_other_classes_mean_days'] if v['later_spells_in_other_classes_mean_days'] is None else round(v['later_spells_in_other_classes_mean_days'], 1)} d "
              f"(median {v['later_spells_in_other_classes_median_days']}, n={v['n_later_spells']}), never left {v['episodes_never_leaving']}")
print("\nfirst observed vs design class (main):", R["first_class_vs_design_class_main"])
