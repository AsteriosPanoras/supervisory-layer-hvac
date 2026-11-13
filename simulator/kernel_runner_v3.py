"""SB26 paper: the Kaggle run kernel for one stage of one experiment (NOT a submission kernel).  VERSION 3 = research environment KRE-v2 (9 Nov 2026): the VERSION 2 text below plus ONE functional change: in the writable simulator root the competition's smart_control/utils/scenario_generator.py is replaced by the starter-kit PR #1 file (git blob 1a3a01e8 -> 030f2437; the SimulationTracker no longer samples occupancy with VAV ids). The base and overlay sha256 are verified before any episode (fail-fast) and the effective sha256 of every simulator .py file is recorded; any other difference stops the kernel.
Previous docstring: VERSION 2 (2 Nov 2026, experiment E09): the text of kernel_runner.py plus two additions:
  * crossed scenarios: a job whose weather id is not a scenario of the scenarios dataset but a row of <experiment>/cross.csv (weather, station, start, convection_coefficient) runs the
    building of the first or second set on that weather window; its config.gin is the building's own text with the weather lines replaced by runner/cross_config.py, the other three
    files and weather/<station>.csv are linked; the sha256 of every crossed configuration goes into the manifest;
  * the default episode runner is run_episode_k07_v2.py (the same simulation plus a passive feature log for the default arm).

tools/make_run_kernels.py fills in EXPERIMENT, STAGE, SHARD and NSHARDS and writes one kernel folder per shard.  The kernel runs the jobs of
<experiment>/jobs/<stage>.jobs ("building weather policy_key steps", one per line) whose index % NSHARDS == SHARD, one process per episode and
one per CPU, and writes into /kaggle/working:
  out/<stage>/<tag>.summary.json and <tag>.npz      per run, the format of runner/run_episode_k07.py (the same as the pilots' runs)
  out/manifest_<stage>_shard<SHARD>.json            the session, the inputs (sha256 of the dataset markers) and every job's status
  logs/<stage>/<tag>.log                            each episode's stdout and stderr
tools/collect_outputs.py downloads them into the results dataset.

Inputs attached in kernel-metadata.json, found by content, not by mount name:
  * the competition data (the simulator package smart_buildings), read-only;
  * asteriospanoras/sb26-paper-scenarios: <set>/<building>/<weather>/{config.gin, config.yaml, floor_plan.npy, zone_map.npy} with config.gin
    paths relative to the simulator root, weather/<station>.csv, and scenarios.csv (the station of each scenario);
  * asteriospanoras/sb26-paper-experiments: k07/ (the published K07 classes), runner/run_episode_k07.py, <experiment>/jobs/<stage>.jobs.
It never starts the competition inference server and never writes a submission file.  Environment variables of the same names override
the settings; the local dry run uses RUNNER_LOCAL_ROOTS="<folder holding smart_buildings/>:<scenarios dataset>:<experiments dataset>" and
RUNNER_EPISODE=dryrun_episode.py (the kit's own baseline policy instead of K07).
"""
from __future__ import annotations

import csv
import glob
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def setting(name: str, default: str) -> str:
    value = os.environ.get(name, default)
    if value.startswith("__"):
        raise SystemExit(f"{name} is not filled in: build the kernel with tools/make_run_kernels.py")
    return value


EXPERIMENT = setting("EXPERIMENT", "__EXPERIMENT__")
STAGE = setting("STAGE", "__STAGE__")
SHARD = int(setting("SHARD", "__SHARD__"))                   # this kernel runs the jobs with index % NSHARDS == SHARD
NSHARDS = int(setting("NSHARDS", "__NSHARDS__"))
WORKERS = int(os.environ.get("WORKERS", "0"))                  # 0 = one per CPU the session reports (sched_getaffinity)
STEPS_OVERRIDE = int(os.environ.get("STEPS_OVERRIDE", "0"))    # 0 = the steps written in the jobs file
JOBS_LIMIT = int(os.environ.get("JOBS_LIMIT", "0"))            # 0 = every job of this shard; the local dry run takes the first few
KEEP_NPZ = os.environ.get("KEEP_NPZ", "1") == "1"           # per-step rewards; about 40 KB per 9-day episode, kept like the pilots'
BUDGET_S = float(os.environ.get("BUDGET_S", str(11.0 * 3600)))  # stop STARTING jobs after this; Kaggle's session cap is the hard limit
EPISODE = os.environ.get("RUNNER_EPISODE", "run_episode_k07_v2.py")
LOCAL_ROOT_ENV = "RUNNER_LOCAL_ROOTS"
KRE_VERSION = "KRE-v2"
KRE_OVERLAY_REL = "smart_control/utils/scenario_generator.py"
KRE_EXPECTED_BASE_SHA256 = "87f69478712b0eb0a72d1a65e4d9acdbadd1192b706f13e4525b0788c537912c"      # the competition data package file (git blob 1a3a01e862c9cbae7102ae08d4dff9101b0271a1)
KRE_EXPECTED_OVERLAY_SHA256 = "443aec77bcfe0e38d2dde2e7e12fdade490ef389f795d986c686883e7a0b90f3"   # starter-kit PR #1 head file (git blob 030f2437374055c42b2cd848615b9e19301f5991)

T0 = time.time()
WORK = Path(os.environ.get("RUNNER_WORK", "/kaggle/working"))
TMP = Path(os.environ.get("RUNNER_TMP", "/kaggle/tmp" if Path("/kaggle").exists() else "/tmp")) / "sb26_runner"
OUT = WORK / "out" / STAGE
LOGS = WORK / "logs" / STAGE
for d in (OUT, LOGS, TMP):
    d.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(f"[runner {time.time() - T0:7.0f}s] {msg}", flush=True)


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def package_files(pkg: Path) -> dict:
    """sha256 of every .py file of the simulator package, by relative path: shows which file differs between two simulator versions even
    when evaluate.py is the same."""
    return {str(p.relative_to(pkg)): sha(p) for p in sorted(q for q in pkg.rglob("*.py") if "__pycache__" not in q.parts)}


def apply_kre2_overlay(pkg: Path, comp: Path, exp: Path) -> dict:
    """KRE-v2: materialise smart_control/ and smart_control/utils/ of the writable simulator root as real directories (every other entry
    stays a link to the read-only competition files) and put the PR #1 tracker file in place; verify base, overlay and the effective tree."""
    real = comp / "smart_buildings"
    base_sha = sha(real / KRE_OVERLAY_REL)
    if base_sha != KRE_EXPECTED_BASE_SHA256:
        raise SystemExit(f"KRE-v2: the competition tracker file is not the frozen base ({base_sha}); refusing to run")
    src = one([p for p in exp.rglob("scenario_generator.py") if p.parent.name == "kre2_overlay"], "kre2_overlay/scenario_generator.py")
    if sha(src) != KRE_EXPECTED_OVERLAY_SHA256:
        raise SystemExit(f"KRE-v2: the overlay file is not the frozen PR #1 file ({sha(src)}); refusing to run")
    d = pkg
    for name, keep_out in (("smart_control", "utils"), ("utils", "scenario_generator.py")):
        real = real / name
        link = d / name
        if link.is_symlink():
            link.unlink()
        link.mkdir()
        for child in real.iterdir():
            if child.name not in (keep_out, "__pycache__"):
                (link / child.name).symlink_to(child)
        d = link
    shutil.copy2(src, d / "scenario_generator.py")
    comp_files = package_files(comp / "smart_buildings")
    effective = {rel: sha(pkg / rel) for rel in comp_files}
    differs = sorted(rel for rel in comp_files if effective[rel] != comp_files[rel])
    if differs != [KRE_OVERLAY_REL] or effective[KRE_OVERLAY_REL] != KRE_EXPECTED_OVERLAY_SHA256:
        raise SystemExit(f"KRE-v2: the effective simulator differs from the competition package in {differs}; expected only {KRE_OVERLAY_REL}")
    log(f"{KRE_VERSION}: tracker overlay applied ({KRE_OVERLAY_REL} {base_sha[:12]} -> {effective[KRE_OVERLAY_REL][:12]}); {len(comp_files)} simulator .py files checked, 1 differs")
    return {"kre_version": KRE_VERSION, "overlay_rel": KRE_OVERLAY_REL, "base_sha256": base_sha, "overlay_sha256": effective[KRE_OVERLAY_REL],
            "base_git_blob": "1a3a01e862c9cbae7102ae08d4dff9101b0271a1", "overlay_git_blob": "030f2437374055c42b2cd848615b9e19301f5991",
            "source": "starter-kit PR #1 (github.com/smart-buildings-2026/starter-kit/pull/1); hosts topic 749896", "effective_differs_from_competition": differs,
            "effective_smart_buildings_py_files": effective}


def lib_versions() -> dict:
    import importlib.metadata as md
    out = {}
    for name in ("numpy", "scipy", "pandas", "gin-config", "numba", "gymnasium", "protobuf", "absl-py", "matplotlib"):
        try:
            out[name] = md.version(name)
        except Exception:
            out[name] = None
    return out


def session_facts() -> dict:
    """What the session really provides (measured, recorded in the manifest)."""
    facts = {"python": sys.version.split()[0], "platform": platform.platform(), "os_cpu_count": os.cpu_count()}
    try:
        facts["affinity_cpus"] = len(os.sched_getaffinity(0))
    except Exception:
        pass
    try:
        mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        facts["MemTotal"], facts["MemAvailable"] = mem["MemTotal"].strip(), mem["MemAvailable"].strip()
        cpuinfo = Path("/proc/cpuinfo").read_text()
        facts["cpu_model"] = next((l.split(":", 1)[1].strip() for l in cpuinfo.splitlines() if l.startswith("model name")), None)
    except Exception:
        pass
    try:
        import numpy
        facts["numpy"] = numpy.__version__
    except Exception:
        pass
    return facts


def find_roots() -> tuple[Path, Path, Path]:
    """(folder that holds smart_buildings/, scenarios dataset, experiments dataset)."""
    if os.environ.get(LOCAL_ROOT_ENV):
        a, b, c = os.environ[LOCAL_ROOT_ENV].split(":")
        return Path(a), Path(b), Path(c)
    comp = next((Path(p).parent.parent for p in sorted(glob.glob("/kaggle/input/**/smart_buildings/evaluate.py", recursive=True))), None)
    scen = next((Path(p).parent for p in sorted(glob.glob("/kaggle/input/**/SB26_SCENARIOS_DATASET", recursive=True))), None)
    exp = next((Path(p).parent for p in sorted(glob.glob("/kaggle/input/**/SB26_EXPERIMENTS_DATASET", recursive=True))), None)
    if comp is None or scen is None or exp is None:
        raise SystemExit(f"inputs not found: competition={comp} scenarios={scen} experiments={exp}; "
                         f"/kaggle/input has {sorted(glob.glob('/kaggle/input/*/*'))}")
    return comp, scen, exp


def one(paths: list[Path], what: str) -> Path:
    if len(paths) != 1:
        raise SystemExit(f"expected exactly one {what}, found {[str(p) for p in paths]}")
    return paths[0]


def build_sim_root(comp: Path, scen: Path, needed: set[tuple[str, str]], cross: dict | None = None, cross_config=None) -> tuple[Path, dict]:
    """A writable simulator root: every entry of the competition's smart_buildings/ linked, and evaluation_scenarios/<b>/<w>/ rebuilt as
    real directories whose files link to the scenarios dataset; <b>/<w>/weather.csv links to weather/<station>.csv."""
    root = TMP / "simroot"
    if root.exists():
        shutil.rmtree(root)
    pkg = root / "smart_buildings"
    pkg.mkdir(parents=True)
    for entry in (comp / "smart_buildings").iterdir():
        if entry.name not in ("evaluation_scenarios", "__pycache__"):
            (pkg / entry.name).symlink_to(entry)
    station = {(r["building"], r["weather"]): r["station"] for r in csv.DictReader(open(scen / "scenarios.csv"))}
    where = {}
    for cfg in scen.rglob("config.gin"):
        key = (cfg.parent.parent.name, cfg.parent.name)
        if key in where:
            raise SystemExit(f"scenario {key} is in the scenarios dataset twice: {where[key]} and {cfg.parent}")
        where[key] = cfg.parent
    weather = {p.stem: p for p in scen.rglob("*.csv") if p.parent.name == "weather"}
    cross = cross or {}
    sources: dict[str, list[Path]] = {}
    for (b, w), p in where.items():
        sources.setdefault(b, []).append(p)
    crossed: dict[str, str] = {}
    for b, w in sorted(needed):
        if w in cross and (b, w) not in where:                      # a crossed scenario: the building b on the weather window w
            c = cross[w]
            if len(sources.get(b, [])) != 1 or c["station"] not in weather:
                raise SystemExit(f"cannot cross {b} with {w}: building folders {sources.get(b)}, station {c['station']}")
            src = sources[b][0]
            d = pkg / "evaluation_scenarios" / b / w
            d.mkdir(parents=True)
            text = cross_config.cross_config((src / "config.gin").read_text(), b, src.name, w, c["convection_coefficient"], c["start"])
            (d / "config.gin").write_text(text)
            crossed[f"{b}/{w}"] = hashlib.sha256(text.encode()).hexdigest()
            for name in ("config.yaml", "floor_plan.npy", "zone_map.npy"):
                if (src / name).exists():
                    (d / name).symlink_to(src / name)
            (d / "weather.csv").symlink_to(weather[c["station"]])
            (d / "metrics").mkdir()
            continue
        if (b, w) not in where or (b, w) not in station:
            raise SystemExit(f"scenario {b}/{w} is not in the scenarios dataset")
        d = pkg / "evaluation_scenarios" / b / w
        d.mkdir(parents=True)
        for name in ("config.gin", "config.yaml", "floor_plan.npy", "zone_map.npy"):
            if (where[(b, w)] / name).exists():
                (d / name).symlink_to(where[(b, w)] / name)
        (d / "weather.csv").symlink_to(weather[station[(b, w)]])
        (d / "metrics").mkdir()
    log(f"simulator root {root}: {len(needed)} scenarios, {len(crossed)} of them crossed")
    return root, crossed


def prepare_tools(exp: Path) -> tuple[Path, Path]:
    """Copy the K07 classes and the episode runner to a writable place: (tools dir, K07 dir)."""
    tools = TMP / "tools"
    if tools.exists():
        shutil.rmtree(tools)
    (tools / "k07").mkdir(parents=True)
    for name in ("their_k07_policy.py", "k07_submission_constants.json"):
        shutil.copy2(one([p for p in exp.rglob(name) if p.parent.name == "k07"], f"k07/{name}"), tools / "k07" / name)
    shutil.copy2(one([p for p in exp.rglob(EPISODE) if p.parent.name == "runner"], f"runner/{EPISODE}"), tools / EPISODE)
    return tools, tools / "k07"


def main() -> int:
    facts = session_facts()
    log("session: " + json.dumps(facts))
    comp, scen, exp = find_roots()
    log(f"competition root {comp} | scenarios {scen} | experiments {exp}")
    jobs_file = one([p for p in exp.rglob(f"{STAGE}.jobs") if p.parent.name == "jobs" and p.parent.parent.name == EXPERIMENT],
                    f"{EXPERIMENT}/jobs/{STAGE}.jobs")
    jobs_all = [l.split() for l in jobs_file.read_text().splitlines() if l.strip() and not l.startswith("#")]
    bad = [j for j in jobs_all if len(j) != 4]
    if bad:
        raise SystemExit(f"job lines need 4 fields (building weather policy_key steps): {bad[:3]}")
    jobs = [j for i, j in enumerate(jobs_all) if i % NSHARDS == SHARD]
    if JOBS_LIMIT:
        jobs = jobs[:JOBS_LIMIT]
    cross_files = [p for p in exp.rglob("cross.csv") if p.parent.name == EXPERIMENT]
    cross = {r["weather"]: r for r in csv.DictReader(open(cross_files[0]))} if cross_files else {}
    cc_file = one([p for p in exp.rglob("cross_config.py") if p.parent.name == "runner"], "runner/cross_config.py")
    sys.path.insert(0, str(cc_file.parent))
    import cross_config
    sim_root, crossed = build_sim_root(comp, scen, {(j[0], j[1]) for j in jobs}, cross, cross_config)
    kre = apply_kre2_overlay(sim_root / "smart_buildings", comp, exp)
    tools, k07 = prepare_tools(exp)
    workers = WORKERS or len(os.sched_getaffinity(0))
    log(f"{EXPERIMENT} / {STAGE}, shard {SHARD}/{NSHARDS}: {len(jobs)} of {len(jobs_all)} jobs, {workers} workers, episode runner {EPISODE}")

    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
               NUMBA_NUM_THREADS="1", SB_SCENARIOS_DIR="smart_buildings/evaluation_scenarios", NUMBA_CACHE_DIR=str(TMP / "numba"),
               MPLCONFIGDIR=str(TMP / "mpl"), MPLBACKEND="Agg")

    def run(job: list[str]) -> dict:
        b, w, key, steps = job[0], job[1], job[2], int(STEPS_OVERRIDE or job[3])
        sk = key.replace('@', '__').replace(',', '_').replace('=', '-')
        if len(sk) > 100: sk = sk[:50] + '_h' + hashlib.sha1(key.encode()).hexdigest()[:16]    # N10-S: same rule as the episode script
        tag = f"{b}_{w}_{sk}_{steps}"
        if time.time() - T0 > BUDGET_S:
            return {"tag": tag, "status": "NOT_STARTED_BUDGET"}
        started = time.time()
        with open(LOGS / f"{tag}.log", "w") as fh:
            rc = subprocess.call([sys.executable, "-I", str(tools / EPISODE), str(sim_root), str(k07), str(OUT), b, w, key, str(steps)],
                                 stdout=fh, stderr=subprocess.STDOUT, env=env)
        if not KEEP_NPZ:
            (OUT / f"{tag}.npz").unlink(missing_ok=True)
        r = {"tag": tag, "exit": rc, "wall_s": round(time.time() - started, 1)}
        try:
            s = json.loads((OUT / f"{tag}.summary.json").read_text())
            r.update(status=s["status"], score=s["score_abs_mean_reward"], steps_done=s["steps_done"], kit_commit=s["kit_commit"])
        except Exception as e:
            r.update(status="NO_SUMMARY", error=str(e))
        log(json.dumps(r))
        return r

    with ThreadPoolExecutor(max_workers=workers) as pool:       # threads only wait on subprocesses: one process per episode
        results = list(pool.map(run, jobs))
    markers = {"scenarios": scen / "SB26_SCENARIOS_DATASET", "experiments": exp / "SB26_EXPERIMENTS_DATASET"}
    manifest = {"experiment": EXPERIMENT, "stage": STAGE, "shard": SHARD, "nshards": NSHARDS, "workers": workers, "episode_runner": EPISODE,
                "session": facts | {"lib_versions": lib_versions()}, "kre": kre, "inputs": {f"{k}_marker_sha256": sha(p) for k, p in markers.items()} |
                {"evaluate_py_sha256": sha(comp / "smart_buildings" / "evaluate.py"), "jobs_file_sha256": sha(jobs_file),
                 "cross_csv_sha256": sha(cross_files[0]) if cross_files else "", "cross_config_py_sha256": sha(cc_file), "crossed_config_sha256": crossed,
                 "smart_buildings_py_files": package_files(comp / "smart_buildings")},
                "wall_s": round(time.time() - T0, 1), "n_jobs": len(jobs), "n_ok": sum(r.get("status") == "OK" for r in results),
                "results": results, "note": "SB26 paper run kernel; own synthetic scenarios; not a submission, not a leaderboard score"}
    (WORK / "out" / f"manifest_{STAGE}_shard{SHARD}.json").write_text(json.dumps(manifest, indent=2))
    log(f"done: {manifest['n_ok']} / {len(jobs)} OK in {manifest['wall_s']} s")
    return 0   # always 0: the per-job status is in the manifest and the summaries; a non-zero exit would only mark the kernel ERROR


if __name__ == "__main__":
    sys.exit(main())
