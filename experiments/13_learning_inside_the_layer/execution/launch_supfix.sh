#!/usr/bin/env bash
# K07SUPFIX launcher (PREREG_K07SUPFIX_v1): waits until the graph-network laptop stage has ended, runs the stage with 5 workers, files the outputs (SUPFIX key -> results_k07supfix,
# SUP key -> results_k07supfix_repro, so the two laptop SUP re-runs are never mixed with the Kaggle SUP arm), runs p298 and writes
# laptop/SF_DONE, which the H3 scheduler waits for before it uses the laptop (H3 execution note 1).  It prints no reward itself.
set -u
Z=$(cd "$(dirname "$0")" && pwd); N=$(dirname $Z); PY=<repo>/.venv/bin/python
mkdir -p $Z/laptop; LOG=$Z/laptop/SF_LAUNCH.log; echo "$(date '+%F %T') waiting for the graph-network laptop stage" >> $LOG
while [ ! -s $N/gnnsup/laptop/work_gnb_pl/END ]; do sleep 60; done
echo "$(date '+%F %T') graph-network stage ended: $(cat $N/gnnsup/laptop/work_gnb_pl/END); starting K07SUPFIX" >> $LOG
bash $Z/run_stage_laptop_supfix.sh 5
echo "$(date '+%F %T') stage ended: $(cat $Z/laptop/work_sf/END)" >> $LOG
$PY -I - "$Z/laptop/work_sf/out" "$N/results_k07supfix" "$N/results_k07supfix_repro" >> $LOG 2>&1 <<'EOF'
import sys, os, glob, json, shutil
src, fix, rep = sys.argv[1:4]; os.makedirs(fix, exist_ok=True); os.makedirs(rep, exist_ok=True); n = {"fix": 0, "rep": 0}
for f in glob.glob(f"{src}/**/*.summary.json", recursive=True):
    d = fix if json.load(open(f))["policy_key"].endswith(",ALPHA=0") else rep
    shutil.copy2(f, d); z = f.replace(".summary.json", ".npz")
    if os.path.exists(z): shutil.copy2(z, d)
    n["fix" if d == fix else "rep"] += 1
print("filed", n)
EOF
$PY -I $Z/p298_supfix.py $Z/RESULT_K07SUPFIX.json >> $LOG 2>&1
st=$(cat $Z/laptop/work_sf/END); case "$st" in DONE*) echo "DONE $(date '+%F %T')" > $Z/laptop/SF_DONE;; *) echo "FAILED $(date '+%F %T')" > $Z/laptop/SF_DONE;; esac
echo "$(date '+%F %T') SF_DONE: $(cat $Z/laptop/SF_DONE)" >> $LOG
