#!/usr/bin/env bash
# K07SUPFIX on host-B.
# 1. checks that the blind scenes, the tracker kit and the venv exist there; 2. copies exp_lap and the stage script; 3. starts the stage
# detached; 4. waits for its END (one ssh read a minute: this script is the exit watcher); 5. copies the outputs back, files them as
# launch_supfix.sh does (SUPFIX key -> results_k07supfix, SUP key -> results_k07supfix_repro), runs p298 and writes host-B/SF_DONE.
# usage: launch_supfix_host-B.sh <WORKERS>
set -u
Z=$(cd "$(dirname "$0")" && pwd); N=$(dirname $Z); PY=<repo>/.venv/bin/python
W=${1:?workers}; SSHK=~/.ssh/<key>; PH=<host-b>; SSH="ssh -i $SSHK -o BatchMode=yes -o ConnectTimeout=25"
PB=<host-b>; H=$PB/k07supfix; LOG=$Z/laptop/SF_host-B.log; mkdir -p $Z/laptop
say() { echo "$(date '+%F %T') $*" >> $LOG; }
$SSH $PH "test -d $PB/ztsup/scen && test -d $PB/k07x2/trk_kit && test -x $PB/venv/bin/python && grep -q building_52200 $PB/ztsup/scen/scenarios.csv" || { say "host-B kit check FAILED"; exit 2; }
$SSH $PH "mkdir -p $H/exp" && rsync -a -e "$SSH" $Z/exp_lap/ $PH:$H/exp/ && rsync -a -e "$SSH" $Z/run_stage_host-B_supfix.sh $PH:$H/ || { say "copy FAILED"; exit 3; }
$SSH $PH "cd $H && nohup bash run_stage_host-B_supfix.sh $W > /dev/null 2>&1 &" && say "started on host-B with $W workers"
until [ -n "$($SSH $PH "cat $H/work_sf/END 2>/dev/null")" ]; do sleep 60; done
say "stage ended: $($SSH $PH "cat $H/work_sf/END")"
mkdir -p $Z/host-B/work_sf && rsync -a -e "$SSH" --include='*/' --include='*.summary.json' --include='*.npz' --include='run.log' --exclude='*' $PH:$H/work_sf/ $Z/host-B/work_sf/
$PY -I - "$Z/host-B/work_sf/out" "$N/results_k07supfix" "$N/results_k07supfix_repro" >> $LOG 2>&1 <<'EOF'
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
st=$($SSH $PH "cat $H/work_sf/END"); case "$st" in DONE*) echo "DONE $(date '+%F %T')" > $Z/host-B/SF_DONE;; *) echo "FAILED $(date '+%F %T')" > $Z/host-B/SF_DONE;; esac
say "SF_DONE: $(cat $Z/host-B/SF_DONE)"
