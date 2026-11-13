#!/usr/bin/env bash
# K07SUPFIX on host-B (PREREG_K07SUPFIX_v1, execution note 2 when used): the kernel_runner_v3, KRE-v2 overlay and competition-tracker
# kit of the Residual PPO blind stage (axsup/run_stage_host-B_axsup.sh), the K07-X3 blind scenes (ztsup/scen), and the frozen layer
# runner run_episode_k07_v9sup.py; the experiment folder is copied from k07supfix/exp_lap.
# usage (on host-B): run_stage_host-B_supfix.sh <WORKERS>
set -u
PB=<host-b>; H=$PB/k07supfix; PY=$PB/venv/bin/python
WORKERS=$1; WD=$H/work_sf; mkdir -p $WD/tmp; rm -f $WD/END
export EXPERIMENT=K07SUPFIX_L STAGE=stage1 SHARD=0 NSHARDS=1 WORKERS=$WORKERS STEPS_OVERRIDE=0 KEEP_NPZ=1 RUNNER_WORK=$WD RUNNER_TMP=$WD/tmp RUNNER_EPISODE=run_episode_k07_v9sup.py
export RUNNER_LOCAL_ROOTS="$PB/k07x2/trk_kit:$PB/ztsup/scen:$H/exp" PYTHONHASHSEED=0 MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
echo "start $(date '+%F %T') exp=K07SUPFIX_L workers=$WORKERS (host-B)" > $WD/run.log
nice -n 19 taskset -c 0-15 $PY $H/exp/runner/kernel_runner_v3.py >> $WD/run.log 2>&1 < /dev/null
rc=$?; if [ $rc -eq 0 ]; then echo "DONE rc=$rc $(date '+%F %T')" > $WD/END; else echo "FAILED rc=$rc $(date '+%F %T')" > $WD/END; fi
