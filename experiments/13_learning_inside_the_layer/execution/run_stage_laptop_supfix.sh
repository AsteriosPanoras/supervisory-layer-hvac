#!/usr/bin/env bash
# K07SUPFIX on the laptop (PREREG_K07SUPFIX_v1): the KRE-v2 kernel runner v3, the competition-tracker kit and the K07 scenario folder
# of the graph-network laptop stages (gnnsup/run_stage_laptop_gnn.sh), with the frozen layer runner run_episode_k07_v9sup.py.
# usage: run_stage_laptop_supfix.sh <WORKERS>
set -u
Z=$(cd "$(dirname "$0")" && pwd); N=$(dirname $Z); G0=<local>/jobs/issue1448_g0_2026-10-20; T=<tmp>/trk_kit
WORKERS=$1; SCEN=$N/k07x3/kaggle_k07x3_scen; WD=$Z/laptop/work_sf; mkdir -p $WD/tmp; rm -f $WD/END
export EXPERIMENT=K07SUPFIX_L STAGE=stage1 SHARD=0 NSHARDS=1 WORKERS=$WORKERS STEPS_OVERRIDE=0 KEEP_NPZ=1 RUNNER_WORK=$WD RUNNER_TMP=$WD/tmp RUNNER_EPISODE=run_episode_k07_v9sup.py
export RUNNER_LOCAL_ROOTS="$T:$SCEN:$Z/exp_lap" PYTHONHASHSEED=0 MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
echo "start $(date '+%F %T') exp=K07SUPFIX_L workers=$WORKERS scen=$SCEN" > $WD/run.log
nice -n 19 taskset -c 0-11 $G0/venv/bin/python $Z/exp_lap/runner/kernel_runner_v3.py >> $WD/run.log 2>&1 < /dev/null
rc=$?; if [ $rc -eq 0 ]; then echo "DONE rc=$rc $(date '+%F %T')" > $WD/END; else echo "FAILED rc=$rc $(date '+%F %T')" > $WD/END; fi
