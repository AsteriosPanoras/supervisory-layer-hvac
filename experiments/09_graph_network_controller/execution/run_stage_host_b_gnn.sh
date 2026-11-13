#!/usr/bin/env bash
# GNN (+ unchanged supervisor) blind stage on host-B: kernel_runner_v3 + KRE-v2 overlay on the competition-tracker kit, the blind
# scenes (ztsup/scen, as the Residual PPO test), AX_DIR/AX_TORCH as the Residual PPO test, GNN_DIR = this folder, GNN_WEIGHTS = $4.
# usage: run_stage_host-B_gnn.sh <EXPERIMENT> <WORKERS> <NAME> <WEIGHTS>
set -u
PB=<host-b>; H=$PB/gnnsup; PY=$PB/venv/bin/python
EXP=$1; WORKERS=$2; NAME=$3; WEIGHTS=$4; WD=$H/work_$NAME; mkdir -p $WD/tmp; rm -f $WD/END
export EXPERIMENT=$EXP STAGE=stage1 SHARD=0 NSHARDS=1 WORKERS=$WORKERS STEPS_OVERRIDE=0 KEEP_NPZ=1 RUNNER_WORK=$WD RUNNER_TMP=$WD/tmp RUNNER_EPISODE=run_episode_k07_v9supe_gnn.py
export RUNNER_LOCAL_ROOTS="$PB/k07x2/trk_kit:$PB/ztsup/scen:$H/exp" PYTHONHASHSEED=0 MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export AX_DIR=$PB/axsup/alex_dataset AX_TORCH=$PB/axsup/torch_site GNN_DIR=$H GNN_WEIGHTS=$WEIGHTS
unset GNN_REC_DIR
echo "start $(date '+%F %T') exp=$EXP workers=$WORKERS weights=$WEIGHTS" > $WD/run.log
nice -n 19 $PY $H/exp/runner/kernel_runner_v3.py >> $WD/run.log 2>&1 < /dev/null
rc=$?; if [ $rc -eq 0 ]; then echo "DONE rc=$rc $(date '+%F %T')" > $WD/END; else echo "FAILED rc=$rc $(date '+%F %T')" > $WD/END; fi
