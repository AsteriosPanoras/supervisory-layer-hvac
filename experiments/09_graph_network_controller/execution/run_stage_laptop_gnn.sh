#!/usr/bin/env bash
# GNN stages on the laptop: the same kernel_runner_v3, KRE-v2 overlay and competition-tracker kit as the Residual PPO test
# (axsup/run_stage_laptop_ax.sh), with the GNN runner.  MODE = rec (GNN_REC_DIR=<work>/rec) | gnn:<weights file> | ax (neither).
# usage: run_stage_laptop_gnn.sh <EXP> <WORKERS> <NAME> <MODE> [CORES] [SCEN]
set -u
Z=$(cd "$(dirname "$0")" && pwd); H=$Z/laptop; N=$(dirname $Z); G0=<local>/jobs/issue1448_g0_2026-10-20; T=<tmp>/trk_kit
EXP=$1; WORKERS=$2; NAME=$3; MODE=$4; CORES=${5:-0-11}; SCEN=${6:-$N/kaggle_k07x_scen}; WD=$H/work_$NAME; mkdir -p $WD/tmp; rm -f $WD/END
export EXPERIMENT=$EXP STAGE=stage1 SHARD=0 NSHARDS=1 WORKERS=$WORKERS STEPS_OVERRIDE=0 KEEP_NPZ=1 RUNNER_WORK=$WD RUNNER_TMP=$WD/tmp RUNNER_EPISODE=run_episode_k07_v9supe_gnn.py
export RUNNER_LOCAL_ROOTS="$T:$SCEN:$Z/exp_lap" PYTHONHASHSEED=0 MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export AX_DIR=$N/official_adapter/public_src/alex_dataset AX_TORCH=$N/axsup/torch_site GNN_DIR=$Z
unset GNN_REC_DIR GNN_WEIGHTS
case "$MODE" in rec) mkdir -p $WD/rec; export GNN_REC_DIR=$WD/rec;; gnn:*) export GNN_WEIGHTS=${MODE#gnn:};; ax) ;; *) echo "bad mode $MODE"; exit 2;; esac
echo "start $(date '+%F %T') exp=$EXP workers=$WORKERS mode=$MODE cores=$CORES scen=$SCEN" > $WD/run.log
nice -n 19 taskset -c $CORES $G0/venv/bin/python $Z/exp_lap/runner/kernel_runner_v3.py >> $WD/run.log 2>&1 < /dev/null
rc=$?; if [ $rc -eq 0 ]; then echo "DONE rc=$rc $(date '+%F %T')" > $WD/END; else echo "FAILED rc=$rc $(date '+%F %T')" > $WD/END; fi
