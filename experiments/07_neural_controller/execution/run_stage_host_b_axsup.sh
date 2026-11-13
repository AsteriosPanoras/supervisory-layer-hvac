#!/usr/bin/env bash
# Residual PPO (+ unchanged supervisor) stage on host-B: kernel_runner_v3 + KRE-v2 overlay on the competition-tracker kit, the K07-X3 blind
# scenes of the ZT Best test (ztsup/scen), AX_DIR = the controller's public files, AX_TORCH = torch only (appended to sys.path by the runner).
# usage: run_stage_host-B_axsup.sh <EXPERIMENT> <WORKERS> <NAME>
set -u
PB=<host-b>; H=$PB/axsup; PY=$PB/venv/bin/python
EXP=$1; WORKERS=$2; NAME=$3; WD=$H/work_$NAME; mkdir -p $WD/tmp; rm -f $WD/END
export EXPERIMENT=$EXP STAGE=stage1 SHARD=0 NSHARDS=1 WORKERS=$WORKERS STEPS_OVERRIDE=0 KEEP_NPZ=1 RUNNER_WORK=$WD RUNNER_TMP=$WD/tmp RUNNER_EPISODE=run_episode_k07_v9supe_ax.py
export RUNNER_LOCAL_ROOTS="$PB/k07x2/trk_kit:$PB/ztsup/scen:$H/exp" PYTHONHASHSEED=0 MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export AX_DIR=$H/alex_dataset AX_TORCH=$H/torch_site
echo "start $(date '+%F %T') exp=$EXP workers=$WORKERS ax_dir=$AX_DIR" > $WD/run.log
nice -n 19 $PY $H/exp/runner/kernel_runner_v3.py >> $WD/run.log 2>&1 < /dev/null
rc=$?; if [ $rc -eq 0 ]; then echo "DONE rc=$rc $(date '+%F %T')" > $WD/END; else echo "FAILED rc=$rc $(date '+%F %T')" > $WD/END; fi
