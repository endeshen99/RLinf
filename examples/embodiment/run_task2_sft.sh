#!/bin/bash
# Train the task-2 SFT condition on one LeRobot dataset with the GRPO-matched episode-pass budget, then export the
# quarter-point checkpoints in the /ckpts/exports/step20 layout.
# Usage (inside the rlinf-osmesa container): bash examples/embodiment/run_task2_sft.sh <dataset_root> [export_dir] [extra hydra overrides...]
#   EPISODE_PASSES (default 1280 = 5 iters x 64 eps x 4 update epochs) sets the budget;
#   MAX_STEPS (default 2118) caps the computed budget; SMOKE_STEPS=<n> replaces it with n steps (save every n/4, min 1).
set -euo pipefail
DATASET=${1:?dataset root}; EXPORT_DIR=${2:-/ckpts/exports/task2_sft}; shift $(( $# > 1 ? 2 : $# ))
CONFIG_NAME=${CONFIG_NAME:-task2_sft_step20}
EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"; REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export EMBODIED_PATH PYTHONPATH=${REPO_PATH}:${PYTHONPATH:-}
GBS=$(python -c "import yaml,sys;print(yaml.safe_load(open('$EMBODIED_PATH/config/$CONFIG_NAME.yaml'))['actor']['global_batch_size'])")
read -r N_EP N_FR < <(python -c "import json;d=json.load(open('$DATASET/meta/info.json'));print(d['total_episodes'],d['total_frames'])")
if [ -n "${SMOKE_STEPS:-}" ]; then STEPS=$SMOKE_STEPS; else
  STEPS=$(python -c "import math;print(math.ceil(${EPISODE_PASSES:-1280}/$N_EP*$N_FR/$GBS))"); fi
CAP=${MAX_STEPS:-2118}   # budget cap: ~5 s/step on one A100 makes the uncapped 1280-pass budget ~11 h for the 250-episode set
if [ -z "${SMOKE_STEPS:-}" ] && [ "$STEPS" -gt "$CAP" ]; then echo "[task2_sft] capping max_steps $STEPS -> $CAP (MAX_STEPS)"; STEPS=$CAP; fi
SAVE=$(( (STEPS + 3) / 4 )); [ "$SAVE" -lt 1 ] && SAVE=1   # saves at SAVE, 2*SAVE, 3*SAVE and the last step
echo "[task2_sft] dataset=$DATASET episodes=$N_EP frames=$N_FR global_batch=$GBS -> max_steps=$STEPS save_interval=$SAVE"
LOG_DIR=${LOG_DIR:-/results/task2_sft}; mkdir -p $LOG_DIR
python ${REPO_PATH}/examples/sft/train_vla_sft.py --config-path ${EMBODIED_PATH}/config/ --config-name ${CONFIG_NAME} \
  data.train_data_paths=$DATASET runner.max_steps=$STEPS runner.save_interval=$SAVE runner.logger.log_path=$LOG_DIR "$@" 2>&1 | tee -a $LOG_DIR/run_task2_sft.log
EXP=$(python -c "import yaml;print(yaml.safe_load(open('$EMBODIED_PATH/config/$CONFIG_NAME.yaml'))['runner']['logger']['experiment_name'])")
for d in $LOG_DIR/$EXP/checkpoints/global_step_*; do
  n=${d##*global_step_}; python ${REPO_PATH}/toolkits/standalone_eval_scripts/openpi/export_rlinf_ckpt.py $d /ckpts/exports/step20 $EXPORT_DIR/step$n
done
