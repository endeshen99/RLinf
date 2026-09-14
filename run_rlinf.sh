#!/bin/bash
# Run a command inside the rlinf-osmesa image with NFS mounts. Usage: run_rlinf.sh <name> <bash command...>
NFS=/lambda/nfs/continual-learning; NAME=$1; shift
docker run --rm --gpus all --shm-size 20g --network host --name "$NAME" \
  -v $NFS/rlinf:/workspace/RLinf -v $NFS/rlinf_ckpts:/ckpts -v $NFS/rollouts/act3_rl:/results \
  -w /workspace/RLinf -e PYTHONPATH=/workspace/RLinf -e MUJOCO_GL=osmesa -e PYOPENGL_PLATFORM=osmesa -e TOKENIZERS_PARALLELISM=false \
  rlinf-osmesa bash -lc "source switch_env openpi >/dev/null 2>&1; $*"
