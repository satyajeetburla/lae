#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/quad-swarm-rl:${PYTHONPATH:-}"

TRAIN_DIR="${TRAIN_DIR:-outputs/train_dir}"
EXPERIMENT="${EXPERIMENT:-02_sim2real_obst_density_see_1111_q.o.den_0.2}"
LAE_CONFIG="${LAE_CONFIG:-artifacts/lae/paper_h250_m10/config.json}"
DEVICE="${DEVICE:-cpu}"

if [[ "${TRAIN_DIR}" != /* ]]; then
  TRAIN_DIR="${ROOT_DIR}/${TRAIN_DIR}"
fi

python -m swarm_rl.enjoy \
  --env=quadrotor_multi \
  --train_dir="${TRAIN_DIR}" \
  --experiment="${EXPERIMENT}" \
  --device="${DEVICE}" \
  --rnn_size=10 \
  --sae_encoder_mode=13 \
  --lae_config="${LAE_CONFIG}" \
  --quads_init_mode=random \
  "$@"
