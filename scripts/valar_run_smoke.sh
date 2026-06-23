#!/usr/bin/env bash
module load cuda/12.8.0 >/dev/null 2>&1 || true
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export RLM_TRAIN_EXEC_TIMEOUT_S=180
cd ~/ERLM/prime-rl
exec ~/ERLM/prime-rl/.venv/bin/python ~/ERLM/scripts/valar_full_smoke.py
