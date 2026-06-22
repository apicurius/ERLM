#!/usr/bin/env bash
# Launch the efficient-RLM RL run using ONLY upstream harness code.
# Sets the per-REPL-block wall-clock guard (read by
# training/src/rlm_train/worker.py) to kill runaway sequential sub-call loops —
# the paper's "RLMs without asynchronous LM calls are slow" failure mode.
# Assumes prime-rl is installed and this repo + training env are wired in
# (see training/README.md). Run from the repo root.
set -euo pipefail

CFG="${1:-training/configs/rlm-qwen3-30b-efficient.toml}"
export RLM_TRAIN_EXEC_TIMEOUT_S="${RLM_TRAIN_EXEC_TIMEOUT_S:-180}"

echo "Launching efficient RLM RL: $CFG (RLM_TRAIN_EXEC_TIMEOUT_S=$RLM_TRAIN_EXEC_TIMEOUT_S)"
exec uv run rl @ "$CFG"
