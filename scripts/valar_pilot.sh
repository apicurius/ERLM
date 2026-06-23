#!/usr/bin/env bash
# Runs the thesis A/B pilot (lambda=0 control then lambda=0.2 treatment), 15 steps each,
# on the held 8x A40 alloc. Outputs -> ~/ERLM/outputs. Run via srun --overlap.
set -x
module load cuda/12.8.0 2>/dev/null || module load cuda 2>/dev/null || true
export PATH="$HOME/.local/bin:$PATH"
export UV_FROZEN=1
export RLM_TRAIN_EXEC_TIMEOUT_S=180
export RLM_TRAIN_WORKER_STARTUP_TIMEOUT_S=180  # tolerate cold worker imports on BeeGFS
export WANDB_MODE=offline
set +x  # do not trace secret values
[ -f "$HOME/.erlm-keep/wandb.key" ]  && export WANDB_API_KEY="$(cat "$HOME/.erlm-keep/wandb.key")"
[ -f "$HOME/.erlm-keep/openai.key" ] && export OPENAI_API_KEY="$(cat "$HOME/.erlm-keep/openai.key")"
set -x
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# datasets/model from the verified local cache (avoids slow throttled re-download).
# OpenAI judge uses api.openai.com (separate) so it still works offline-of-HF.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd "$HOME/ERLM/prime-rl"
export PATH="$HOME/ERLM/prime-rl/.venv/bin:$PATH"  # torchrun + venv console scripts on PATH

SNAP="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots"
dcp_guard(){ for S in "$SNAP"/*/; do if [ -d "$S/prime" ] && [ ! -f "$S/prime/.metadata" ]; then rm -rf "$S/prime"; echo "GUARD removed partial DCP $S/prime"; fi; done; }
purge(){ pkill -9 -f "[P]RIME-RL" 2>/dev/null; pkill -9 -f "[V]LLM" 2>/dev/null; pkill -9 -f "[p]t_elastic" 2>/dev/null; pkill -9 -f "[V]erifiers" 2>/dev/null; sleep 10; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader; }

echo "===== PILOT START $(date) on $(hostname) ====="
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
run_arm(){ local cfg="$1" log="$2"; echo "===== ARM $cfg @ $(date) ====="; dcp_guard; bash "$HOME/ERLM/scripts/launch_efficient.sh" "$cfg" > "$log" 2>&1; echo "ARM rc=$? @ $(date)"; purge; }
purge
run_arm "$HOME/ERLM/training/configs/pilot-l0.toml"  "$HOME/ERLM/outputs/pilot-l0.console.log"
run_arm "$HOME/ERLM/training/configs/pilot-l02.toml" "$HOME/ERLM/outputs/pilot-l02.console.log"
echo "PILOT_DONE $(date)"
