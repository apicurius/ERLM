#!/usr/bin/env python3
"""Validate prime-rl / verifiers / W&B / LoRA-adapter training wiring.

This is a network-free structural check against the current Prime Intellect
prime-rl config surface:

- local env packages are registered as `verifiers.environments` entry points;
- every RLM Qwen3-30B config has a shared `[wandb]` block so prime-rl's `rl`
  launcher logs trainer + orchestrator metrics into one W&B run;
- LoRA settings match the released adapter shape (r=32, alpha=64, q/k/v/o);
- prime-rl is instructed to save the raw PEFT adapter directory
  `weights/step_N/lora_adapters/`;
- configs use current `prime-rl` names (`group_size`,
  `pre_batch_filters`) instead of old aliases/removed fields;
- the upload helper publishes only the adapter directory, not merged full-model
  weights.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "training/configs"
ENVS = ROOT / "training/environments"

EXPECTED_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
CONFIGS = sorted(CONFIG_DIR.glob("rlm-qwen3-30b-*.toml"))
ENV_PACKAGES = ["oolong", "oolong_pairs", "browsecomp_plus", "longbench_codeqa"]


def _read(path: Path) -> dict:
    return tomllib.loads(path.read_text())


def _check_config(path: Path) -> dict[str, bool]:
    cfg = _read(path)
    text = path.read_text()
    trainer = cfg.get("trainer", {})
    orchestrator = cfg.get("orchestrator", {})
    lora = trainer.get("model", {}).get("lora", {})
    trainer_ckpt = trainer.get("ckpt", {})
    weights = trainer_ckpt.get("weights", {})

    return {
        f"{path.name}:has_wandb_project": bool(cfg.get("wandb", {}).get("project")),
        f"{path.name}:has_wandb_name": bool(cfg.get("wandb", {}).get("name")),
        f"{path.name}:lora_rank_32": lora.get("rank") == 32,
        f"{path.name}:lora_alpha_64": float(lora.get("alpha", 0.0)) == 64.0,
        f"{path.name}:lora_targets_match_hf_adapter": lora.get("target_modules")
        == EXPECTED_LORA_TARGETS,
        f"{path.name}:saves_adapter_separately": weights.get("save_adapter_separately")
        is True,
        f"{path.name}:has_orchestrator_lora_name": bool(
            orchestrator.get("model", {}).get("lora", {}).get("name")
        ),
        f"{path.name}:uses_group_size_not_old_rollouts_alias": "rollouts_per_example" not in text,
        f"{path.name}:uses_pre_batch_filters_not_removed_filters": "[[orchestrator.filters]]"
        not in text
        and "[[orchestrator.pre_batch_filters]]" in text,
        f"{path.name}:does_not_use_removed_eval_base_model": "eval_base_model" not in text,
        f"{path.name}:runs_startup_base_eval": cfg.get("orchestrator", {})
        .get("eval", {})
        .get("skip_first_step")
        is False,
    }


def _check_env_packages() -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for pkg in ENV_PACKAGES:
        pyproject = ENVS / pkg / "pyproject.toml"
        checks[f"{pkg}:pyproject_exists"] = pyproject.exists()
        if not pyproject.exists():
            checks[f"{pkg}:verifiers_entrypoint"] = False
            checks[f"{pkg}:depends_on_verifiers"] = False
            continue
        cfg = _read(pyproject)
        entrypoints = cfg.get("project", {}).get("entry-points", {}).get(
            "verifiers.environments", {}
        )
        deps = cfg.get("project", {}).get("dependencies", [])
        checks[f"{pkg}:verifiers_entrypoint"] = entrypoints.get(pkg) == f"{pkg}:load_environment"
        checks[f"{pkg}:depends_on_verifiers"] = any(str(d).startswith("verifiers") for d in deps)
    return checks


def main() -> int:
    checks: dict[str, bool] = {}
    checks["configs_exist"] = bool(CONFIGS)
    checks.update(_check_env_packages())
    for path in CONFIGS:
        checks.update(_check_config(path))

    helper = ROOT / "tools/upload_prime_rl_lora_adapter.py"
    checks["upload_helper_exists"] = helper.exists()
    if helper.exists():
        helper_text = helper.read_text()
        checks["upload_helper_targets_lora_adapters_dir"] = "lora_adapters" in helper_text
        checks["upload_helper_rejects_full_model_files"] = "FULL_MODEL_FILE_GLOBS" in helper_text

    for key, value in checks.items():
        print(f"{key}: {value}")
    if not all(checks.values()):
        print("FAILED: prime-rl/verifiers/W&B/LoRA adapter wiring is incomplete.")
        return 1
    print("PASSED: prime-rl/verifiers/W&B/LoRA adapter wiring is structurally valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
