#!/usr/bin/env python3
"""Upload only a prime-rl LoRA adapter checkpoint to Hugging Face.

prime-rl writes raw PEFT adapters to:

    <output_dir>/weights/step_<N>/lora_adapters/

when `[trainer.ckpt.weights] save_adapter_separately = true` is set. That
directory is the HF adapter artifact (typically tens of MB for the RLM Qwen3
30B A3B r=32 q/k/v/o adapter). The parent `step_<N>/` directory also contains
merged full-model weights and should not be uploaded to an adapter-only repo.

Examples:

    # Validate latest adapter path without network writes.
    python tools/upload_prime_rl_lora_adapter.py outputs/rlm-qwen3-30b-thesis --dry-run

    # Upload only the adapter files.
    python tools/upload_prime_rl_lora_adapter.py outputs/rlm-qwen3-30b-thesis \\
      --step 200 --repo-id mit-oasys/rlm-qwen3-30b-a3b-v0.1 --commit-message "Upload LoRA adapter"
"""

from __future__ import annotations

import argparse
import fnmatch
import subprocess
from pathlib import Path

FULL_MODEL_FILE_GLOBS = (
    "model*.safetensors",
    "model*.bin",
    "pytorch_model*.bin",
    "consolidated*.safetensors",
    "consolidated*.bin",
    "*.index.json",
)
ADAPTER_WEIGHT_GLOBS = (
    "adapter_model.safetensors",
    "adapter_model.bin",
    "adapter_model*.safetensors",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="prime-rl output_dir containing weights/")
    parser.add_argument("--step", type=str, default="latest", help='Step number or "latest"')
    parser.add_argument("--repo-id", help="HF repo id to upload to, e.g. org/model-adapter")
    parser.add_argument("--revision", default="main", help="HF branch/revision (default: main)")
    parser.add_argument("--commit-message", default="Upload LoRA adapter", help="HF commit message")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only validate and print the adapter path/size; do not upload.",
    )
    return parser.parse_args()


def resolve_step(weights_dir: Path, step: str) -> Path:
    if step != "latest":
        step_dir = weights_dir / f"step_{int(step)}"
        if not step_dir.is_dir():
            raise FileNotFoundError(f"step directory not found: {step_dir}")
        return step_dir

    candidates: list[tuple[int, Path]] = []
    for path in weights_dir.glob("step_*"):
        if not path.is_dir():
            continue
        suffix = path.name.removeprefix("step_")
        if suffix.isdigit():
            candidates.append((int(suffix), path))
    if not candidates:
        raise FileNotFoundError(f"no step_* directories found under {weights_dir}")
    return max(candidates, key=lambda item: item[0])[1]


def validate_adapter_dir(adapter_dir: Path) -> int:
    if not adapter_dir.is_dir():
        raise FileNotFoundError(
            f"adapter directory not found: {adapter_dir}\n"
            "Set `[trainer.ckpt.weights] save_adapter_separately = true` and wait for a checkpoint."
        )
    config = adapter_dir / "adapter_config.json"
    if not config.is_file():
        raise FileNotFoundError(f"missing PEFT adapter config: {config}")
    files = [p for p in adapter_dir.rglob("*") if p.is_file()]
    if not any(any(fnmatch.fnmatch(p.name, pat) for pat in ADAPTER_WEIGHT_GLOBS) for p in files):
        raise FileNotFoundError(f"missing adapter_model weights in {adapter_dir}")
    forbidden = [
        p
        for p in files
        if any(fnmatch.fnmatch(p.name, pattern) for pattern in FULL_MODEL_FILE_GLOBS)
    ]
    if forbidden:
        joined = "\n".join(f"  - {p}" for p in forbidden)
        raise ValueError(
            "adapter directory contains full-model-looking files; refusing adapter-only upload:\n"
            f"{joined}"
        )
    return sum(p.stat().st_size for p in files)


def human_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def main() -> int:
    args = parse_args()
    weights_dir = args.output_dir / "weights"
    step_dir = resolve_step(weights_dir, args.step)
    adapter_dir = step_dir / "lora_adapters"
    size = validate_adapter_dir(adapter_dir)
    print(f"Adapter directory: {adapter_dir}")
    print(f"Adapter payload size: {human_size(size)}")

    if args.dry_run:
        print("Dry run: not uploading.")
        return 0
    if not args.repo_id:
        raise ValueError("--repo-id is required unless --dry-run is set")

    cmd = [
        "hf",
        "upload",
        args.repo_id,
        str(adapter_dir),
        ".",
        "--revision",
        args.revision,
        "--commit-message",
        args.commit_message,
    ]
    print("Uploading adapter only with:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
