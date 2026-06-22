#!/usr/bin/env python3
"""Validate the efficient-RLM config against upstream RLM code.

This intentionally inspects `git show HEAD:...` rather than importing the local
working tree, because this checkout may contain local experiments that are not
part of upstream alexzhang13/rlm.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import tomllib
from pathlib import Path

CONFIG = Path("training/configs/rlm-qwen3-30b-efficient.toml")
UPSTREAM_OOLONG = "HEAD:training/environments/oolong/oolong/env.py"
UPSTREAM_EXAMPLE = "HEAD:training/configs/rlm-qwen3-30b-example.toml"
LOCAL_ONLY_ENV_KNOBS = {
    "subcall_penalty_rate",
    "free_context_reads",
    "per_call_tokens",
    "gate_floor",
}


def git_show(pathspec: str) -> str:
    return subprocess.check_output(["git", "show", pathspec], text=True)


def load_environment_params() -> set[str]:
    tree = ast.parse(git_show(UPSTREAM_OOLONG))
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "load_environment"
    )
    return {a.arg for a in fn.args.args} | {a.arg for a in fn.args.kwonlyargs}


def collect_env_args(config: dict) -> set[str]:
    used: set[str] = set()
    for env in config["orchestrator"]["train"]["env"]:
        used |= set(env.get("args", {}))
    for env in config["orchestrator"]["eval"]["env"]:
        used |= set(env.get("args", {}))
    return used


def collect_sections(value: object, prefix: str = "") -> set[str]:
    out: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}{key}"
            out.add(name)
            out |= collect_sections(child, f"{name}.")
    elif isinstance(value, list):
        for child in value:
            out |= collect_sections(child, f"{prefix}[].")
    return out


def main() -> int:
    cfg = tomllib.loads(CONFIG.read_text())
    upstream_example = tomllib.loads(git_show(UPSTREAM_EXAMPLE))

    used_args = collect_env_args(cfg)
    upstream_args = load_environment_params()
    unknown_args = sorted(used_args - upstream_args)
    leaked_knobs = sorted(used_args & LOCAL_ONLY_ENV_KNOBS)

    extra_sections = sorted(collect_sections(cfg) - collect_sections(upstream_example))

    errors: list[str] = []
    if unknown_args:
        errors.append(f"env args not accepted by upstream oolong.load_environment: {unknown_args}")
    if leaked_knobs:
        errors.append(f"local-only efficiency/cost knobs leaked into config: {leaked_knobs}")
    if extra_sections:
        errors.append(f"config uses sections absent from upstream example schema: {extra_sections}")

    print(f"config: {CONFIG}")
    print(f"env args used: {sorted(used_args)}")
    print(f"upstream env args: {sorted(upstream_args)}")
    print(f"unknown env args: {unknown_args or 'NONE'}")
    print(f"local-only knob leakage: {leaked_knobs or 'NONE'}")
    print(f"extra schema sections vs upstream example: {extra_sections or 'NONE'}")

    if errors:
        print("\nFAILED:", file=sys.stderr)
        for err in errors:
            print(f"- {err}", file=sys.stderr)
        return 1

    print("\nPASSED: efficient config is upstream-compatible and contains no local-only cost knobs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
