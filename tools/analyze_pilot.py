"""Analyze the lambda=0 vs lambda=0.2 RLM efficiency pilot.

Reads prime-rl **offline** W&B runs (no `wandb sync` needed) from each arm's
output dir and prints a control-vs-treatment comparison: eval accuracy delta
(base step-0 -> trained step-N), trajectory cost (mean + any logged percentiles,
per axis: iterations / sub-LLM calls / sub-LLM tokens), and shortcut diagnostics.

Each arm's output dir contains TWO offline runs:
  <out>/run_default/wandb/offline-run-*/run-*.wandb   orchestrator: eval + rollout/cost metrics
  <out>/wandb/offline-run-*/run-*.wandb               trainer: reward / loss

History items are read via the W&B DataStore, honoring `nested_key` (prime-rl logs
nested metric paths, so `item.key` is often empty).

Usage:
  python tools/analyze_pilot.py <pilot-l0-dir> <pilot-l02-dir>
  python tools/analyze_pilot.py --keys <any-output-dir>      # just dump discovered metric keys
"""

from __future__ import annotations

import glob
import json
import os
import sys


def read_wandb_history(wandb_file: str) -> dict[str, list[tuple]]:
    """Return {metric_key: [(step, value), ...]} from one offline .wandb file."""
    import wandb.proto.wandb_internal_pb2 as pb  # type: ignore
    from wandb.sdk.internal.datastore import DataStore

    ds = DataStore()
    ds.open_for_scan(wandb_file)
    series: dict[str, list[tuple]] = {}
    while True:
        try:
            rec = ds.scan_record()
        except Exception:
            break
        if rec is None:
            break
        r = pb.Record()
        try:
            r.ParseFromString(rec[1])
        except Exception:
            continue
        if not r.HasField("history"):
            continue
        items: dict[str, str] = {}
        for it in r.history.item:
            k = it.key or "/".join(it.nested_key)
            items[k] = it.value_json
        st = items.get("_step")
        try:
            st = int(json.loads(st)) if st is not None else None
        except Exception:
            st = None
        for k, vj in items.items():
            if k.startswith("_"):
                continue
            try:
                v = json.loads(vj)
            except Exception:
                v = vj
            if isinstance(v, (int, float)):
                series.setdefault(k, []).append((st, v))
    return series


def load_arm(out_dir: str) -> dict[str, list[tuple]]:
    """Merge the orchestrator + trainer offline runs for one arm."""
    merged: dict[str, list[tuple]] = {}
    pats = [
        os.path.join(out_dir, "run_default", "wandb", "offline-run-*", "run-*.wandb"),
        os.path.join(out_dir, "wandb", "offline-run-*", "run-*.wandb"),
    ]
    for pat in pats:
        for f in sorted(glob.glob(pat)):
            for k, v in read_wandb_history(f).items():
                merged.setdefault(k, []).extend(v)
    for k in merged:
        merged[k].sort(key=lambda sv: (sv[0] is None, sv[0]))
    return merged


def first_last(series: list[tuple]) -> tuple:
    """(first_value, last_value, first_step, last_step) ignoring None steps where possible."""
    vals = [(s, v) for s, v in series if v is not None]
    if not vals:
        return (None, None, None, None)
    return (vals[0][1], vals[-1][1], vals[0][0], vals[-1][0])


# Pattern groups for auto-selecting the metrics we care about.
EVAL_PAT = ("eval",)
ACC_PAT = ("reward", "correct", "score", "accuracy", "pass")
COST_PAT = ("iter", "sub_llm", "subcall", "sub_call", "token")
DIAG_PAT = (
    "usage_missing",
    "efficiency",
    "shaped",
    "gibberish",
    "repetition",
    "truncat",
    "zero_advantage",
)
PCTL_HINT = ("/p50", "/p90", "/p95", "/p99", "/median", "/mean", "/max", "/min", "/std")


def _match(key: str, pats) -> bool:
    kl = key.lower()
    return any(p in kl for p in pats)


def select(keys, *patgroups):
    out = []
    for k in sorted(keys):
        if all(any(p in k.lower() for p in pats) for pats in patgroups):
            out.append(k)
    return out


def dump_keys(out_dir: str) -> None:
    s = load_arm(out_dir)
    print(f"=== {len(s)} numeric metric keys in {out_dir} ===")
    for k in sorted(s):
        fv, lv, fs, ls = first_last(s[k])
        print(f"  {k}  steps[{fs}..{ls}]  first={fv}  last={lv}")


def report(d0: str, d02: str) -> None:
    a0, a2 = load_arm(d0), load_arm(d02)
    allkeys = set(a0) | set(a2)

    def fl(arm, k):
        return first_last(arm.get(k, []))

    print("=" * 78)
    print("RLM EFFICIENCY PILOT  --  lambda=0 (control) vs lambda=0.2 (treatment)")
    print(f"  control   = {d0}")
    print(f"  treatment = {d02}")
    print("=" * 78)

    # ---- EVAL ACCURACY (per env): base (step 0) -> trained (last) ----
    eval_acc = [
        k
        for k in select(allkeys, EVAL_PAT, ACC_PAT)
        if not _match(k, ("cancel", "error", "queued", "inflight"))
    ]
    print("\n## EVAL ACCURACY  (step0 base -> last trained;  D = trained-base)")
    print(f"{'metric':<48}{'ctrl base->last (D)':>22}{'treat base->last (D)':>24}")
    for k in eval_acc:
        b0, l0, _, _ = fl(a0, k)
        b2, l2, _, _ = fl(a2, k)
        d0v = (l0 - b0) if (b0 is not None and l0 is not None) else None
        d2v = (l2 - b2) if (b2 is not None and l2 is not None) else None

        def fmt(b, last, d):
            if b is None:
                return "n/a"
            return f"{b:.3f}->{last:.3f} ({d:+.3f})" if d is not None else f"{b:.3f}"

        print(f"{k:<48}{fmt(b0, l0, d0v):>22}{fmt(b2, l2, d2v):>24}")
    if not eval_acc:
        print("  (no eval-accuracy keys yet — eval not logged; rerun after a completed eval)")

    # ---- TRAJECTORY COST (per axis): mean + percentiles, control vs treatment ----
    cost = select(allkeys, COST_PAT)
    print("\n## TRAJECTORY COST  (last value; lower under treatment = efficiency working)")
    print(f"{'metric':<52}{'control(last)':>14}{'treat(last)':>14}{'D%':>9}")
    for k in cost:
        _, l0, _, _ = fl(a0, k)
        _, l2, _, _ = fl(a2, k)
        dpct = ((l2 - l0) / l0 * 100) if (l0 not in (None, 0) and l2 is not None) else None
        print(
            f"{k:<52}{(f'{l0:.2f}' if l0 is not None else 'n/a'):>14}"
            f"{(f'{l2:.2f}' if l2 is not None else 'n/a'):>14}"
            f"{(f'{dpct:+.1f}' if dpct is not None else ''):>9}"
        )
    if not cost:
        print("  (no cost keys yet)")

    # ---- SHORTCUT / HEALTH DIAGNOSTICS ----
    diag = select(allkeys, DIAG_PAT)
    print("\n## SHORTCUT / HEALTH DIAGNOSTICS  (control vs treatment, last)")
    for k in diag:
        _, l0, _, _ = fl(a0, k)
        _, l2, _, _ = fl(a2, k)
        print(f"  {k:<50} ctrl={l0}  treat={l2}")
    if not diag:
        print("  (no diagnostic keys yet)")

    # quick read for the decision rule
    print("\n## DECISION-RULE READ (heuristic)")
    print("  - efficiency working  : treatment cost (iterations/sub_llm/tokens) < control")
    print("  - NOT shortcutting     : treatment eval accuracy ~>= control (within noise)")
    print(
        "  - token axis live      : rlm_sub_llm_usage_missing ~ 0 (else token efficiency is a no-op)"
    )


def main() -> int:
    args = sys.argv[1:]
    if args and args[0] == "--keys":
        dump_keys(args[1])
        return 0
    if len(args) != 2:
        print(__doc__)
        return 2
    report(args[0], args[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
