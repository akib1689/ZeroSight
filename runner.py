#!/usr/bin/env python3
"""Runner — dataset-agnostic benchmark driver.

Usage:
    uv run python runner.py --dataset hotpotqa --model nvidia-gpt-oss-120b --samples 5 --trace
    uv run python runner.py --dataset musique --model zai --samples 10
    uv run python runner.py --resume results/bird_glm-5.3-flash_eag_<ts>_201.json
    uv run python runner.py --retry-errors results/bird_glm-5.3-flash_eag_<ts>_201.json

The runner:
1. Loads the dataset adapter (HotpotQA, MuSiQue, …)
2. For each task: builds per-task tools, runs the ReAct agent, scores EM/F1
3. Aggregates results and writes a JSON file to results/

Results are rewritten atomically after every item, so an interrupted run
leaves a partial file behind. ``--resume <path>`` (which must be the only
argument) continues that file in place, taking all configuration from it.
``--retry-errors <path>`` re-runs only the items whose result has an
``error`` (e.g. provider rate limits) in place; ``truncated`` items are not
errors and are left alone. Works on partial and complete files alike.
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# These imports must come after load_dotenv() (model.py reads env vars at init).
from agent import ReactAgent  # noqa: E402
from benchmarks.registry import get_dataset, list_datasets  # noqa: E402
from model import build_chat_model, list_models  # noqa: E402

logger = logging.getLogger(__name__)


def _samples_arg(value: str) -> int | str:
    """Parse --samples: an integer count, or 'full' for the entire set.

    'full' is kept as the string sentinel so main() can tell it apart from
    "flag not passed" (None).
    """
    if value == "full":
        return "full"
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--samples must be an integer or 'full', got {value!r}"
        ) from exc


def _summarize(results: list[dict], n_tasks: int) -> dict:
    """Derive the run summary from per-item results (single source of truth)."""
    total_score = 0.0
    total_correct = 0
    total_tokens = 0
    errors = 0
    scoring_errors = 0
    truncated_count = 0
    total_unresolved = 0
    total_steps = 0
    total_tool_calls = 0
    total_obs_tokens = 0
    total_meta_tokens = 0
    total_data_tokens = 0
    total_data_redacted_tokens = 0
    exposure_by_tool: dict[str, dict] = {}
    metric_name = "score"

    for item in results:
        total_score += item.get("score", 0.0)
        if item.get("correct"):
            total_correct += 1
        total_tokens += (item.get("usage") or {}).get("total_tokens", 0)
        if item.get("error"):
            errors += 1
        if item.get("metric") == "scoring_error":
            scoring_errors += 1
        if item.get("truncated"):
            truncated_count += 1
        total_unresolved += item.get("unresolved_placeholders", 0)
        total_steps += item.get("n_steps", 0)
        total_tool_calls += item.get("n_tool_calls", 0)
        metric_name = item.get("metric", "score")
        for step in item.get("steps") or []:
            obs_tokens = step.get("observation_tokens", 0)
            tool_type = step.get("tool_type", "unknown")
            tool_name = step.get("action", "unknown")
            total_obs_tokens += obs_tokens
            if tool_type == "metadata":
                total_meta_tokens += obs_tokens
            elif tool_type == "data":
                total_data_tokens += obs_tokens
            elif tool_type == "data_redacted":
                total_data_redacted_tokens += obs_tokens
            if tool_name not in exposure_by_tool:
                exposure_by_tool[tool_name] = {"calls": 0, "tokens": 0}
            exposure_by_tool[tool_name]["calls"] += 1
            exposure_by_tool[tool_name]["tokens"] += obs_tokens

    return {
        "samples": n_tasks,
        "accuracy": round(total_correct / n_tasks, 4) if n_tasks else 0.0,
        "avg_score": round(total_score / n_tasks, 4) if n_tasks else 0.0,
        "metric": metric_name,
        "total_tokens": total_tokens,
        "total_observation_tokens": total_obs_tokens,
        "total_metadata_tokens": total_meta_tokens,
        "total_data_tokens": total_data_tokens,
        "mean_data_tokens_per_task": (
            round(total_data_tokens / n_tasks, 2) if n_tasks else 0
        ),
        "total_data_redacted_tokens": total_data_redacted_tokens,
        "mean_data_redacted_tokens_per_task": (
            round(total_data_redacted_tokens / n_tasks, 2) if n_tasks else 0
        ),
        "mean_steps_per_task": round(total_steps / n_tasks, 2) if n_tasks else 0,
        "mean_tool_calls_per_task": round(total_tool_calls / n_tasks, 2) if n_tasks else 0,
        "total_unresolved_placeholders": total_unresolved,
        "exposure_by_tool": exposure_by_tool,
        "errors": errors,
        "scoring_errors": scoring_errors,
        "truncated": truncated_count,
    }


def _load_results_file(path: Path, *, allow_complete: bool) -> dict:
    """Load and validate a results file for --resume / --retry-errors."""
    if not path.exists():
        raise SystemExit(f"Results file not found: {path}")
    if path.suffix == ".tmp":
        raise SystemExit(f"Results target is a temp file, not a results file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Results target is not valid JSON: {path} ({exc})") from exc
    cfg = payload.get("config")
    results = payload.get("results")
    if not isinstance(cfg, dict) or not isinstance(results, list):
        raise SystemExit(
            f"Results target is missing a valid 'config'/'results' block: {path}"
        )
    missing = [
        key for key in ("dataset", "model", "mode", "samples", "timestamp")
        if cfg.get(key) is None
    ]
    if missing:
        raise SystemExit(f"Results target config is missing {missing}: {path}")
    if not allow_complete and payload.get("summary") is not None:
        raise SystemExit(f"Resume target is already complete: {path}")
    if not results:
        raise SystemExit(
            f"Results target has no completed items — start a fresh run: {path}"
        )
    return payload


def run_benchmark(
    dataset_name: str | None,
    model_name: str | None,
    samples: int | None,
    max_steps: int | None = None,
    trace: bool = False,
    delay: float = 1.0,
    mode: str = "baseline",
    test_variant: str | None = None,
    resume_path: Path | None = None,
    retry_path: Path | None = None,
) -> dict:
    """Run the ReAct agent on a dataset and return aggregated results.

    Args:
        samples: Number of items to evaluate, or None for the full set.
        mode: "baseline" (raw tool observations) or "eag" (the ZeroSight mode:
        value-free shape-only observations; "eag" is the historical codename
        kept for artifact compatibility).
        test_variant: Optional adapter-specific test-split selector (e.g.
            HiTab: "subset" | "qualitycheck" | "original"). Ignored by
            adapters without a ``set_test_variant``.
        resume_path: Path to a partial results file from a previous run.
            When set, dataset/model/samples/max_steps/mode/test_variant are
            all taken from that file's config block and the remaining items
            are appended in place.
        retry_path: Like ``resume_path``, but re-runs only the items whose
            stored result has an ``error`` (in place, preserving order) and
            works on complete files too. ``truncated`` items are untouched.
    """
    results: list[dict] = []
    output_path: Path
    timestamp: str
    config_extra: dict = {}
    start = 0
    retry_indices: list[int] | None = None

    if resume_path is not None or retry_path is not None:
        source_path = resume_path if resume_path is not None else retry_path
        payload = _load_results_file(source_path, allow_complete=retry_path is not None)
        cfg = payload["config"]
        results = payload["results"]
        dataset_name = cfg["dataset"]
        model_name = cfg["model"]
        mode = cfg["mode"]
        test_variant = cfg.get("test_variant")
        max_steps = cfg.get("max_steps")
        samples = cfg["samples"]
        timestamp = cfg["timestamp"]
        output_path = source_path
        # Carry provenance from prior in-place invocations, then record this one.
        config_extra = {
            key: cfg[key]
            for key in ("resumed", "resume_events", "retried", "retry_events")
            if key in cfg
        }
        event_key, flag = (
            ("resume_events", "resumed")
            if resume_path is not None
            else ("retry_events", "retried")
        )
        events = config_extra.get(event_key)
        config_extra[event_key] = (
            [*events, datetime.now().strftime("%Y%m%d_%H%M%S")]
            if isinstance(events, list)
            else [datetime.now().strftime("%Y%m%d_%H%M%S")]
        )
        config_extra[flag] = True
    else:
        dataset_name = dataset_name or "hotpotqa"
        model_name = model_name or "nvidia-gpt-oss-120b"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    dataset = get_dataset(dataset_name)
    if test_variant and hasattr(dataset, "set_test_variant"):
        dataset.set_test_variant(test_variant)
    # Effective variant (adapter default when the flag is omitted), for reporting.
    resolved_variant = test_variant or getattr(dataset, "_test_variant", None)
    model = build_chat_model(model_name)
    tasks = dataset.load(limit=samples)
    eag_mode = mode == "eag"

    if max_steps is None:
        max_steps = dataset.default_max_steps

    if resume_path is not None or retry_path is not None:
        verb = "resume" if resume_path is not None else "retry"
        if len(tasks) != samples:
            raise SystemExit(
                f"Cannot {verb} {source_path}: the dataset now yields "
                f"{len(tasks)} tasks but the run started with {samples} "
                "(subset or source data changed since the run began)."
            )
        done_ids = [item["id"] for item in results]
        if retry_path is not None:
            prefix_ids = [task.id for task in tasks]
        else:
            prefix_ids = [task.id for task in tasks[: len(results)]]
        if done_ids != prefix_ids:
            raise SystemExit(
                f"Cannot {verb} {source_path}: stored results do not match "
                "the current task ordering (subset regenerated or dataset "
                "changed). Start a fresh run instead."
            )
        if resume_path is not None:
            start = len(results)
        else:
            retry_indices = [i for i, item in enumerate(results) if item.get("error")]
            if not retry_indices:
                raise SystemExit(f"No errored items to retry in: {source_path}")
            config_extra["retried_items"] = len(retry_indices)
    else:
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        output_path = results_dir / (
            f"{dataset.name}_{model_name}_{mode}_{timestamp}_{len(tasks)}.json"
        )

    if trace:
        logging.basicConfig(level=logging.INFO, format="  %(message)s")

    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset.name} | Model: {model_name} | Samples: {len(tasks)} | Mode: {mode}")
    if resume_path is not None:
        print(f"  Resuming {output_path.name} at item {start + 1}/{len(tasks)}")
    elif retry_indices is not None:
        print(f"  Retrying {len(retry_indices)} errored item(s) in {output_path.name}")
    print(f"{'='*60}\n")

    def _write_results(summary: dict | None) -> dict:
        payload = {
            "config": {
                "dataset": dataset.name,
                "model": model_name,
                "mode": mode,
                "max_steps": max_steps,
                "samples": len(tasks),
                "test_variant": resolved_variant,
                "timestamp": timestamp,
                **config_extra,
            },
            "summary": summary,
            "completed_items": len(results),
            "results": results,
        }
        tmp_path = output_path.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        os.replace(tmp_path, output_path)
        return payload

    indices = retry_indices if retry_indices is not None else range(start, len(tasks))
    for i in indices:
        task = tasks[i]
        print(f"[{i+1}/{len(tasks)}] {task.id}: {task.question[:80]}...")

        # Build per-task tools (search/lookup bound to this question's context)
        tools = dataset.make_tools(task, eag_mode=eag_mode)

        # Construct agent with per-task tools + dataset system prompt
        agent = ReactAgent(
            model=model,
            tools=tools,
            system_prompt=dataset.system_prompt,
            max_steps=max_steps,
            eag_mode=eag_mode,
        )

        # Run
        traj = agent.run(task.question, trace=trace)

        # Score using dataset-specific scorer
        if traj.error:
            print(f"  ERROR: {traj.error[:100]}")
            scores = {"correct": False, "score": 0.0, "metric": "error"}
        else:
            if traj.truncated:
                print(f"  TRUNCATED (ran out of turns after {traj.n_turns})")
            try:
                scores = dataset.score(traj, task)
            except Exception as e:
                print(f"  SCORING ERROR: {str(e)[:100]}")
                scores = {
                    "correct": False,
                    "score": 0.0,
                    "metric": "scoring_error",
                    "error_detail": str(e),
                }

        if traj.error:
            tag = "ERROR"
        elif scores.get("metric") == "scoring_error":
            tag = "SCORING ERROR"
        elif traj.truncated:
            tag = "TRUNCATED" if not scores.get("correct") else "TRUNCATED*"
        else:
            tag = "CORRECT" if scores.get("correct") else "WRONG"
        answer_preview = traj.answer[:80] if traj.answer else "(empty)"
        score_str = f"Score: {scores.get('score', 0):.2f}"
        if "em" in scores:
            score_str += f" EM: {scores['em']:.1f} F1: {scores.get('f1', 0):.2f}"
        print(
            f"  {tag} | Answer: {answer_preview} | Gold: {task.gold_answer[:60]} "
            f"| {score_str} | Turns: {traj.n_turns} (calls: {traj.n_tool_calls})"
            f" | {traj.latency_ms:.0f}ms"
        )
        print()

        result_item = {
            "id": task.id,
            "question": task.question,
            "answer": traj.answer,
            "answer_raw": traj.answer_raw,
            "unresolved_placeholders": traj.unresolved_placeholders,
            "gold": task.gold_answer,
            "correct": scores.get("correct", False),
            "score": scores.get("score", 0.0),
            "metric": scores.get("metric", "score"),
            "steps": traj.steps,
            "usage": traj.usage,
            "latency_ms": round(traj.latency_ms, 2),
            "error": traj.error,
            "truncated": traj.truncated,
            "n_steps": traj.n_turns,
            "n_tool_calls": traj.n_tool_calls,
        }
        # Include any extra scoring details
        for k, v in scores.items():
            if k not in result_item:
                result_item[f"score_{k}"] = v
        if retry_indices is not None:
            results[i] = result_item
        else:
            results.append(result_item)
        _write_results(None)

        if i < len(tasks) - 1:
            time.sleep(delay)

    # Aggregate — summary is derived from the results array, never accumulated
    n = len(tasks)
    summary = {
        "dataset": dataset.name,
        "model": model_name,
        "mode": mode,
        **_summarize(results, n),
        "timestamp": timestamp,
    }

    # Print summary
    total_correct = sum(1 for item in results if item.get("correct"))
    print(f"{'='*60}")
    print(f"  Results: {dataset.name} / {model_name}")
    print(f"{'='*60}")
    print(f"  Accuracy:    {summary['accuracy']:.4f} ({total_correct}/{n})")
    print(f"  Avg Score:   {summary['avg_score']:.4f} ({summary['metric']})")
    print(f"  Tokens:      {summary['total_tokens']}")
    print(
        f"  Exposure:    {summary['total_observation_tokens']} obs tokens "
        f"(metadata: {summary['total_metadata_tokens']}, "
        f"data: {summary['total_data_tokens']}, "
        f"redacted: {summary['total_data_redacted_tokens']})"
    )
    print(f"  Errors:      {summary['errors']}")
    print(f"  Scoring err: {summary['scoring_errors']}")
    print(f"  Truncated:   {summary['truncated']}")
    print()

    output = _write_results(summary)

    print(f"  Saved: {output_path}")
    print()

    return output


def main():
    parser = argparse.ArgumentParser(
        description="ReAct agent benchmark runner"
    )
    parser.add_argument(
        "--dataset",
        choices=list_datasets(),
        default=None,
        help="Dataset to evaluate on (default: hotpotqa)",
    )
    parser.add_argument(
        "--model",
        choices=list_models(),
        default=None,
        help="Model config from configs/models.yaml (default: nvidia-gpt-oss-120b)",
    )
    parser.add_argument(
        "--samples",
        type=_samples_arg,
        default=None,
        help="Number of items to evaluate, or 'full' for the entire set (default: 10)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Max agent reasoning steps (default: per-dataset, typically 12)",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Print step-by-step agent execution",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=None,
        help="Seconds between items (default: 1.0)",
    )
    parser.add_argument(
        "-e",
        "--zerosight",
        "--eag",
        action="store_true",
        help="Enable ZeroSight mode (value-free, shape-only tool observations "
        "with local placeholder resolution; --eag is a legacy alias). "
        "Default: baseline (raw observations).",
    )
    parser.add_argument(
        "--test-variant",
        choices=["qualitycheck", "original", "subset", "full"],
        default=None,
        help="Test-split variant for adapters that ship more than one "
        "(hitab: 'subset' default = committed 169-question stratified sample, "
        "'qualitycheck' = full corrected split, 'original' = pre-correction labels; "
        "bird/spider: 'subset' default = committed 200-question stratified sample, "
        "'full' = entire set)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume a partial results file in place. Must be the only argument: "
        "dataset, model, mode, variant, max-steps and samples all come from the file.",
    )
    parser.add_argument(
        "--retry-errors",
        type=Path,
        default=None,
        help="Re-run only the errored items of a results file in place (order "
        "preserved; truncated items untouched). Works on partial and complete "
        "files. Run configuration comes from the file; only --trace/--delay "
        "may accompany it.",
    )
    args = parser.parse_args()

    if args.resume is not None or args.retry_errors is not None:
        is_resume = args.resume is not None
        conflicts = [
            flag
            for flag, value in (
                ("--dataset", args.dataset),
                ("--model", args.model),
                ("--samples", args.samples),
                ("--max-steps", args.max_steps),
                ("--test-variant", args.test_variant),
                ("--zerosight", args.zerosight),
                ("--delay", args.delay if is_resume else None),
                ("--trace", args.trace if is_resume else False),
            )
            if value
        ]
        if args.resume is not None and args.retry_errors is not None:
            conflicts.append("--resume" if is_resume else "--retry-errors")
        if conflicts:
            mode_flag = "--resume" if is_resume else "--retry-errors"
            allowed = "" if is_resume else " (only --trace/--delay may accompany it)"
            parser.error(
                f"{mode_flag} takes no other options{allowed} (got "
                + ", ".join(conflicts)
                + "); all run configuration comes from the results file"
            )
        if is_resume:
            run_benchmark(
                dataset_name=None,
                model_name=None,
                samples=None,
                resume_path=args.resume,
            )
        else:
            run_benchmark(
                dataset_name=None,
                model_name=None,
                samples=None,
                retry_path=args.retry_errors,
                trace=args.trace,
                delay=args.delay if args.delay is not None else 1.0,
            )
        return

    samples = args.samples
    if samples == "full":
        samples = None
    elif samples is None:
        samples = 10

    run_benchmark(
        dataset_name=args.dataset,
        model_name=args.model,
        samples=samples,
        max_steps=args.max_steps,
        trace=args.trace,
        delay=args.delay if args.delay is not None else 1.0,
        mode="eag" if args.zerosight else "baseline",
        test_variant=args.test_variant,
    )


if __name__ == "__main__":
    main()
