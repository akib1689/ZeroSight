"""Freeze benchmark runs into results/paper-experiments/ for the paper.

Two legal frozen shapes per dataset x model cell (statistical significance):

- partial:  exactly 3 baseline + 3 EAG complete runs -> status "partial"
- complete: exactly 5 baseline + 5 EAG complete runs -> status "complete"

Nothing else is ever written to the manifest. When more runs than needed
exist (e.g. 4 baseline + 3 EAG), a partial freeze takes the FIRST 3 of each
mode by timestamp and leaves the extras in results/ for the completion step.

Usage::

    uv run python scripts/freeze_results.py --status
    uv run python scripts/freeze_results.py --dataset spider --model glm-5.3-flash  # dry-run
    uv run python scripts/freeze_results.py --dataset spider --model glm-5.3-flash --apply
    uv run python scripts/freeze_results.py --dataset spider --model glm-5.3-flash --partial --apply
    uv run python scripts/freeze_results.py --dataset spider --model glm-5.3-flash \
        --unfreeze [--apply]
    uv run python scripts/freeze_results.py --check

Command matrix:

- not frozen + full freeze:  needs exactly 5+5 in results/ -> status "complete"
- not frozen + --partial:    needs >= 3+3 in results/   -> first 3+3, status "partial"
- frozen-partial + full:     refused (completion is explicit)
- frozen-partial + --partial: needs exactly 2+2 NEW complete runs -> fills run-4/run-5,
  promotes to "complete"; fewer -> refused; extra complete runs are never moved
- frozen-complete:           no-op under both commands
- frozen-partial + --unfreeze: revert the freeze — moves run-1..run-3 back to results/
  under their original filenames (sha256-verified first) and drops the manifest cell;
  frozen-complete cells are refused. Note: a later re-freeze takes the first 3+3 by
  timestamp, which may select different runs than the previously frozen set.

Caveats: fix errored runs with ``runner.py --retry-errors`` BEFORE freezing a
cell; frozen files live under new paths, so post-freeze retries must be
re-frozen. Manifest is the single source of truth; summary.csv is always
regenerated from it. Committing the freeze is left to the operator (a partial
freeze and its later completion are naturally two commits).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FREEZE_DIR = RESULTS_DIR / "paper-experiments"
MANIFEST_PATH = FREEZE_DIR / "MANIFEST.json"
SUMMARY_CSV_PATH = FREEZE_DIR / "summary.csv"
EXPECTED_PER_MODE = 5
PARTIAL_PER_MODE = 3
MODES = ("baseline", "eag")

FILE_RE = re.compile(
    r"^(?P<dataset>[a-z0-9]+)_(?P<model>.+?)_(?P<mode>baseline|eag)"
    r"_(?P<ts>\d{8}_\d{6})_(?P<samples>\d+)\.json$"
)

CSV_FIELDS = [
    "dataset",
    "model",
    "mode",
    "run",
    "samples",
    "test_variant",
    "toolset",
    "metric",
    "accuracy",
    "mean_steps_per_task",
    "mean_tool_calls_per_task",
    "total_metadata_tokens",
    "total_data_tokens",
    "mean_data_tokens_per_task",
    "total_unresolved_placeholders",
    "errors",
    "truncated",
    "scoring_errors",
    "exposure_by_tool",
    "original_filename",
    "sha256",
    "freeze_timestamp",
    "cell_status",
]


class RunFile:
    def __init__(self, path: Path, m: re.Match):
        self.path = path
        self.dataset = m["dataset"]
        self.model = m["model"]
        self.mode = m["mode"]
        self.ts = m["ts"]
        self.samples = int(m["samples"])
        self.complete = False
        self.summary: dict | None = None
        self.test_variant: str | None = None
        self.toolset: str = ""

    @property
    def cell(self) -> str:
        return f"{self.dataset}/{self.model}"


def toolset_fingerprint(dataset: str, exposure_by_tool: dict | None) -> str:
    """Derive the toolset version of a run from its own tool accounting.

    hitab v2 (2026-09-15) adds ``compare_cells`` — address-level comparison
    over model-chosen cells; EAG data exposure stays zero. v1 = the 7-tool
    set without it. Derived from ``exposure_by_tool`` (not dates or config)
    so the marker backfills already-frozen entries and survives file moves.
    Datasets with a single toolset version fingerprint as "".
    """
    tools = exposure_by_tool or {}
    if dataset == "hitab":
        return "v2" if "compare_cells" in tools else "v1"
    return ""


def scan_results() -> list[RunFile]:
    runs: list[RunFile] = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        m = FILE_RE.match(path.name)
        if not m:
            continue
        run = RunFile(path, m)
        with open(path) as fh:
            data = json.load(fh)
        run.summary = data.get("summary")
        run.complete = run.summary is not None
        run.test_variant = (data.get("config") or {}).get("test_variant")
        run.toolset = toolset_fingerprint(
            run.dataset, (run.summary or {}).get("exposure_by_tool")
        )
        runs.append(run)
    return runs


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {"version": 1, "cells": {}}
    with open(MANIFEST_PATH) as fh:
        return json.load(fh)


def normalize_manifest(manifest: dict) -> dict:
    """Backfill cell status for manifests written before --partial existed."""
    for section in manifest.get("cells", {}).values():
        if "status" in section or "runs" not in section:
            continue
        counts = {m: sum(1 for e in section["runs"] if e["mode"] == m) for m in MODES}
        if all(c == EXPECTED_PER_MODE for c in counts.values()):
            section["status"] = "complete"
            section.setdefault("completed_timestamp", section["freeze_timestamp"])
        else:
            section["status"] = "partial"
    return manifest


def write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=2, sort_keys=False)
        fh.write("\n")
    os.replace(tmp, path)


def build_run_entry(run: RunFile, index: int, digest: str) -> dict:
    s = run.summary or {}
    return {
        "run": index,
        "mode": run.mode,
        "sha256": digest,
        "original_filename": run.path.name,
        "original_path": str(run.path.relative_to(RESULTS_DIR.parent)),
        "frozen_relative_path": str(
            Path(run.dataset) / run.model / run.mode / f"run-{index}.json"
        ),
        "samples": run.samples,
        "test_variant": run.test_variant,
        "toolset": run.toolset,
        "completed_at": run.ts,
        "metrics": {
            "metric": s.get("metric"),
            "accuracy": s.get("accuracy"),
            "avg_score": s.get("avg_score"),
            "total_tokens": s.get("total_tokens"),
            "total_metadata_tokens": s.get("total_metadata_tokens"),
            "total_data_tokens": s.get("total_data_tokens"),
            "mean_data_tokens_per_task": s.get("mean_data_tokens_per_task"),
            "mean_data_redacted_tokens_per_task": s.get(
                "mean_data_redacted_tokens_per_task"
            ),
            "mean_steps_per_task": s.get("mean_steps_per_task"),
            "mean_tool_calls_per_task": s.get("mean_tool_calls_per_task"),
            "total_unresolved_placeholders": s.get("total_unresolved_placeholders"),
            "errors": s.get("errors"),
            "scoring_errors": s.get("scoring_errors"),
            "truncated": s.get("truncated"),
            "exposure_by_tool": s.get("exposure_by_tool"),
        },
    }


def manifest_rows(manifest: dict) -> list[dict]:
    rows = []
    for cell in sorted(manifest["cells"]):
        section = manifest["cells"][cell]
        entries = sorted(section["runs"], key=lambda e: (e["mode"], e["run"]))
        for entry in entries:
            metrics = entry["metrics"]
            rows.append(
                {
                    "dataset": section["dataset"],
                    "model": section["model"],
                    "mode": entry["mode"],
                    "run": entry["run"],
                    "samples": entry["samples"],
                    "test_variant": entry["test_variant"],
                    "toolset": toolset_fingerprint(
                        section["dataset"], metrics.get("exposure_by_tool")
                    ),
                    "metric": metrics.get("metric"),
                    "accuracy": metrics.get("accuracy"),
                    "mean_steps_per_task": metrics.get("mean_steps_per_task"),
                    "mean_tool_calls_per_task": metrics.get("mean_tool_calls_per_task"),
                    "total_metadata_tokens": metrics.get("total_metadata_tokens"),
                    "total_data_tokens": metrics.get("total_data_tokens"),
                    "mean_data_tokens_per_task": metrics.get("mean_data_tokens_per_task"),
                    "total_unresolved_placeholders": metrics.get(
                        "total_unresolved_placeholders"
                    ),
                    "errors": metrics.get("errors"),
                    "truncated": metrics.get("truncated"),
                    "scoring_errors": metrics.get("scoring_errors"),
                    "exposure_by_tool": json.dumps(
                        metrics.get("exposure_by_tool"), separators=(",", ":")
                    ),
                    "original_filename": entry["original_filename"],
                    "sha256": entry["sha256"],
                    "freeze_timestamp": section["freeze_timestamp"],
                    "cell_status": section["status"],
                }
            )
    return rows


def regenerate_summary_csv(manifest: dict) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in manifest_rows(manifest):
        writer.writerow(row)
    return buf.getvalue()


def write_summary_csv(manifest: dict) -> None:
    FREEZE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = SUMMARY_CSV_PATH.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as fh:
        fh.write(regenerate_summary_csv(manifest))
    os.replace(tmp, SUMMARY_CSV_PATH)


def group_by_cell(runs: list[RunFile]) -> dict[str, dict[str, list[RunFile]]]:
    cells: dict[str, dict[str, list[RunFile]]] = {}
    for run in runs:
        cells.setdefault(run.cell, {}).setdefault(run.mode, []).append(run)
    return cells


def complete_counts(modes: dict[str, list[RunFile]]) -> dict[str, int]:
    return {m: sum(1 for r in modes.get(m, []) if r.complete) for m in MODES}


def move_and_verify(run: RunFile, dst: Path) -> str:
    digest = sha256(run.path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(run.path), dst)
    if sha256(dst) != digest:
        raise SystemExit(f"FATAL: sha256 mismatch after moving {run.path} -> {dst}")
    return digest


def plan_moves(
    runs_by_index: list[tuple[RunFile, int]], dataset: str, model: str, apply: bool
) -> list[tuple[RunFile, int, str]]:
    moved: list[tuple[RunFile, int, str]] = []
    for run, index in runs_by_index:
        dst = FREEZE_DIR / dataset / model / run.mode / f"run-{index}.json"
        if apply:
            digest = move_and_verify(run, dst)
            moved.append((run, index, digest))
            print(f"  {run.path.name} -> {dst.relative_to(RESULTS_DIR)}")
        else:
            print(f"  [dry-run] {run.path.name} -> {dst.relative_to(RESULTS_DIR)}")
    return moved


def cmd_status() -> int:
    manifest = normalize_manifest(load_manifest())
    cells = group_by_cell(scan_results())
    all_cells = sorted(set(cells) | set(manifest["cells"]))
    if not all_cells:
        print("No result files found in", RESULTS_DIR)
        return 0
    width = max(len(c) for c in all_cells)
    print(f"{'cell':<{width}}  {'baseline':>10}  {'eag':>10}  status")
    ready_full: list[str] = []
    ready_partial: list[str] = []
    for cell in all_cells:
        modes = cells.get(cell, {})
        counts = complete_counts(modes)
        partials = {m: sum(1 for r in modes.get(m, []) if not r.complete) for m in MODES}
        section = manifest["cells"].get(cell)
        if section:
            if section["status"] == "complete":
                status = "frozen (complete 5+5)"
            else:
                status = "frozen (partial 3+3)"
                needed = EXPECTED_PER_MODE - PARTIAL_PER_MODE
                if all(counts[m] >= needed for m in MODES):
                    status += " -- ready to complete (--partial --apply)"
        elif all(counts[m] == EXPECTED_PER_MODE for m in MODES):
            status = "READY to freeze (full 5+5)"
            ready_full.append(cell)
        elif all(counts[m] >= PARTIAL_PER_MODE for m in MODES):
            status = "READY to freeze (partial 3+3)"
            ready_partial.append(cell)
        else:
            status = ", ".join(
                f"{m}: {counts[m]}/{EXPECTED_PER_MODE}"
                + (f" (+{partials[m]} partial)" if partials[m] else "")
                for m in MODES
            )
        toolsets = sorted(
            {
                r.toolset
                for r in modes.get("baseline", []) + modes.get("eag", [])
                if r.complete and r.toolset
            }
        )
        if len(toolsets) > 1:
            status += f"  [TOOLSET MIXED {'/'.join(toolsets)} — do not freeze as one cell]"
        print(
            f"{cell:<{width}}  {counts['baseline']:>7}/{EXPECTED_PER_MODE}"
            f"  {counts['eag']:>7}/{EXPECTED_PER_MODE}  {status}"
        )
    print()
    if ready_full:
        print("Ready for full freeze (5+5):", ", ".join(ready_full))
    if ready_partial:
        print("Ready for partial freeze (3+3):", ", ".join(ready_partial))
    if not ready_full and not ready_partial:
        print("No new cells ready to freeze.")
    return 0


def freeze_cell(dataset: str, model: str, apply: bool, partial: bool) -> int:
    cell = f"{dataset}/{model}"
    modes = group_by_cell(scan_results()).get(cell, {})
    manifest = normalize_manifest(load_manifest())
    section = manifest["cells"].get(cell)
    complete_runs = {
        m: sorted((r for r in modes.get(m, []) if r.complete), key=lambda r: r.ts)
        for m in MODES
    }

    if section and section["status"] == "complete":
        print(f"{cell}: already frozen (complete) at {section['freeze_timestamp']}.")
        leftovers = complete_counts(modes)
        if any(leftovers.values()):
            print(
                f"  note: {leftovers['baseline']} baseline / {leftovers['eag']} eag "
                "complete runs still in results/ (beyond the frozen 5+5); not frozen."
            )
        return 0

    if section:  # frozen-partial (exactly 3+3 by construction)
        if not partial:
            print(f"ABORT: {cell} is frozen-partial; completion runs through --partial.")
            return 1
        return complete_cell(cell, section, complete_runs, manifest, apply)

    if partial:
        for mode in MODES:
            if len(complete_runs[mode]) < PARTIAL_PER_MODE:
                print(
                    f"ABORT: partial freeze of {cell} needs at least "
                    f"{PARTIAL_PER_MODE}+{PARTIAL_PER_MODE} complete runs; "
                    f"{mode} has {len(complete_runs[mode])}."
                )
                return 1
        selected = {m: complete_runs[m][:PARTIAL_PER_MODE] for m in MODES}
        status = "partial"
    else:
        for mode in MODES:
            n = len(complete_runs[mode])
            if n != EXPECTED_PER_MODE:
                hint = (
                    " Consider --partial (first 3+3) in the meantime."
                    if n >= PARTIAL_PER_MODE
                    else ""
                )
                print(
                    f"ABORT: full freeze of {cell} needs exactly "
                    f"{EXPECTED_PER_MODE}+{EXPECTED_PER_MODE} complete runs; "
                    f"{mode} has {n}.{hint}"
                )
                return 1
        selected = complete_runs
        status = "complete"

    plan = [
        (selected[mode][i], mode, i + 1)
        for mode in MODES
        for i in range(len(selected[mode]))
    ]
    verb = "partial-freeze (first 3+3)" if partial else "freeze (5+5)"
    print(f"Cell {cell}: {len(plan)} runs to {verb}, resulting status: {status}.")
    moved = plan_moves([(run, index) for run, mode, index in plan], dataset, model, apply)
    if partial:
        for mode in MODES:
            extras = len(complete_runs[mode]) - PARTIAL_PER_MODE
            if extras > 0:
                print(
                    f"  note: {extras} extra complete {mode} run(s) left in results/ "
                    "for the completion step."
                )
    if not apply:
        print("Dry-run only; pass --apply to execute.")
        return 0

    now = datetime.now().astimezone().isoformat(timespec="seconds")
    new_section = {
        "dataset": dataset,
        "model": model,
        "status": status,
        "freeze_timestamp": now,
        "git_sha": git_sha(),
        "runs": [
            build_run_entry(run, index, digest) for run, index, digest in moved
        ],
    }
    if status == "complete":
        new_section["completed_timestamp"] = now
    manifest["cells"][cell] = new_section
    write_json_atomic(MANIFEST_PATH, manifest)
    write_summary_csv(manifest)
    print(f"Frozen {cell} ({status}): manifest + summary.csv updated.")
    print("Next: review `git status` and commit (left to the operator).")
    return 0


def unfreeze_cell(dataset: str, model: str, apply: bool) -> int:
    """Revert a frozen-partial cell: move runs back to results/, drop the manifest cell."""
    cell = f"{dataset}/{model}"
    manifest = normalize_manifest(load_manifest())
    section = manifest["cells"].get(cell)
    if not section:
        print(f"ABORT: {cell} is not frozen (no manifest section).")
        return 1
    if section["status"] == "complete":
        print(f"ABORT: {cell} is frozen-complete; only frozen-partial cells can be unfrozen.")
        return 1

    plan: list[tuple[Path, Path]] = []
    failures: list[str] = []
    for entry in section["runs"]:
        src = FREEZE_DIR / entry["frozen_relative_path"]
        dst = RESULTS_DIR / entry["original_filename"]
        if not src.exists():
            failures.append(f"missing frozen file {src}")
        elif sha256(src) != entry["sha256"]:
            failures.append(f"sha256 mismatch: {src}")
        elif dst.exists():
            failures.append(f"destination already exists: {dst}")
        else:
            plan.append((src, dst))
    if failures:
        print(f"ABORT: cannot unfreeze {cell}:")
        for failure in failures:
            print(" -", failure)
        return 1
    if not plan:
        print(f"{cell}: nothing to unfreeze.")
        return 0

    print(f"Cell {cell}: unfreezing {len(plan)} run(s) back to results/.")
    for src, dst in plan:
        if apply:
            digest = sha256(src)
            shutil.move(str(src), dst)
            if sha256(dst) != digest:
                raise SystemExit(f"FATAL: sha256 mismatch after moving {src} -> {dst}")
            print(f"  {src.relative_to(RESULTS_DIR)} -> {dst.name}")
        else:
            print(f"  [dry-run] {src.relative_to(RESULTS_DIR)} -> {dst.name}")
    if not apply:
        print("Dry-run only; pass --apply to execute.")
        return 0

    for src, _ in plan:
        for parent in (src.parent, src.parent.parent):
            if parent == FREEZE_DIR:
                break
            try:
                parent.rmdir()
            except OSError:
                break

    del manifest["cells"][cell]
    write_json_atomic(MANIFEST_PATH, manifest)
    write_summary_csv(manifest)
    print(f"Unfroze {cell}: manifest + summary.csv updated; runs are back in results/.")
    print("Next: review `git status` and commit (left to the operator).")
    return 0


def complete_cell(
    cell: str,
    section: dict,
    complete_runs: dict[str, list[RunFile]],
    manifest: dict,
    apply: bool,
) -> int:
    dataset, model = section["dataset"], section["model"]
    needed = EXPECTED_PER_MODE - PARTIAL_PER_MODE
    if all(len(complete_runs[m]) == 0 for m in MODES):
        print(f"{cell}: no new complete runs in results/; nothing to complete.")
        return 0
    plan: list[tuple[RunFile, int]] = []
    for mode in MODES:
        new_runs = complete_runs[mode]
        if len(new_runs) < needed:
            print(
                f"ABORT: completing {cell} needs exactly {needed}+{needed} new "
                f"complete runs in results/; {mode} has {len(new_runs)}/{needed}."
            )
            return 1
        start = PARTIAL_PER_MODE
        plan.extend((run, start + 1 + offset) for offset, run in enumerate(new_runs[:needed]))
        if len(new_runs) > needed:
            print(
                f"  note: {len(new_runs) - needed} extra complete {mode} run(s) in "
                "results/ will stay unfrozen (only the first "
                f"{needed} by timestamp are used)."
            )
    print(f"Cell {cell}: completing partial freeze with {len(plan)} runs -> 5+5.")
    moved = plan_moves(plan, dataset, model, apply)
    if not apply:
        print("Dry-run only; pass --apply to execute.")
        return 0

    now = datetime.now().astimezone().isoformat(timespec="seconds")
    section["runs"].extend(
        build_run_entry(run, index, digest) for run, index, digest in moved
    )
    section["status"] = "complete"
    section["completed_timestamp"] = now
    section["completed_git_sha"] = git_sha()
    write_json_atomic(MANIFEST_PATH, manifest)
    write_summary_csv(manifest)
    print(f"Completed {cell} to 5+5: manifest + summary.csv updated.")
    print("Next: review `git status` and commit (left to the operator).")
    return 0


def cmd_check() -> int:
    manifest = normalize_manifest(load_manifest())
    if not manifest["cells"]:
        print("Nothing frozen yet.")
        return 0
    failures: list[str] = []
    n_files = 0
    for cell in sorted(manifest["cells"]):
        section = manifest["cells"][cell]
        expected = EXPECTED_PER_MODE if section["status"] == "complete" else PARTIAL_PER_MODE
        for mode in MODES:
            entries = sorted(
                (e for e in section["runs"] if e["mode"] == mode), key=lambda e: e["run"]
            )
            if len(entries) != expected:
                failures.append(
                    f"{cell}/{mode}: {len(entries)} runs in manifest (expected {expected})"
                )
            if [e["run"] for e in entries] != list(range(1, len(entries) + 1)):
                failures.append(f"{cell}/{mode}: non-contiguous run numbering")
            for entry in entries:
                dst = FREEZE_DIR / entry["frozen_relative_path"]
                if not dst.exists():
                    failures.append(f"{cell}/{mode}/run-{entry['run']}: missing {dst.name}")
                    continue
                n_files += 1
                if sha256(dst) != entry["sha256"]:
                    failures.append(f"{cell}/{mode}/run-{entry['run']}: sha256 mismatch")
    expected_csv = regenerate_summary_csv(manifest)
    if SUMMARY_CSV_PATH.exists():
        if SUMMARY_CSV_PATH.read_text() != expected_csv:
            failures.append("summary.csv out of sync with manifest")
    else:
        failures.append("summary.csv missing")
    if failures:
        print("CHECK FAILED:")
        for failure in failures:
            print(" -", failure)
        return 1
    print(f"OK: {n_files} frozen files verified across {len(manifest['cells'])} cell(s).")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--status", action="store_true", help="show per-cell completeness and readiness"
    )
    parser.add_argument("--dataset", help="dataset of the cell to freeze (with --model)")
    parser.add_argument("--model", help="model config name of the cell to freeze (with --dataset)")
    parser.add_argument(
        "--partial",
        action="store_true",
        help="partial freeze: first 3+3 runs of an unfrozen cell, or complete a "
        "frozen-partial cell to 5+5",
    )
    parser.add_argument("--apply", action="store_true", help="execute the move (default: dry-run)")
    parser.add_argument(
        "--unfreeze",
        action="store_true",
        help="revert a frozen-partial cell: move its runs back to results/ under "
        "their original filenames and drop it from the manifest (sha256-verified)",
    )
    parser.add_argument(
        "--check", action="store_true", help="verify manifest, files, and summary.csv"
    )
    args = parser.parse_args()

    if args.check:
        return cmd_check()
    if args.status or (not args.dataset and not args.model):
        return cmd_status()
    if bool(args.dataset) != bool(args.model):
        parser.error("--dataset and --model must be given together")
    if args.unfreeze:
        if args.partial:
            parser.error("--unfreeze cannot be combined with --partial")
        return unfreeze_cell(args.dataset, args.model, args.apply)
    return freeze_cell(args.dataset, args.model, args.apply, args.partial)


if __name__ == "__main__":
    sys.exit(main())
