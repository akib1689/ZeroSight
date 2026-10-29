"""CLI orchestration for the disclosure-metrics pass."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from .analyze import analyze_run, rollup
from .common import ROOT
from .contexts import (
    HitabData,
    LindisfarneData,
    build_bird_context,
    build_spider_context,
)
from .report import write_split
from .validate import validate


def main() -> None:
    ap = argparse.ArgumentParser(description="Derive disclosure metrics (read-only)")
    ap.add_argument(
        "--runs-root", type=Path, default=ROOT / "results" / "paper-experiments"
    )
    ap.add_argument(
        "--out-dir", type=Path, default=None,
        help="output directory (default: <runs-root>/derived)",
    )
    ap.add_argument("--only", default=None, help="substring filter on cell names")
    ap.add_argument(
        "--validate",
        action="store_true",
        help="run known-example spot checks and record them in the output",
    )
    args = ap.parse_args()

    runs_root = args.runs_root
    out_dir = args.out_dir or (runs_root / "derived")
    manifest = json.loads((runs_root / "MANIFEST.json").read_text())
    cells = {
        k: v for k, v in manifest["cells"].items() if v.get("status") == "complete"
    }
    if args.only:
        cells = {k: v for k, v in cells.items() if args.only in k}

    by_dataset: dict[str, list[str]] = defaultdict(list)
    for cell in cells:
        ds = cell.split("/")[0]
        by_dataset[ds].append(cell)

    contexts: dict[str, object] = {
        "lindisfarne": LindisfarneData(),
        "hitab": HitabData(),
        "bird": build_bird_context(),
        "spider": build_spider_context(),
    }
    for ds, ds_cells in sorted(by_dataset.items()):
        ctx = contexts[ds]
        cell_dirs = sorted({f"{ds}/{c.split('/')[1]}" for c in ds_cells})
        t0 = time.time()
        ctx.prepare(runs_root, cell_dirs)
        print(f"[prepare] {ds}: {time.time() - t0:.1f}s", flush=True)

    result: dict = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "definitions": {
            "verification": {
                "lindisfarne": "full corpus values for every touched record",
                "hitab": "full table cells, structured observation parse",
                "bird/spider": (
                    "full per-database distinct-value inventory + question exclusion"
                ),
            },
        },
        "cells": {},
    }

    for cell in sorted(cells):
        ds = cell.split("/")[0]
        ctx = contexts[ds]
        run_entries = []
        for run_info in sorted(cells[cell]["runs"], key=lambda r: r["run"]):
            path = runs_root / run_info["frozen_relative_path"]
            payload = analyze_run(path, ctx)
            run_entries.append({"run": run_info["run"], **payload})
            mt = payload["totals"]
            print(
                f"[{cell}] run-{run_info['run']} {payload['mode']}: "
                f"val_tok={mt['value_tokens']:.0f} dr={payload['dr']:.3f} "
                f"vvf={mt['vvf_violations']:.0f} rd={mt['records_disclosed']:.0f} "
                f"hd={mt['handle_count']:.0f} cd={mt['cardinality_hits']:.0f} "
                f"shape={mt['shape_tokens']:.0f}",
                flush=True,
            )
        by_mode: dict[str, list[dict]] = defaultdict(list)
        for entry in run_entries:
            by_mode[entry["mode"]].append(entry)
        result["cells"][cell] = {
            "dataset": ds,
            "runs": run_entries,
            "rollup": {m: rollup(runs) for m, runs in sorted(by_mode.items())},
        }

    validation = validate(runs_root, result) if args.validate else None
    if validation is not None:
        for check in validation["checks"]:
            status = "PASS" if check["ok"] else "FAIL"
            print(f"[validate] {status}: {check['name']}: {check['detail']}")

    written = write_split(result, validation, out_dir)
    for path in written:
        print(f"wrote {path}")
