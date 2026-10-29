"""Split-output writer: one reviewable JSON file per headline metric.

Emits under ``<runs-root>/derived/``:

  common.json   definitions, verification coverage, run inventory, validation,
                content-class ledger (handles / cardinality / shape) per run
                and per task
  dr.json       Disclosure Rate + per-task {id, value_observations,
                data_observations}
  vvf.json      Verified Value-Freeness (EAG violations, leaked tokens) +
                model-emitted arg-collision channel + per-task evidence
  rd.json       Records Disclosed (lindisfarne cells) + per-task counts
  maxleak.json  worst-case per-task value tokens + per-task value_tokens

Each metric file is self-contained: definition, coverage, per-run values,
rollup mean+/-std, and its own task-level evidence. Per-task fields are
partitioned so no field appears twice in review-critical detail; small
run-level duplication (samples, mode) makes every file navigable alone.
"""

from __future__ import annotations

import json
from pathlib import Path

from .analyze import RUN_METRIC_KEYS

_LEDGER_FIELDS = [
    "handle_count", "handle_tokens",
    "cardinality_hits", "cardinality_tokens", "shape_tokens",
]
_DR_FIELDS = ["value_observations", "data_observations"]
_VVF_FIELDS = ["vvf_violations", "value_tokens", "arg_value_violations"]
_RD_FIELDS = ["records_disclosed"]
_MAXLEAK_FIELDS = ["value_tokens"]

_DEFINITIONS = {
    "values": "record-store literals absent from the task question (disclosure)",
    "handles": "record IDs / cell addresses -- deliberate disclosure policy",
    "cardinality": "match/row/cell counts -- deliberate functional feedback",
    "shape": "structure scaffolding -- cost, never disclosure",
}

_COVERAGE = {
    "dr": "share of data-tool observations containing >=1 record-store value "
          "absent from the task question",
    "vvf": "EAG data observations containing >=1 such value (expect 0); "
           "arg_value_violations is the separate model-emitted-args channel",
    "rd": "distinct records with >=1 disclosed identifying value per run "
          "(lindisfarne only)",
    "maxleak": "worst-case per-task disclosed value tokens",
}

# Per metric file: per-task fields, rollup keys, run-level values as
# (name, source) pairs where source is "top" (analyze_run scalar) or "totals".
_METRIC_SPECS = {
    "dr": {
        "per_task": _DR_FIELDS,
        "rollup": ["dr"],
        "run_values": [("dr", "top"),
                       ("value_observations", "totals"),
                       ("data_observations", "totals")],
    },
    "vvf": {
        "per_task": _VVF_FIELDS,
        "rollup": ["vvf_violations", "value_tokens", "arg_value_violations"],
        "run_values": [("vvf_violations", "totals"),
                       ("value_tokens", "totals"),
                       ("arg_value_violations", "totals")],
    },
    "rd": {
        "per_task": _RD_FIELDS,
        "rollup": ["records_disclosed", "records_disclosed_tasks"],
        "run_values": [("records_disclosed", "totals"),
                       ("records_disclosed_tasks", "top")],
    },
    "maxleak": {
        "per_task": _MAXLEAK_FIELDS,
        "rollup": ["maxleak", "mean_value_tokens_per_task"],
        "run_values": [("maxleak", "top"),
                       ("mean_value_tokens_per_task", "top")],
    },
}


def _slice(per_task: list[dict], fields: list[str]) -> list[dict]:
    return [{"id": t["id"], **{f: t[f] for f in fields}} for t in per_task]


def _run_entry(entry: dict, spec: dict) -> dict:
    out = {"run": entry["run"], "mode": entry["mode"], "samples": entry["samples"]}
    for name, source in spec["run_values"]:
        out[name] = entry["totals"][name] if source == "totals" else entry[name]
    out["per_task"] = _slice(entry["per_task"], spec["per_task"])
    return out


def _mode_blocks(cells: dict) -> dict:
    out: dict[str, dict] = {}
    for cell, cell_data in cells.items():
        by_mode: dict[str, list[dict]] = {}
        for entry in cell_data["runs"]:
            by_mode.setdefault(entry["mode"], []).append(entry)
        for name, spec in _METRIC_SPECS.items():
            mode_blocks = {}
            for mode, entries in sorted(by_mode.items()):
                roll = cell_data["rollup"][mode]
                mode_blocks[mode] = {
                    "runs": [_run_entry(e, spec) for e in entries],
                    "rollup": {k: roll[k] for k in spec["rollup"]},
                }
            out.setdefault(cell, {"dataset": cell_data["dataset"], "metrics": {}})
            out[cell]["metrics"][name] = mode_blocks
    # reshape: cells -> cell -> metrics -> metric -> modes
    return out


def write_split(result: dict, validation: dict | None, out_dir: Path) -> list[Path]:
    cells = result["cells"]
    generated = result["generated_at"]
    split = _mode_blocks(cells)

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, spec in _METRIC_SPECS.items():
        payload = {
            "metric": name,
            "generated_at": generated,
            "definition": _COVERAGE[name],
            "cells": {
                cell: {"dataset": data["dataset"], "modes": data["metrics"][name]}
                for cell, data in split.items()
            },
        }
        path = out_dir / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2))
        written.append(path)

    common_cells = {}
    for cell, cell_data in cells.items():
        by_mode: dict[str, list[dict]] = {}
        for entry in cell_data["runs"]:
            by_mode.setdefault(entry["mode"], []).append(entry)
        common_cells[cell] = {
            "dataset": cell_data["dataset"],
            "modes": {
                mode: {
                    "runs": [
                        {"run": e["run"], "samples": e["samples"],
                         "totals": {k: e["totals"][k] for k in RUN_METRIC_KEYS},
                         "stored_summary_reference": e["stored_summary_reference"],
                         "per_task": _slice(e["per_task"], _LEDGER_FIELDS)}
                        for e in entries
                    ],
                }
                for mode, entries in sorted(by_mode.items())
            },
        }
    common = {
        "generated_at": generated,
        "definitions": {
            **_DEFINITIONS,
            "verification": result["definitions"]["verification"],
        },
        "cells": common_cells,
    }
    if validation is not None:
        common["validation"] = validation
    path = out_dir / "common.json"
    path.write_text(json.dumps(common, indent=2))
    written.append(path)
    return written
