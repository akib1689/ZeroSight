"""Hierarchical-table tools for the HiTab benchmark (QA over header trees).

Data model (HiTab ``hmt`` preprocessed tables, ``data/hitab/tables/hmt/``):

  - ``top_root``  — column-header tree; ``left_root`` — row-header tree.
  - ``data``      — pure data grid ``data[row][col] = {"value": ...}``
    (no headers interleaved).
  - Indexing: EVERY tree node (inner or leaf) whose ``line_idx`` is not
    None labels a data row (left tree) or data column (top tree). A node's
    own row often holds the aggregate/total while its children hold the
    breakdown ("all industries" [9] vs its children "manufacturing" [10],
    "nonmanufacturing" [11]). Some inner indices label unit rows
    ('percent', blank) that precede a section.

A data cell is addressed ``(row, col)``: the row a ``left_root`` node
labels crossed with the column a ``top_root`` node labels. Navigating the
hierarchy — matching question phrases to header paths — is the benchmark
skill the tool set forces.

Tool tiers (``exposure_type``):

  Metadata (seen in both baseline and EAG):
    get_table_overview, find_header   — header text is schema
  Data (baseline: raw values; EAG: addresses/types only):
    get_cells, find_cells             — the values are the private payload
    compare_cells                     — predicate over model-chosen
                                        addresses; discloses which
                                        addresses match / hold the extreme
                                        (address-level info, same class as
                                        find_cells match addresses)
  Local compute (shape-only in EAG):
    aggregate (EvaluatorTool subclass; ``compute`` is reused from
    tools/evaluator.py)
  Evaluation:
    submit_answer

Leak rules for EAG redaction: observations disclose addresses (the model
chose them), types, and counts — never values.
"""

import json
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from tools.evaluator import ComputeTool, EvaluatorTool
from tools.sql import EAGBaseTool, ToolResult

_MAX_CELLS = 32
_MAX_FIND_RESULTS = 50


# ── Table loading / helpers ───────────────────────────────────────────────────


def _load_table(table_path: str | Path) -> dict:
    """Load an ``hmt``-variant HiTab table and normalize its data grid."""
    with open(table_path) as f:
        raw = json.load(f)
    grid = [[cell.get("value") for cell in row] for row in raw.get("data", [])]
    return {
        "title": raw.get("title", ""),
        "top_root": raw.get("top_root", {}),
        "left_root": raw.get("left_root", {}),
        "grid": grid,
        "n_rows": len(grid),
        "n_cols": len(grid[0]) if grid else 0,
    }


def _children(node: dict) -> list[dict]:
    return node.get("children_dict") or []


def _subtree_indices(node: dict) -> list[int]:
    """Sorted ``line_idx`` values of every labeled node strictly beneath ``node``.

    Includes intermediate nodes — they own real rows (aggregates) just like
    leaves own detail rows.
    """
    out: list[int] = []

    def walk(n: dict) -> None:
        for c in _children(n):
            idx = c.get("line_idx")
            if idx is not None:
                out.append(idx)
            walk(c)

    walk(node)
    return sorted(set(out))


def _walk(root: dict, path: tuple[str, ...] = ()):
    """Yield ``(path_tuple, node)`` for every node in a header tree."""
    yield path, root
    for child in _children(root):
        yield from _walk(child, path + (child.get("name", ""),))


def _render_tree(root: dict, what: str) -> str:
    """Indented rendering of a header tree with per-node index annotations."""
    lines = [f"{what}:"]
    root_name = root.get("name", "<ROOT>")
    lines.append(f"  {root_name}")
    for path, node in _walk(root):
        if not path:
            continue
        depth = len(path)
        idx = node.get("line_idx")
        idx_tag = f"  -> index {idx}" if idx is not None else ""
        lines.append(f"{'  ' * (depth + 1)}{node.get('name', '')}{idx_tag}")
    return "\n".join(lines)


def _value_type(v: Any) -> str:
    if v is None or (isinstance(v, str) and not v.strip()):
        return "empty"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    return "str"


def _as_number(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip().replace(",", "").replace("$", "").rstrip("%")
        try:
            return float(s)
        except ValueError:
            return None
    return None


# ── Input schemas ─────────────────────────────────────────────────────────────


class _FindHeaderInput(BaseModel):
    query: str = Field(..., description="Phrase to find in header labels")
    axis: str = Field(
        "both",
        description='Which header tree to search: "top" (column headers), '
        '"left" (row headers), or "both" (default)',
    )


class _GetCellsInput(BaseModel):
    cells: list[dict] = Field(
        ...,
        min_length=1,
        description='Cells to read, e.g. [{"row": 3, "col": 0}] — row/col are '
        "data-grid indices as shown by get_table_overview / find_header",
    )


class _FindCellsInput(BaseModel):
    col: int = Field(..., description="Data column index to scan")
    op: str = Field(
        ...,
        description='Comparison: "==", "!=", "<", "<=", ">", ">=", "contains"',
    )
    value: str | float | int = Field(
        ..., description="Value to compare against (numbers compared numerically)"
    )
    limit: int = Field(10, description="Max matches to return (default 10)")


class _CompareCellsInput(BaseModel):
    cells: list[dict] = Field(
        ...,
        min_length=2,
        description='Cell addresses to compare, e.g. [{"row": 3, "col": 1}, '
        '{"row": 3, "col": 2}] — resolved via find_header / get_table_overview',
    )
    op: str = Field(
        ...,
        description='"max"/"min" (no value — locate the extreme cell) or a '
        'comparison "==", "!=", "<", "<=", ">", ">=" against value',
    )
    value: str | float | int | None = Field(
        None,
        description="Required for comparison ops — a number or {{ placeholder }} "
        'reference (e.g. "{{ aggregate.result }}")',
    )


class _AggregateInput(BaseModel):
    op: str = Field(..., description="Aggregation: 'sum', 'avg', 'min', or 'max'")
    items: list[str | float | int] = Field(
        ...,
        min_length=2,
        description="Numbers to aggregate — literals and/or {{ placeholder }} "
        "references (e.g. [\"{{ get_cells.cells[0].value }}\", 52.1])",
    )


class _SubmitAnswerInput(BaseModel):
    # Permissive on purpose: EAG middleware resolves {{ }} placeholders to
    # native types (float/int) before execution; a str-only schema would
    # fail pydantic validation and echo the resolved value back to the
    # model through the error message (the exact leak EAG must prevent).
    answer: str | float | int | bool = Field(
        ...,
        description="Final answer — a literal in baseline mode, or a "
        "{{ placeholder }} reference (e.g. \"{{ aggregate.result }}\") in EAG mode",
    )


# ── Metadata tools ────────────────────────────────────────────────────────────


class TableOverviewTool(EAGBaseTool):
    """Table structure: title + both header trees + data-grid dimensions."""

    name: str = "get_table_overview"
    description: str = (
        "Get the table's structure: title, the column-header tree (top), the "
        "row-header tree (left), and the data-grid dimensions. Every node "
        "annotated '-> index N' labels a data column (top tree) or data row "
        "(left tree); a node's own row may hold the total while its children "
        "hold the breakdown. Use find_header to map question phrases onto "
        "those indices."
    )
    exposure_type: str = "metadata"
    table_path: str = ""

    def execute(self, **kwargs) -> ToolResult:
        table = _load_table(self.table_path)
        out = (
            f"Title: {table['title']}\n"
            f"Data grid: {table['n_rows']} rows x {table['n_cols']} columns "
            "(row 0..n_rows-1, col 0..n_cols-1; headers excluded)\n\n"
            f"{_render_tree(table['top_root'], 'Column headers (top tree)')}\n\n"
            f"{_render_tree(table['left_root'], 'Row headers (left tree)')}"
        )
        return ToolResult(
            output=out,
            row_count=table["n_rows"],
            data={
                "title": table["title"],
                "n_rows": table["n_rows"],
                "n_cols": table["n_cols"],
                "top_root": table["top_root"],
                "left_root": table["left_root"],
            },
        )


class FindHeaderTool(EAGBaseTool):
    """Find header nodes whose label contains a query phrase."""

    name: str = "find_header"
    description: str = (
        'Find header labels matching a phrase, e.g. find_header(query="female", '
        'axis="top"). Returns each match with its path and the data indices '
        "(column indices for top-tree matches, row indices for left-tree "
        "matches): the node's own index when it labels a row/column, plus "
        "every leaf index beneath it (breakdown rows)."
    )
    args_schema: type[BaseModel] = _FindHeaderInput
    exposure_type: str = "metadata"
    table_path: str = ""

    def execute(self, **kwargs) -> ToolResult:
        query = str(kwargs.get("query", "")).strip().lower()
        axis = str(kwargs.get("axis", "both")).strip().lower()
        if not query:
            return ToolResult(output="Error: empty query", error="empty query")
        if axis not in ("top", "left", "both"):
            return ToolResult(output=f"Error: invalid axis '{axis}'", error="bad axis")

        table = _load_table(self.table_path)
        lines: list[str] = []
        matches: list[dict] = []
        axes = []
        if axis in ("top", "both"):
            axes.append(("top", table["top_root"], "column"))
        if axis in ("left", "both"):
            axes.append(("left", table["left_root"], "row"))

        for ax_name, root, noun in axes:
            for path, node in _walk(root):
                if not path:
                    continue
                if query in str(node.get("name", "")).lower():
                    own = node.get("line_idx")
                    sub = _subtree_indices(node)
                    idxs = sorted({i for i in ([own] if own is not None else []) + sub})
                    if not idxs:
                        continue
                    if not _children(node):
                        note = "leaf"
                    elif own is not None:
                        note = f"own {noun} {own}" + (
                            f", rows beneath: {sub}" if sub else ""
                        )
                    else:
                        note = f"rows beneath: {sub}"
                    lines.append(
                        f'{ax_name}: "{" > ".join(p for p in path if p)}" '
                        f"-> {noun} indices {idxs} ({note})"
                    )
                    matches.append(
                        {"axis": ax_name, "path": list(path), "indices": idxs}
                    )

        if not lines:
            out = f'No header matches "{kwargs.get("query")}" ({axis}).'
        else:
            out = "\n".join(lines)
        return ToolResult(
            output=out,
            row_count=len(matches),
            data={"matches": matches},
        )


# ── Data tools ────────────────────────────────────────────────────────────────


class GetCellsTool(EAGBaseTool):
    """Read specific data cells by (row, col) address."""

    name: str = "get_cells"
    description: str = (
        "Read data-cell values by address, e.g. "
        'get_cells(cells=[{"row": 3, "col": 0}]). Resolve row/col indices '
        "first via find_header / get_table_overview."
    )
    args_schema: type[BaseModel] = _GetCellsInput
    exposure_type: str = "data"
    table_path: str = ""

    def validate(self, **kwargs) -> tuple[bool, str]:
        cells = kwargs.get("cells", [])
        if not cells or len(cells) > _MAX_CELLS:
            return False, f"Provide 1..{_MAX_CELLS} cells"
        for c in cells:
            if not isinstance(c, dict) or "row" not in c or "col" not in c:
                return False, "Each cell needs 'row' and 'col' keys"
        return True, ""

    def execute(self, **kwargs) -> ToolResult:
        table = _load_table(self.table_path)
        cells = kwargs.get("cells", [])
        out_lines: list[str] = []
        data_cells: list[dict] = []
        for c in cells:
            r, col = int(c["row"]), int(c["col"])
            if not (0 <= r < table["n_rows"] and 0 <= col < table["n_cols"]):
                return ToolResult(
                    output=(
                        f"Error: address (row={r}, col={col}) outside data grid "
                        f"({table['n_rows']} rows x {table['n_cols']} cols)"
                    ),
                    error="out of bounds",
                )
            v = table["grid"][r][col]
            out_lines.append(f"(row={r}, col={col}): {v!r}")
            data_cells.append(
                {"row": r, "col": col, "value": v, "type": _value_type(v)}
            )
        return ToolResult(
            output="\n".join(out_lines),
            row_count=len(data_cells),
            data={"cells": data_cells},
        )

    def generate_metadata(self, result: ToolResult) -> str:
        if result.error:
            return f"[EAG] get_cells failed — {result.output}"
        cells = result.data["cells"]
        lines = [
            f"(row={c['row']}, col={c['col']}): <{c['type']}>" for c in cells
        ]
        return (
            f"{len(cells)} cell(s), values withheld:\n" + "\n".join(lines)
        )


class FindCellsTool(EAGBaseTool):
    """Scan a data column for cells matching a condition."""

    name: str = "find_cells"
    description: str = (
        "Scan one data column for cells matching a condition, e.g. "
        'find_cells(col=0, op=">", value=50). Returns matching rows. Useful '
        "for argmax/argmin-style questions (scan with > threshold, count "
        "matches, bisect)."
    )
    args_schema: type[BaseModel] = _FindCellsInput
    exposure_type: str = "data"
    table_path: str = ""

    def validate(self, **kwargs) -> tuple[bool, str]:
        if kwargs.get("op") not in ("==", "!=", "<", "<=", ">", ">=", "contains"):
            return False, f"Invalid op: {kwargs.get('op')}"
        return True, ""

    def execute(self, **kwargs) -> ToolResult:
        table = _load_table(self.table_path)
        col = int(kwargs.get("col", -1))
        op = kwargs.get("op")
        value = kwargs.get("value")
        limit = max(1, min(int(kwargs.get("limit", 10)), _MAX_FIND_RESULTS))

        if not (0 <= col < table["n_cols"]):
            return ToolResult(
                output=f"Error: col={col} outside grid ({table['n_cols']} columns)",
                error="out of bounds",
            )

        val_num = _as_number(value)
        out_lines: list[str] = []
        data_cells: list[dict] = []
        total = 0
        for r in range(table["n_rows"]):
            cell = table["grid"][r][col]
            c_num = _as_number(cell)
            if op == "contains":
                ok = str(value).lower() in str(cell).lower()
            elif op == "==":
                ok = (
                    c_num is not None
                    and val_num is not None
                    and c_num == val_num
                ) or str(cell).strip().lower() == str(value).strip().lower()
            elif op == "!=":
                ok = not (
                    (
                        c_num is not None
                        and val_num is not None
                        and c_num == val_num
                    )
                    or str(cell).strip().lower() == str(value).strip().lower()
                )
            else:
                if c_num is None or val_num is None:
                    continue
                ok = {
                    "<": c_num < val_num,
                    "<=": c_num <= val_num,
                    ">": c_num > val_num,
                    ">=": c_num >= val_num,
                }[op]
            if not ok:
                continue
            total += 1
            if len(data_cells) < limit:
                out_lines.append(f"(row={r}, col={col}): {cell!r}")
                data_cells.append(
                    {
                        "row": r,
                        "col": col,
                        "value": cell,
                        "type": _value_type(cell),
                    }
                )

        header = (
            f"{total} match(es) in column {col}"
            f"{' (showing first ' + str(len(data_cells)) + ')' if total > len(data_cells) else ''}:"
        )
        return ToolResult(
            output=(header + "\n" + "\n".join(out_lines)) if out_lines else header,
            row_count=total,
            data={"cells": data_cells, "match_count": total},
        )

    def generate_metadata(self, result: ToolResult) -> str:
        if result.error:
            return f"[EAG] find_cells failed — {result.output}"
        cells = result.data["cells"]
        total = result.data.get("match_count", len(cells))
        lines = [
            f"(row={c['row']}, col={c['col']}): <{c['type']}>" for c in cells
        ]
        out = f"{total} match(es); addresses kept, values withheld"
        if lines:
            out += ":\n" + "\n".join(lines)
        return out


class CompareCellsTool(EAGBaseTool):
    """Predicate over model-chosen cell addresses; returns passing addresses.

    The observable (both modes) is address-level only — which of the given
    coordinates hold the extreme or satisfy the condition — never the
    values. This gives the model a branch-free comparison channel: the
    harness executes the comparison, the model reads the winning address
    and maps it back to a header label via the metadata tools.
    """

    name: str = "compare_cells"
    description: str = (
        "Compare specific cells by address, e.g. compare_cells(cells=["
        '{"row": 3, "col": 1}, {"row": 3, "col": 2}], op="max") — returns '
        "the coordinate(s) holding the largest value. op is \"max\"/\"min\" "
        '(no value) or a comparison "==", "!=", "<", "<=", ">", ">=" '
        "against value; the matching coordinates are returned. Use it for "
        "which-is-larger / which-is-smaller questions. Resolve row/col via "
        "find_header / get_table_overview first."
    )
    args_schema: type[BaseModel] = _CompareCellsInput
    exposure_type: str = "data"
    table_path: str = ""

    def validate(self, **kwargs) -> tuple[bool, str]:
        cells = kwargs.get("cells", [])
        if not cells or len(cells) > _MAX_CELLS:
            return False, f"Provide 2..{_MAX_CELLS} cells"
        for c in cells:
            if not isinstance(c, dict) or "row" not in c or "col" not in c:
                return False, "Each cell needs 'row' and 'col' keys"
        op = kwargs.get("op")
        if op in ("max", "min"):
            return True, ""
        if op in ("==", "!=", "<", "<=", ">", ">="):
            if kwargs.get("value") is None:
                return False, f'op "{op}" needs a value (number or {{ placeholder }})'
            return True, ""
        return False, (
            'op must be "max", "min", or one of "==", "!=", "<", "<=", ">", ">="'
        )

    def execute(self, **kwargs) -> ToolResult:
        table = _load_table(self.table_path)
        cells = kwargs.get("cells", [])
        op = kwargs.get("op")
        value = kwargs.get("value")

        entries: list[dict] = []
        for c in cells:
            r, col = int(c["row"]), int(c["col"])
            if not (0 <= r < table["n_rows"] and 0 <= col < table["n_cols"]):
                return ToolResult(
                    output=(
                        f"Error: address (row={r}, col={col}) outside data grid "
                        f"({table['n_rows']} rows x {table['n_cols']} cols)"
                    ),
                    error="out of bounds",
                )
            v = table["grid"][r][col]
            entries.append(
                {"row": r, "col": col, "value": v, "num": _as_number(v)}
            )

        if op in ("max", "min"):
            numeric = [e for e in entries if e["num"] is not None]
            if not numeric:
                return ToolResult(
                    output=(
                        "Error: none of the given cells is numeric — "
                        "compare_cells needs numeric cells"
                    ),
                    error="non-numeric cells",
                )
            pick = max if op == "max" else min
            best = pick(numeric, key=lambda e: e["num"])  # ties: first given
            matched = [best]
        else:
            val_num = _as_number(value)
            val_str = str(value).strip().lower()
            matched = []
            for e in entries:
                if op == "==":
                    ok = (
                        e["num"] is not None
                        and val_num is not None
                        and e["num"] == val_num
                    ) or str(e["value"]).strip().lower() == val_str
                elif op == "!=":
                    ok = not (
                        (
                            e["num"] is not None
                            and val_num is not None
                            and e["num"] == val_num
                        )
                        or str(e["value"]).strip().lower() == val_str
                    )
                else:
                    ok = (
                        e["num"] is not None
                        and val_num is not None
                        and {
                            "<": e["num"] < val_num,
                            "<=": e["num"] <= val_num,
                            ">": e["num"] > val_num,
                            ">=": e["num"] >= val_num,
                        }[op]
                    )
                if ok:
                    matched.append(e)

        if op in ("max", "min"):
            word = "largest" if op == "max" else "smallest"
            head = f"{word} of {len(entries)} cell(s):"
        else:
            head = (
                f'{len(matched)} of {len(entries)} cell(s) satisfy "{op}" '
                "condition:"
            )
        out_lines = [f"(row={e['row']}, col={e['col']}): {e['value']!r}" for e in matched]
        out_cells = [
            {"row": e["row"], "col": e["col"], "value": e["value"], "type": _value_type(e["value"])}
            for e in matched
        ]
        return ToolResult(
            output=head + "\n" + "\n".join(out_lines) if out_lines else head,
            row_count=len(matched),
            data={"cells": out_cells, "match_count": len(matched), "op": op},
        )

    def generate_metadata(self, result: ToolResult) -> str:
        if result.error:
            return f"[EAG] compare_cells failed — {result.output}"
        cells = result.data["cells"]
        op = result.data.get("op")
        addrs = ", ".join(f"(row={c['row']}, col={c['col']})" for c in cells)
        if op in ("max", "min"):
            word = "largest" if op == "max" else "smallest"
            if not cells:
                return "[EAG] compare_cells → no numeric cell among the given addresses"
            return f"[EAG] compare_cells → {word}: {addrs}"
        return (
            f"[EAG] compare_cells → {len(cells)} match(es); addresses kept, "
            f"values withheld:\n{addrs if cells else '(none)'}"
        )


# ── Local compute ─────────────────────────────────────────────────────────────


class AggregateTool(EvaluatorTool):
    """Multi-operand aggregation computed locally (sum/avg/min/max)."""

    name: str = "aggregate"
    description: str = (
        "Aggregate N numbers locally (ops: sum, avg, min, max) — e.g. "
        'aggregate(op="sum", items=["{{ get_cells.cells[0].value }}", '
        '"{{ get_cells.cells[1].value }}"]). Reference the outcome as '
        "{{ aggregate.result }}."
    )
    args_schema: type[BaseModel] = _AggregateInput

    def validate(self, **kwargs) -> tuple[bool, str]:
        if kwargs.get("op") not in ("sum", "avg", "min", "max"):
            return False, "op must be one of ['sum', 'avg', 'min', 'max']"
        items = kwargs.get("items", [])
        if not isinstance(items, list) or len(items) < 2:
            return False, "items must be a list of at least 2 numbers"
        return True, ""

    def execute(self, **kwargs) -> ToolResult:
        op = kwargs["op"]
        nums: list[float] = []
        for item in kwargs["items"]:
            n = _as_number(item)
            if n is None:
                return ToolResult(
                    output=(
                        f"Error: item {item!r} is not a number — pass numeric "
                        "literals or {{ placeholder }} references resolving to "
                        "numbers"
                    ),
                    error="non-numeric operand",
                )
            nums.append(n)
        result = {
            "sum": sum(nums),
            "avg": sum(nums) / len(nums),
            "min": min(nums),
            "max": max(nums),
        }[op]
        pretty = int(result) if result == int(result) else result
        return ToolResult(
            output=f"{op} of {len(nums)} numbers: {pretty}",
            row_count=1,
            data={"result": pretty, "op": op, "n_items": len(nums)},
        )


# ── Evaluation tool ───────────────────────────────────────────────────────────


class SubmitAnswerTool(EAGBaseTool):
    """Submit the final answer for evaluation."""

    name: str = "submit_answer"
    description: str = (
        "Submit your final answer. Call this when you know the answer (a "
        "literal in baseline mode; a {{ placeholder }} reference in EAG "
        'mode, e.g. "{{ aggregate.result }}"). After calling submit_answer, '
        "respond with a brief text answer — do not call any more tools."
    )
    args_schema: type[BaseModel] = _SubmitAnswerInput
    exposure_type: str = "evaluation"

    def execute(self, **kwargs) -> ToolResult:
        answer = kwargs.get("answer")
        answer = answer if isinstance(answer, str) else str(answer)
        return ToolResult(
            output=(
                "Answer submitted successfully. It will be evaluated against "
                "the gold answer."
            ),
            data={"submitted": answer},
        )


# ── Factory ───────────────────────────────────────────────────────────────────


def make_hitab_tools(table_path: str, eag_mode: bool = False) -> list[BaseTool]:
    """Create the 8 HiTab tools for one table.

    Args:
        table_path: Path to the task's ``hmt``-variant table JSON.
        eag_mode: If True, data tools return metadata-only observations.
    """
    return [
        TableOverviewTool(table_path=table_path, eag_mode=eag_mode),
        FindHeaderTool(table_path=table_path, eag_mode=eag_mode),
        GetCellsTool(table_path=table_path, eag_mode=eag_mode),
        FindCellsTool(table_path=table_path, eag_mode=eag_mode),
        CompareCellsTool(table_path=table_path, eag_mode=eag_mode),
        AggregateTool(eag_mode=eag_mode),
        ComputeTool(eag_mode=eag_mode),
        SubmitAnswerTool(eag_mode=eag_mode),
    ]
