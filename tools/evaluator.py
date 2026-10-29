"""Evaluator tools — local compute tier for EAG agents.

In EAG mode the model never sees raw values, so it cannot compute over them
in-context. These seven tools compute *locally* over referenced artifacts:
the middleware resolves ``{{ }}`` placeholders in the arguments before
execution (type-preserving), the tool computes, and the result rides the
``ToolMessage.artifact`` — the observation stays shape-only (result *type*
is disclosed, never the value). The model answers symbolically, e.g.
``{{ compare.result }}``.

Dataset-agnostic by design: operands are plain values or placeholder
references; nothing here knows about the employees domain.

Leak rules:
- observations disclose structure and types only
- error messages name types/paths, never operand values
- permissive operand schemas (str|int|float|bool|list|dict) so middleware-
  resolved values never hit pydantic validation failures (whose error echo
  would leak the resolved value back into the model context)
"""

import json
import re
from typing import Any, Literal

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from tools.sql import EAGBaseTool, ToolResult

_VALUE_TYPES = (str, int, float, bool)
Operand = str | int | float | bool
RefValue = list | dict | str | int | float | bool

_COMPARE_OPS = ("==", "!=", "<", "<=", ">", ">=")
_COMPUTE_OPS = ("+", "-", "*", "/")

# ISO dates/datetimes compare lexicographically — detect them so ordered
# comparisons work on 'YYYY-MM-DD[THH:MM[:SS]]' strings (lindisfarne temporal).
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2})?(\.\d+)?Z?)?$")


def _iso_date_pair(a: Any, b: Any) -> tuple[str, str] | None:
    """Both operands as ISO date/datetime strings, else None."""
    if isinstance(a, str) and isinstance(b, str):
        a_s, b_s = a.strip(), b.strip()
        if _ISO_DATE_RE.match(a_s) and _ISO_DATE_RE.match(b_s):
            return a_s, b_s
    return None


def _type_name(v: Any) -> str:
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "str"
    if isinstance(v, list):
        return "list"
    if isinstance(v, dict):
        return "object"
    return type(v).__name__


def _as_number(v: Any) -> int | float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return v
    if isinstance(v, str):
        s = v.strip()
        try:
            return int(s)
        except ValueError:
            try:
                return float(s)
            except ValueError:
                return None
    return None


def _scalar_eq(a: Any, b: Any) -> bool:
    numeric = (
        isinstance(a, (int, float))
        and isinstance(b, (int, float))
        and not isinstance(a, bool)
        and not isinstance(b, bool)
    )
    if numeric or type(a) is type(b):
        return bool(a == b)
    return str(a) == str(b)


def _to_text(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def _shape_of(value: Any, depth: int = 0) -> str:
    if isinstance(value, dict):
        if depth >= 4:
            return "{...}"
        inner = ", ".join(f"{k}: {_shape_of(v, depth + 1)}" for k, v in value.items())
        return "{" + inner + "}"
    if isinstance(value, list):
        if not value:
            return "[] (empty)"
        return f"[{len(value)} items, each {_shape_of(value[0], depth + 1)}]"
    return f"<{_type_name(value)}>"


# ── Input schemas ────────────────────────────────────────────────────────────


class _CompareInput(BaseModel):
    left: Operand = Field(
        ..., description="Left value — a literal or a {{ placeholder }} reference"
    )
    op: Literal["==", "!=", "<", "<=", ">", ">="]
    right: Operand = Field(
        ..., description="Right value — a literal or a {{ placeholder }} reference"
    )


class _ComputeInput(BaseModel):
    a: Operand = Field(..., description="First number or {{ placeholder }} reference")
    op: Literal["+", "-", "*", "/"]
    b: Operand = Field(..., description="Second number or {{ placeholder }} reference")


class _ConcatInput(BaseModel):
    parts: list[Operand] = Field(
        min_length=1,
        description="Values to join — literals and/or {{ placeholder }} references",
    )
    separator: str = Field(" ", description="Join separator (default single space)")


class _CountInput(BaseModel):
    ref: RefValue = Field(
        ...,
        description="A list result referenced as {{ tool }} (or an object with an "
        "'items' list, referenced as {{ search_employees }})",
    )


class _FilterInput(BaseModel):
    ref: RefValue = Field(
        ...,
        description="List to search — a {{ placeholder }} reference to a prior "
        "list-valued result (or an object with an 'items' list)",
    )
    key: str = Field(..., description="Field name to match on each item")
    value: Operand = Field(
        ..., description="Value the field must equal — literal or {{ placeholder }}"
    )


class _SortItemsInput(BaseModel):
    ref: RefValue = Field(
        ...,
        description="List to sort — a {{ placeholder }} reference to a prior "
        "list-valued result (an object with an 'items' or 'matches' list works too)",
    )
    key: str | None = Field(
        None,
        description="Item field to sort by when items are objects "
        "(omit for scalar lists)",
    )
    order: Literal["asc", "desc"] = Field("asc", description="Sort direction (default asc)")


class _JoinItemsInput(BaseModel):
    ref: RefValue = Field(
        ...,
        description="List to join — a {{ placeholder }} reference to a prior "
        "list-valued result (an object with an 'items' or 'matches' list works too)",
    )
    key: str | None = Field(
        None,
        description="Item field to join when items are objects, e.g. key='name' "
        "over sorted matches (omit for scalar lists)",
    )
    separator: str = Field(", ", description="Join separator (default ', ')")


# ── Base ─────────────────────────────────────────────────────────────────────


class EvaluatorTool(EAGBaseTool):
    """Compute tool with shape-only EAG observations (result type, not value)."""

    exposure_type: str = "data"

    def generate_metadata(self, result: ToolResult) -> str:
        if result.error:
            return f"[EAG] {self.name} failed — {result.output}"
        data = result.data or {}
        if "result" in data:
            return (
                f"[EAG] {self.name} → OK. Result type: {_shape_of(data['result'])}\n"
                f"Reference it as {{{{ {self.name}.result }}}}"
            )
        matches = data.get("matches")
        if isinstance(matches, list):
            if matches:
                head = f"[EAG] {self.name} → OK. Matches: {_shape_of(matches)}"
            else:
                head = f"[EAG] {self.name} → OK. Matches: none"
            return (
                f"{head}\n"
                f"Reference them as {{{{ {self.name}.matches[<i>].<key> }}}}"
            )
        return f"[EAG] {self.name} → OK. Structure: {_shape_of(data)}"


# ── Tools ────────────────────────────────────────────────────────────────────


class CompareTool(EvaluatorTool):
    name: str = "compare"
    description: str = (
        "Compare two values locally and get a boolean — for true/false "
        "questions you cannot answer by seeing values. Pass literals or "
        "{{ placeholder }} references (e.g. left=\"{{ get_employee.salary }}\", "
        'op=">=", right=64423). Ordered ops work on numbers and ISO dates '
        "(YYYY-MM-DD). The outcome is referenced symbolically: "
        "answer with {{ compare.result }}."
    )
    args_schema: type[BaseModel] = _CompareInput

    def validate(self, **kwargs) -> tuple[bool, str]:
        if kwargs.get("op") not in _COMPARE_OPS:
            return False, f"op must be one of {list(_COMPARE_OPS)}"
        return True, ""

    def execute(self, **kwargs) -> ToolResult:
        left, op, right = kwargs["left"], kwargs["op"], kwargs["right"]
        if op in ("==", "!="):
            result = _scalar_eq(left, right)
            if op == "!=":
                result = not result
        else:
            ln, rn = _as_number(left), _as_number(right)
            date_pair = None if ln is not None and rn is not None else _iso_date_pair(left, right)
            if ln is not None and rn is not None:
                result = {
                    "<": ln < rn,
                    "<=": ln <= rn,
                    ">": ln > rn,
                    ">=": ln >= rn,
                }[op]
            elif date_pair is not None:
                ls, rs = date_pair
                result = {
                    "<": ls < rs,
                    "<=": ls <= rs,
                    ">": ls > rs,
                    ">=": ls >= rs,
                }[op]
            else:
                return ToolResult(
                    output=(
                        f"ordered comparison needs numbers or ISO dates — got "
                        f"{_type_name(left)} and {_type_name(right)}. Reference "
                        "numeric/date fields, or use ==/!= for values of any type."
                    ),
                    error="type mismatch",
                )
        return ToolResult(
            output=json.dumps({"result": result}),
            data={"result": result},
            row_count=1,
        )


class ComputeTool(EvaluatorTool):
    name: str = "compute"
    description: str = (
        "Arithmetic on two numbers locally (ops: + - * /) — e.g. differences "
        "between two referenced values: compute(a=\"{{ get_salary_stats[0].avg_salary }}\", "
        'op="-", b=50000). Reference the outcome as {{ compute.result }}.'
    )
    args_schema: type[BaseModel] = _ComputeInput

    def validate(self, **kwargs) -> tuple[bool, str]:
        if kwargs.get("op") not in _COMPUTE_OPS:
            return False, f"op must be one of {list(_COMPUTE_OPS)}"
        return True, ""

    def execute(self, **kwargs) -> ToolResult:
        a, op, b = kwargs["a"], kwargs["op"], kwargs["b"]
        an, bn = _as_number(a), _as_number(b)
        if an is None or bn is None:
            return ToolResult(
                output=(
                    f"compute needs numbers — got {_type_name(a)} and "
                    f"{_type_name(b)}. Use concat to join strings instead."
                ),
                error="type mismatch",
            )
        if op == "/" and bn == 0:
            return ToolResult(output="division by zero is not allowed.", error="div zero")
        result: int | float
        if op == "+":
            result = an + bn
        elif op == "-":
            result = an - bn
        elif op == "*":
            result = an * bn
        else:
            result = an / bn
        return ToolResult(
            output=json.dumps({"result": result}),
            data={"result": result},
            row_count=1,
        )


class ConcatTool(EvaluatorTool):
    name: str = "concat"
    description: str = (
        "Join values into one string locally — e.g. build a full name: "
        'concat(parts=["{{ get_top_earners[0].first_name }}", '
        '"{{ get_top_earners[0].last_name }}"]). Reference as {{ concat.result }}.'
    )
    args_schema: type[BaseModel] = _ConcatInput

    def execute(self, **kwargs) -> ToolResult:
        parts = kwargs["parts"]
        sep = kwargs.get("separator", " ")
        result = sep.join(_to_text(p) for p in parts)
        return ToolResult(
            output=json.dumps({"result": result}),
            data={"result": result},
            row_count=1,
        )


class CountItemsTool(EvaluatorTool):
    name: str = "count_items"
    description: str = (
        "Count items in a list result locally — e.g. how many salary records: "
        'count_items(ref="{{ get_salary_history }}"). For filtered searches '
        "prefer the search's own total field. Reference as {{ count_items.result }}."
    )
    args_schema: type[BaseModel] = _CountInput

    def execute(self, **kwargs) -> ToolResult:
        ref = kwargs["ref"]
        if isinstance(ref, dict) and isinstance(ref.get("items"), list):
            n = len(ref["items"])
        elif isinstance(ref, list):
            n = len(ref)
        else:
            return ToolResult(
                output=(
                    f"ref must be a list (or an object with an 'items' list) — "
                    f"got {_type_name(ref)}. Reference the list-valued field "
                    "directly, e.g. {{ tool.items }}."
                ),
                error="type mismatch",
            )
        return ToolResult(
            output=json.dumps({"result": n}),
            data={"result": n},
            row_count=1,
        )


class FilterItemsTool(EvaluatorTool):
    name: str = "filter_items"
    description: str = (
        "Pick matching rows from a list result locally — use instead of "
        "guessing list indexes when you cannot see values: "
        'filter_items(ref="{{ get_reference.departments }}", key="dept_name", '
        'value="Sales"), then reference {{ filter_items.matches[0].dept_no }}. '
        "value may itself be a {{ placeholder }}."
    )
    args_schema: type[BaseModel] = _FilterInput

    def execute(self, **kwargs) -> ToolResult:
        ref, key, value = kwargs["ref"], kwargs["key"], kwargs["value"]
        if isinstance(ref, dict) and isinstance(ref.get("items"), list):
            target = ref["items"]
        elif isinstance(ref, list):
            target = ref
        else:
            return ToolResult(
                output=(
                    f"ref must be a list (or an object with an 'items' list) — "
                    f"got {_type_name(ref)}."
                ),
                error="type mismatch",
            )
        if not any(isinstance(it, dict) and key in it for it in target):
            sample = next((it for it in target if isinstance(it, dict)), None)
            keys = sorted(sample.keys()) if isinstance(sample, dict) else []
            return ToolResult(
                output=(
                    f"no item has field '{key}'. Fields present: {keys or 'none'}."
                ),
                error="missing key",
            )
        matches = [
            it for it in target
            if isinstance(it, dict) and key in it and _scalar_eq(it[key], value)
        ]
        return ToolResult(
            output=json.dumps({"matches": matches}, ensure_ascii=False),
            data={"matches": matches},
            row_count=len(matches),
        )


# ── Local compute (list shaping) ──────────────────────────────────────────────


def _as_list(ref: Any, wrapper_keys: tuple[str, ...] = ("items", "matches")) -> list | str:
    """Unwrap a list artifact (or an object holding one). Returns list or error."""
    if isinstance(ref, dict):
        for k in wrapper_keys:
            if isinstance(ref.get(k), list):
                return ref[k]
    if isinstance(ref, list):
        return ref
    return (
        f"ref must be a list (or an object with an 'items'/'matches' list) — "
        f"got {_type_name(ref)}."
    )


class SortItemsTool(EvaluatorTool):
    """Sort a list artifact locally (numeric-aware; ISO dates sort correctly)."""

    name: str = "sort_items"
    description: str = (
        "Sort a list result locally — use for alphabetical names or ascending "
        "IDs you cannot see: sort_items(ref=\"{{ find_records.matches }}\", "
        'key="name", order="asc"), then reference {{ sort_items.result }} (a '
        "sorted list). All-numeric lists sort numerically; otherwise "
        "lexicographically (ISO dates stay chronological)."
    )
    args_schema: type[BaseModel] = _SortItemsInput

    def execute(self, **kwargs) -> ToolResult:
        ref = kwargs["ref"]
        key = kwargs.get("key")
        order = kwargs.get("order", "asc")
        target = _as_list(ref)
        if isinstance(target, str):
            return ToolResult(output=target, error="type mismatch")

        if key is not None:
            if not any(isinstance(it, dict) and key in it for it in target):
                sample = next((it for it in target if isinstance(it, dict)), None)
                keys = sorted(sample.keys()) if isinstance(sample, dict) else []
                return ToolResult(
                    output=f"no item has field '{key}'. Fields present: {keys or 'none'}.",
                    error="missing key",
                )

            def val(it: Any) -> Any:
                return it.get(key) if isinstance(it, dict) else None
        else:
            def val(it: Any) -> Any:
                return it

        non_null = [it for it in target if val(it) is not None]
        null = [it for it in target if val(it) is None]
        nums = [_as_number(val(it)) for it in non_null]
        if non_null and all(n is not None for n in nums):
            ranked = sorted(non_null, key=lambda it: _as_number(val(it)))
        else:
            ranked = sorted(non_null, key=lambda it: str(val(it)).lower())
        if order == "desc":
            ranked.reverse()
        result = ranked + null
        return ToolResult(
            output=json.dumps({"result": result}, ensure_ascii=False),
            data={"result": result, "n": len(result), "order": order},
            row_count=len(result),
        )


class JoinItemsTool(EvaluatorTool):
    """Join a list artifact into one string locally (optional field projection)."""

    name: str = "join_items"
    description: str = (
        "Join a list result into one string locally — e.g. render a sorted "
        'name list: join_items(ref="{{ sort_items.result }}", key="name", '
        'separator=", "). key picks an item field when items are objects; '
        "omit it for scalar lists. Reference the outcome as "
        "{{ join_items.result }}."
    )
    args_schema: type[BaseModel] = _JoinItemsInput

    def execute(self, **kwargs) -> ToolResult:
        ref = kwargs["ref"]
        key = kwargs.get("key")
        sep = kwargs.get("separator", ", ")
        target = _as_list(ref)
        if isinstance(target, str):
            return ToolResult(output=target, error="type mismatch")
        if key is not None:
            if not any(isinstance(it, dict) and key in it for it in target):
                sample = next((it for it in target if isinstance(it, dict)), None)
                keys = sorted(sample.keys()) if isinstance(sample, dict) else []
                return ToolResult(
                    output=f"no item has field '{key}'. Fields present: {keys or 'none'}.",
                    error="missing key",
                )
            parts = [it.get(key) for it in target if isinstance(it, dict) and key in it]
        else:
            parts = target
        result = sep.join(_to_text(x) for x in parts)
        return ToolResult(
            output=json.dumps({"result": result}, ensure_ascii=False),
            data={"result": result, "n": len(parts)},
            row_count=1,
        )


# ── Factory ──────────────────────────────────────────────────────────────────


def make_evaluator_tools(eag_mode: bool = False) -> list[BaseTool]:
    """The 7 local compute tools (compare, compute, concat, count_items,
    filter_items, sort_items, join_items). Same surface in baseline and EAG
    modes — only the observation redaction differs."""
    return [
        CompareTool(eag_mode=eag_mode),
        ComputeTool(eag_mode=eag_mode),
        ConcatTool(eag_mode=eag_mode),
        CountItemsTool(eag_mode=eag_mode),
        FilterItemsTool(eag_mode=eag_mode),
        SortItemsTool(eag_mode=eag_mode),
        JoinItemsTool(eag_mode=eag_mode),
    ]
