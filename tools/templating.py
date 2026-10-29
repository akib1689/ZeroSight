"""Deferred placeholder substitution for EAG — unified path grammar.

In EAG mode the model never sees raw data values, so it cannot state a computed
result or pass a record identifier to a downstream tool. Instead it embeds
symbolic references to prior tool-call results, and this module resolves them
*locally* — mid-run against tool-call arguments (via the EAG placeholder
middleware) and post-run against the final answer. Raw values are bound from
artifacts the harness captured (but never showed the model).

## Handle grammar (which step's artifact)

**By tool name** (recommended — the model always knows which tool it called)::

    {{ get_order_details.items[2].item_id }}   last get_order_details result
    {{ execute_sql.col.ratio[0] }}             last execute_sql result, cell
    {{ execute_sql[0].col.x }}                 FIRST execute_sql call (0-based)
    {{ get_user_details[1].orders[0] }}        SECOND get_user_details call

**By global step number** (``step[N]`` is the Nth tool call overall, 1-based,
in the order issued)::

    {{ step[6].items[0].item_id }}

## Access grammar (where in the artifact) — modality-agnostic

The artifact is stored as a parsed JSON tree (``ToolResult.data``). The access
suffix navigates it generically — ``.key`` for object fields, ``[N]`` for
array elements (0-based). Tabular artifacts (``{"columns": [...], "rows":
[...]}``) additionally support SQL conveniences::

    .col.<column_name>[R]   single cell (row R, 0-based) — tabular sugar
    .col.<column_name>      whole column as "v1, v2, ..."
    .rows[R][C]             cell by numeric indices
    .rows[R]                one row
    .rows                   whole result table (formatted)
    .row_count              integer row count
    (empty access)          the entire artifact

Generic navigation examples (nested JSON artifacts)::

    {{ get_user_details.orders[0] }}
    {{ get_order_details.items[2].item_id }}
    {{ get_user_details.payment_methods.credit_card_9513926.id }}
    {{ find_user_id_by_name_zip }}                  scalar artifact, whole value

An unresolved placeholder (missing step, no captured data, bad path) is
replaced with a visible sentinel like ``<unavailable: ...>`` in answers, or
reported as an error by ``resolve_tool_args`` mid-run so the model can retry.

Contract: each action step in ``steps`` may carry ``raw_result`` holding the
parsed artifact (set by the agent from the langchain ``ToolMessage.artifact``).
Steps without ``raw_result`` (metadata/evaluation tools, errored calls) are
not referenceable and do not occupy an occurrence index — ``tool[N]`` counts
only results with captured data.
"""

import json
import re
from typing import Any

_PLACEHOLDER_RE = re.compile(r"\{\{\s*(.*?)\s*\}\}")
_FULL_PLACEHOLDER_RE = re.compile(r"^\{\{\s*(.*?)\s*\}\}$")

# Handle = "step[N]" | <tool_name> (alphanumeric/underscore). An optional
# [occurrence] suffix is parsed separately so it works for tool names only.
_HANDLE_RE = re.compile(r"^(step\[\d+\]|[A-Za-z_]\w*)")

# Tabular sugar: col.<name>[R]? | rows[R][C]? — anchored by the caller.
_COL_RE = re.compile(r"col\.([^\[\]]+)(?:\[(\d+)\])?")

# Generic path tokens: .key (identifier, allows hyphens) or [N].
_TOKEN_RE = re.compile(r"\.([A-Za-z_][A-Za-z0-9_-]*)|\[(\d+)\]")


def _find_col(columns: list[str], name: str) -> int | None:
    """Case-insensitive column lookup; returns index or None."""
    target = name.strip().lower()
    for i, c in enumerate(columns):
        if c.lower() == target:
            return i
    return None


def _format_table(columns: list[str], rows: list[list]) -> str:
    """Format raw rows as a readable table (mirrors tools.sql._format_results)."""
    header = " | ".join(columns)
    if not rows:
        return f"{header}\n\n(0 rows returned)"
    lines = [header] + [" | ".join(str(v) for v in row) for row in rows]
    return "\n".join(lines) + f"\n\n({len(rows)} rows returned)"


def _format_value(value: Any) -> str:
    """Format a resolved value for string interpolation."""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _tokenize(access: str) -> list[tuple[str, Any]] | None:
    """Split an access string into navigation tokens.

    Returns ``[("key", name) | ("idx", int), ...]`` or None if the access
    contains unparseable segments. A leading bare identifier (no dot) is
    accepted, matching the legacy ``rows[0]`` form.
    """
    if access and access[0] not in ".[":
        access = "." + access
    tokens: list[tuple[str, Any]] = []
    pos = 0
    while pos < len(access):
        m = _TOKEN_RE.match(access, pos)
        if not m:
            return None
        if m.group(1) is not None:
            tokens.append(("key", m.group(1)))
        else:
            tokens.append(("idx", int(m.group(2))))
        pos = m.end()
    return tokens


def _navigate(node: Any, tokens: list[tuple[str, Any]]) -> Any:
    """Walk the artifact tree along tokens; returns a sentinel on failure."""
    for kind, ref in tokens:
        if kind == "key":
            if not isinstance(node, dict) or ref not in node:
                # case-insensitive fallback
                if isinstance(node, dict):
                    for k in node:
                        if k.lower() == ref.lower():
                            ref = k
                            break
                    else:
                        return _MISSING
                else:
                    return _MISSING
            node = node[ref]
        else:  # idx
            if not isinstance(node, list) or ref >= len(node) or ref < 0:
                return _MISSING
            node = node[ref]
    return node


class _Missing:
    """Sentinel distinguishing 'resolved to None' from 'path failed'."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover
        return "<missing>"


_MISSING = _Missing()


def _is_tabular(raw: Any) -> bool:
    return (
        isinstance(raw, dict)
        and isinstance(raw.get("columns"), list)
        and isinstance(raw.get("rows"), list)
    )


def _eval_access(access: str, raw: Any) -> Any:
    """Resolve the access suffix against an artifact.

    Returns the referenced value (any type — callers format or use directly),
    or the ``_MISSING`` sentinel if the reference is invalid.
    """
    access = access.strip()

    # Whole artifact.
    if access == "":
        if _is_tabular(raw):
            return _format_table(raw["columns"], raw["rows"]) \
                if raw["rows"] else _MISSING
        return raw

    # Integer row count (tabular) / list length (generic).
    if access == "row_count":
        if _is_tabular(raw):
            return len(raw["rows"])
        if isinstance(raw, list):
            return len(raw)
        return _MISSING

    # Tabular sugar: col.<name>[R]? — runs before generic navigation so
    # column names never collide with JSON keys.
    m = _COL_RE.fullmatch(access)
    if m and _is_tabular(raw):
        columns: list[str] = raw["columns"]
        rows: list[list] = raw["rows"]
        idx = _find_col(columns, m.group(1))
        if idx is None:
            return _MISSING
        if m.group(2) is None:
            vals = [row[idx] for row in rows if idx < len(row)]
            return ", ".join(str(v) for v in vals) if vals else _MISSING
        r = int(m.group(2))
        if r >= len(rows) or idx >= len(rows[r]):
            return _MISSING
        return rows[r][idx]

    # Generic token navigation.
    tokens = _tokenize(access)
    if tokens is None:
        return _MISSING
    return _navigate(raw, tokens)


def _resolve_step(inner: str, steps: list[dict], by_num: dict, by_action: dict):
    """Parse a placeholder's inner text and return ``(raw_result, label)``.

    ``label`` is a human-readable description for the sentinel. Returns
    ``(None, label)`` if the referenced step cannot be resolved.
    """
    m = _HANDLE_RE.match(inner)
    if not m:
        return None, inner.strip() or "value"
    handle = m.group(1)
    rest = inner[m.end():]

    if handle.startswith("step["):
        # Global step-number form: step[N] (.access...)
        access = rest[1:] if rest.startswith(".") else rest
        step_num = int(handle[5:-1])
        step = by_num.get(step_num)
        raw = step.get("raw_result") if step else None
        label = f"step[{step_num}].{access.strip() or 'value'}"
        return raw, label

    # Tool-name form: <name>[occ]? (.access...)
    occ: int | str | None = None
    if rest.startswith("["):
        end = rest.find("]")
        if end == -1:
            return None, inner.strip()
        occ_str = rest[1:end]
        occ = "last" if occ_str == "last" else (int(occ_str) if occ_str.isdigit() else None)
        if occ is None:
            return None, inner.strip()
        rest = rest[end + 1:]
    access = rest[1:] if rest.startswith(".") else rest

    calls = by_action.get(handle, [])
    if not calls:
        label = f"{handle}.{access.strip() or 'value'}"
        return None, label
    if occ is None or occ == "last":
        step = calls[-1]
        occ_lbl = "last"
    else:
        if occ >= len(calls):
            label = f"{handle}[{occ}].{access.strip() or 'value'}"
            return None, label
        step = calls[occ]
        occ_lbl = str(occ)
    raw = step.get("raw_result")
    label = f"{handle}[{occ_lbl}].{access.strip() or 'value'}"
    return raw, label


def _resolve_value(inner: str, steps: list[dict], by_num: dict, by_action: dict):
    """Resolve a placeholder's inner text to its typed value.

    Returns ``(value, ok)``. ``value`` is the raw referenced value (any type)
    when ``ok`` is True; otherwise an error label string.
    """
    raw, label = _resolve_step(inner, steps, by_num, by_action)
    if raw is None:
        return label, False
    m = _HANDLE_RE.match(inner)
    access = inner[m.end():] if m else inner
    occ: int | None = None
    if access.startswith("["):
        end = access.find("]")
        occ_str = access[1:end]
        if occ_str.isdigit():
            occ = int(occ_str)
        access = access[end + 1:]
    access = access[1:] if access.startswith(".") else access
    value = _eval_access(access, raw)
    if value is _MISSING and occ is not None and isinstance(raw, list):
        # Element-index fallback: for list artifacts, ``tool[N].path`` may
        # intend element N of the latest result (models naturally read
        # ``[N]`` as row indexing), not the Nth historical call of that
        # tool. Retry with ``[N]`` as a leading index into the artifact.
        combined = f"[{occ}]" + (f".{access}" if access else "")
        value = _eval_access(combined, raw)
    if value is _MISSING or value is None:
        return label, False
    return value, True


def _index_steps(steps: list[dict]) -> tuple[dict, dict]:
    """Build step-number and tool-name indexes over action steps.

    Occurrence indices (``tool_name[N]``) count only steps with a captured
    artifact (``raw_result``): errored calls leave no referenceable data, so
    they do not consume an index — a model recovering from a failed call
    never hits a dead occurrence.
    """
    action_steps = [s for s in steps if s.get("type") == "action" and "step" in s]
    by_num = {s["step"]: s for s in action_steps}
    by_action: dict[str, list[dict]] = {}
    for s in action_steps:
        if s.get("raw_result") is None:
            continue
        by_action.setdefault(s.get("action", ""), []).append(s)
    return by_num, by_action


def substitute_placeholders(text: str, steps: list[dict]) -> tuple[str, int]:
    """Resolve ``{{ ... }}`` placeholders in ``text`` (final answers).

    Args:
        text: The model's raw final answer (may contain placeholders).
        steps: The trajectory steps; action steps indexed by their ``step``
            number and grouped by ``action`` (tool name), optionally carrying
            ``raw_result``.

    Returns:
        ``(resolved_text, unresolved_count)``. ``resolved_text`` is the answer
        with all valid placeholders spliced in and invalid ones replaced by a
        ``<unavailable: ...>`` sentinel. ``unresolved_count`` is the number of
        placeholders that could not be resolved (for fidelity metrics).
    """
    if not text:
        return text, 0

    by_num, by_action = _index_steps(steps)
    unresolved = 0

    def _replace(match: re.Match) -> str:
        nonlocal unresolved
        inner = match.group(1)
        value, ok = _resolve_value(inner, steps, by_num, by_action)
        if not ok:
            unresolved += 1
            return f"<unavailable: {value}>"
        return _format_value(value)

    return _PLACEHOLDER_RE.sub(_replace, text), unresolved


def resolve_tool_args(args: Any, steps: list[dict]) -> tuple[Any, list[str]]:
    """Resolve ``{{ ... }}`` placeholders inside tool-call arguments (mid-run).

    Recurses through dicts and lists. A string that is *exactly* one
    placeholder is replaced by the referenced value with its type preserved
    (so numeric/array fields stay numeric/array); placeholders embedded in a
    larger string are interpolated as text.

    Args:
        args: The tool-call arguments (typically a dict).
        steps: Trajectory-like steps carrying ``raw_result`` artifacts.

    Returns:
        ``(resolved_args, errors)``. ``errors`` lists the inner texts of
        placeholders that could not be resolved — the caller should fail the
        tool call so the model can retry with a corrected path.
    """
    errors: list[str] = []
    by_num, by_action = _index_steps(steps)

    def _resolve_str(s: str) -> str:
        full = _FULL_PLACEHOLDER_RE.match(s.strip())
        if full:
            value, ok = _resolve_value(full.group(1), steps, by_num, by_action)
            if not ok:
                errors.append(full.group(1))
                return s
            return value
        return _PLACEHOLDER_RE.sub(_interpolate, s)

    def _interpolate(match: re.Match) -> str:
        value, ok = _resolve_value(match.group(1), steps, by_num, by_action)
        if not ok:
            errors.append(match.group(1))
            return match.group(0)
        return _format_value(value)

    def _walk(node: Any) -> Any:
        if isinstance(node, str):
            return _resolve_str(node)
        if isinstance(node, list):
            return [_walk(v) for v in node]
        if isinstance(node, dict):
            return {k: _walk(v) for k, v in node.items()}
        return node

    return _walk(args), errors
