"""EAG placeholder middleware — resolves ``{{ ... }}`` in tool args mid-run.

In EAG mode the model cannot see raw record values (order ids, item ids,
payment method ids, …), but downstream tools need them as arguments. The
system prompt lets the model reference prior tool results symbolically; this
middleware intercepts every tool call *before* execution, resolves those
references locally against artifacts captured in the conversation state, and
passes clean values to the tool. The model never sees the raw values — the
EAG guarantee is preserved while the referential chain stays connected.

Unresolvable placeholders fail the tool call with an explanatory error
listing the valid references for the cited tools (occurrence status + result
shape — structure only, never values), so the model can correct the path and
retry in one step instead of looping. Hallucinated identifiers become
impossible to act on.
"""

from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

from tools.templating import _HANDLE_RE, resolve_tool_args

_MAX_OCC_SHOWN = 4
_MAX_SHAPE_CHARS = 160


def _shape_of(value: Any, depth: int = 0) -> str:
    """Structure-only description of an artifact (types, never values)."""
    if isinstance(value, dict):
        if depth >= 4:
            return "{...}"
        inner = ", ".join(f"{k}: {_shape_of(v, depth + 1)}" for k, v in value.items())
        return "{" + inner + "}"
    if isinstance(value, list):
        if not value:
            return "[] (empty)"
        return f"[{len(value)} items, each {_shape_of(value[0], depth + 1)}]"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    return "str"


def _inventory(handle: str, steps: list[dict]) -> str:
    """Referenceable occurrences for one tool name, most recent last."""
    occs = [s for s in steps
            if s.get("type") == "action" and s.get("action") == handle
            and s.get("raw_result") is not None]
    if not occs:
        return (f"no captured results for '{handle}' — check the tool name "
                "against the tools you called")
    n = len(occs)
    lines = []
    if n <= _MAX_OCC_SHOWN:
        idxs = range(n)
    else:
        idxs = [0] + list(range(n - (_MAX_OCC_SHOWN - 1), n))
        lines.append(f"  (+{n - _MAX_OCC_SHOWN} earlier occurrences not shown)")
    for i in idxs:
        shape = _shape_of(occs[i]["raw_result"])
        if len(shape) > _MAX_SHAPE_CHARS:
            shape = shape[:_MAX_SHAPE_CHARS] + "…"
        lines.append(f"  {{{{ {handle}[{i}] }}}} — {shape}")
    last = occs[-1]["raw_result"]
    if isinstance(last, list):
        hint = (f"list result: pick an element with an index, e.g. "
                f"{{{{ {handle}[{n - 1}][0].<field> }}}}")
    elif isinstance(last, dict):
        hint = f"object result: read a field, e.g. {{{{ {handle}.{next(iter(last))} }}}}"
    else:
        hint = f"scalar result: reference it whole: {{{{ {handle} }}}}"
    return "\n".join(lines) + f"\n  {hint}"


class EAGPlaceholderMiddleware(AgentMiddleware):
    """Intercept tool calls and substitute placeholders in their arguments."""

    def wrap_tool_call(self, request, handler):
        tool_call: dict = dict(request.tool_call)
        args = tool_call.get("args") or {}

        steps = self._steps_from_state(request.state)
        resolved, errors = resolve_tool_args(args, steps)

        if errors:
            # Fail loudly (but recoverably): show WHICH references exist for
            # each cited tool so the model corrects the path in one retry.
            listed = "; ".join("{{ " + e + " }}" for e in errors)
            handles: list[str] = []
            for e in errors:
                m = _HANDLE_RE.match(e)
                if m:
                    h = m.group(1).split("[")[0]
                    if h not in handles:
                        handles.append(h)
            sections = []
            for h in handles:
                if h.startswith("step"):
                    continue
                sections.append(f"'{h}' occurrences (most recent last):\n"
                                + _inventory(h, steps))
            guide = ("\n" + "\n".join(sections) + "\n") if sections else ""
            return ToolMessage(
                content=(
                    f"[EAG placeholder error] Could not resolve: {listed}. "
                    f"{guide}"
                    "Retry with a corrected placeholder path against the "
                    "structures above. Never replace a placeholder with an "
                    "invented literal value."
                ),
                tool_call_id=tool_call.get("id", ""),
            )

        if resolved != args:
            request = request.override(
                tool_call={**tool_call, "args": resolved}
            )
        return handler(request)

    @staticmethod
    def _steps_from_state(state: Any) -> list[dict]:
        """Build trajectory-like steps (with ``raw_result``) from agent state.

        Walks the message history; every ``ToolMessage`` counts as one step
        (matching the agent's step numbering), and those carrying a non-empty
        artifact expose its parsed ``data`` for placeholder resolution.
        """
        if isinstance(state, dict):
            msgs = state.get("messages", [])
        else:
            msgs = getattr(state, "messages", [])

        steps: list[dict] = []
        n = 0
        for msg in msgs:
            if not isinstance(msg, ToolMessage):
                continue
            n += 1
            artifact = getattr(msg, "artifact", None)
            data: Any = None
            if artifact is not None:
                data = getattr(artifact, "data", None)
                if data is None and getattr(artifact, "columns", None):
                    data = {
                        "columns": list(artifact.columns),
                        "rows": [list(r) for r in artifact.rows],
                    }
            steps.append({
                "step": n,
                "type": "action",
                "action": getattr(msg, "name", "") or "unknown",
                "raw_result": data,
            })
        return steps
