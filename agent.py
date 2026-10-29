"""ReAct agent — thin wrapper over langchain's ``create_agent``.

The agent is model-agnostic (any ChatLiteLLM) and dataset-agnostic (tools are
passed per-task by the dataset adapter). It runs the canonical ReAct loop
internally via langchain/langgraph and returns a ``Trajectory`` with the answer,
steps, usage, and latency.

ZeroSight vs baseline differ only in which tools are bound (value-free vs raw
via tool hooks) — same agent class, same loop.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from tools.eag_middleware import EAGPlaceholderMiddleware
from tools.templating import substitute_placeholders

try:
    import litellm

    def _count_tokens(text: str) -> int:
        """Token-count an observation for exposure measurement."""
        if not text:
            return 0
        try:
            return litellm.token_counter(text=text)
        except Exception:
            return len(text) // 4
except ImportError:
    def _count_tokens(text: str) -> int:
        return len(text) // 4

try:
    from langgraph.errors import GraphRecursionError
except ImportError:
    GraphRecursionError = type(None)

logger = logging.getLogger(__name__)


def _extract_text(content: Any) -> str:
    """Extract plain text from message content.

    Handles both plain strings and structured content blocks from reasoning
    models: ``[{'type': 'thinking', ...}, {'type': 'text', 'text': '...'}]``.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return " ".join(parts)
    return str(content)


@dataclass
class Trajectory:
    """Agent execution result — the universal output contract."""
    answer: str = ""
    answer_raw: str = ""
    unresolved_placeholders: int = 0
    steps: list[dict] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    n_turns: int = 0
    n_tool_calls: int = 0
    usage: dict = field(default_factory=lambda: {
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0
    })
    latency_ms: float = 0.0
    error: str | None = None
    truncated: bool = False


class ReactAgent:
    """ReAct agent backed by langchain's create_agent.

    Args:
        model: A langchain BaseChatModel (e.g. ChatLiteLLM).
        tools: List of BaseTool instances.
        system_prompt: System message for the agent.
        max_steps: Max ReAct turns (each turn = one agent call + one tool
            execution). Internally doubled for langgraph's recursion_limit,
            which counts graph node visits (2 per turn).
    """

    def __init__(
        self,
        model,
        tools: list,
        system_prompt: str,
        max_steps: int = 12,
        eag_mode: bool = False,
    ):
        self.model = model
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.eag_mode = eag_mode
        def _effective_exposure(tool) -> str:
            if hasattr(tool, "effective_exposure_type"):
                return tool.effective_exposure_type()
            return getattr(tool, "exposure_type", "unknown")

        self._tool_types: dict[str, str] = {t.name: _effective_exposure(t) for t in tools}

        # In EAG mode, intercept tool calls to resolve {{ }} placeholders in
        # arguments against prior artifacts (referential chain stays connected
        # without exposing raw values). Baseline runs are untouched.
        self._middleware = [EAGPlaceholderMiddleware()] if eag_mode else []

    def run(self, question: str, trace: bool = False) -> Trajectory:
        """Run the agent on a single question."""
        start = time.perf_counter()
        traj = Trajectory()

        try:
            agent = create_agent(
                model=self.model,
                tools=self.tools,
                system_prompt=self.system_prompt,
                middleware=self._middleware,
            )

            # FIX 1: recursion_limit counts node visits (2 per ReAct turn:
            # agent node + tool node). max_steps * 2 yields max_steps turns.
            config = {"recursion_limit": self.max_steps * 2}

            # FIX 3: stream to preserve partial trajectory on recursion error.
            messages = []
            try:
                for chunk in agent.stream(
                    {"messages": [HumanMessage(content=question)]},
                    config=config,
                    stream_mode="values",
                ):
                    messages = chunk.get("messages", messages)
            except GraphRecursionError:
                traj.truncated = True
                if trace:
                    logger.info("  [TRUNCATED: hit step limit]")

            self._extract_result(traj, messages, trace)
            self._resolve_answer_placeholders(traj, trace)

        except Exception as exc:
            traj.error = str(exc)
            logger.exception("Agent execution failed")

        traj.latency_ms = (time.perf_counter() - start) * 1000
        return traj

    def _extract_result(self, traj: Trajectory, messages: list, trace: bool) -> None:
        """Extract answer, steps, and usage from the message trace.

        FIX 2: observations matched to actions by tool_call_id (not order).
        FIX 4: per-step context size + raw observation size recorded.
        FIX 5: exact turn counts — one turn per AIMessage (agent call, what
        max_steps caps); n_tool_calls counts individual tool calls. A
        thinking-only final AIMessage with no text and no tool calls leaves
        no step entry but still counts as a turn.
        """
        total_prompt = 0
        total_completion = 0
        total_tokens = 0
        step_num = 0
        turns = 0
        tool_call_count = 0

        pending_by_id: dict[str, dict] = {}

        for msg in messages:
            msg_type = msg.__class__.__name__
            raw_content = msg.content if hasattr(msg, "content") else str(msg)
            text = _extract_text(raw_content)

            if isinstance(msg, SystemMessage):
                continue

            if isinstance(msg, AIMessage):
                turns += 1
                usage = getattr(msg, "usage_metadata", None)
                step_usage = {}
                if usage:
                    step_usage = {
                        "input_tokens": usage.get("input_tokens", 0),
                        "output_tokens": usage.get("output_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                    }
                    total_prompt += step_usage["input_tokens"]
                    total_completion += step_usage["output_tokens"]
                    total_tokens += step_usage["total_tokens"]

                # Step thought: the model's visible commentary, falling back
                # to the provider's hidden reasoning stream (e.g. zai
                # ``reasoning_content``) when the model emits no commentary.
                # Observability only — answers always come from text blocks.
                thought = text
                if not thought.strip():
                    rc = (getattr(msg, "additional_kwargs", None) or {}).get(
                        "reasoning_content"
                    )
                    if isinstance(rc, str) and rc.strip():
                        thought = rc.strip()

                tool_calls = getattr(msg, "tool_calls", None)

                if tool_calls:
                    tool_call_count += len(tool_calls)
                    for tc in tool_calls:
                        step_num += 1
                        tc_id = tc.get("id", "")
                        step = {
                            "step": step_num,
                            "type": "action",
                            "thought": thought,
                            "action": tc.get("name", "unknown"),
                            "action_input": tc.get("args", {}),
                            "tool_call_id": tc_id,
                            "tool_type": self._tool_types.get(
                                tc.get("name", ""), "unknown"
                            ),
                            "input_tokens": step_usage.get("input_tokens", 0),
                            "output_tokens": step_usage.get("output_tokens", 0),
                            "observation": "",
                            "observation_chars": 0,
                            "observation_tokens": 0,
                        }
                        traj.steps.append(step)
                        pending_by_id[tc_id] = step
                        if trace:
                            logger.info(
                                f"  Step {step_num}: {step['action']}({step['action_input']})"
                            )
                elif text.strip():
                    traj.answer = text.strip()
                    traj.steps.append({
                        "step": step_num + 1,
                        "type": "answer",
                        "content": text.strip(),
                        "input_tokens": step_usage.get("input_tokens", 0),
                        "output_tokens": step_usage.get("output_tokens", 0),
                    })
                    if trace:
                        logger.info(f"  Answer: {text.strip()[:100]}")

            elif isinstance(msg, ToolMessage):
                observation = text if text else ""

                # FIX 2: match observation to action by tool_call_id
                tc_id = getattr(msg, "tool_call_id", "")
                step = pending_by_id.get(tc_id)
                if step is None:
                    for s in reversed(traj.steps):
                        if s.get("type") == "action" and not s.get("observation"):
                            step = s
                            break
                if step is not None:
                    step["observation"] = observation
                    step["observation_chars"] = len(observation)
                    step["observation_tokens"] = _count_tokens(observation)
                    # Stash the raw result artifact (local-only, never shown to
                    # the model) for deferred placeholder substitution. The
                    # artifact is the full ToolResult; its parsed ``data`` (or
                    # tabular columns/rows fallback) is navigable by the
                    # unified placeholder path grammar. In EAG mode the model
                    # only saw the redacted `observation`.
                    artifact = getattr(msg, "artifact", None)
                    if artifact is not None:
                        data = getattr(artifact, "data", None)
                        if data is None and getattr(artifact, "columns", None):
                            data = {
                                "columns": list(artifact.columns),
                                "rows": [list(r) for r in artifact.rows],
                            }
                        if data is not None:
                            step["raw_result"] = data
                if trace:
                    obs_preview = observation[:200] if observation else ""
                    logger.info(f"    -> {obs_preview}")

            traj.messages.append({
                "type": msg_type,
                "content": text,
            })

        traj.usage = {
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": total_tokens,
        }
        traj.n_turns = turns
        traj.n_tool_calls = tool_call_count

    def _resolve_answer_placeholders(self, traj: Trajectory, trace: bool) -> None:
        """Resolve ``{{ ... }}`` placeholders in the final answer post-run.

        The model embeds symbolic references (e.g. ``{{ execute_sql.col.ratio[0] }}``)
        because it never saw the raw values. We bind them here against the
        locally-captured ``raw_result`` artifacts, then strip those artifacts so
        no raw rows leak into the serialized trajectory. The verbatim model
        output is preserved as ``answer_raw`` for fidelity analysis.

        Agents that end their turn with ``submit_answer`` and no final text
        are handled the same way: the submitted argument *is* the answer, so
        its placeholders are resolved here too (the middleware only resolved
        what the tool needed to execute, not the model's verbatim arg).
        A ``submit_answer`` commit takes precedence over any final text:
        it is the explicit evaluation act, and in EAG mode trailing text is
        sometimes a hallucinated literal the model could not verify.
        """
        committed = None
        for step in reversed(traj.steps):
            if (
                step.get("type") == "action"
                and step.get("action") == "submit_answer"
            ):
                submitted = step.get("action_input", {}).get("answer", "")
                if submitted:
                    committed = (
                        submitted if isinstance(submitted, str) else str(submitted)
                    )
                    break
        if committed is not None:
            traj.answer = committed
        traj.answer_raw = traj.answer
        if traj.answer:
            resolved, unresolved = substitute_placeholders(traj.answer, traj.steps)
            traj.answer = resolved
            traj.unresolved_placeholders = unresolved
            if trace and unresolved:
                logger.info(
                    f"  [placeholders: {unresolved} unresolved]"
                )
        # raw_result is in-memory only — drop it before the trajectory leaves
        # so raw rows are never serialized to the results JSON.
        for step in traj.steps:
            step.pop("raw_result", None)
