"""Dataset adapter protocol.

Each benchmark (HotpotQA, MuSiQue, FEVER, PrivacyLens, …) implements this
contract. The runner is dataset-agnostic: it drives a uniform per-item loop
and delegates loading, tool construction, gold extraction, and scoring to
the active adapter.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from agent import Trajectory


@dataclass
class Task:
    """A single benchmark item."""
    id: str
    question: str
    gold_answer: str
    context: list[dict]  # paragraphs for retrieval
    metadata: dict[str, Any] = None
    # For benchmarks with pre-built conversation history (PrivacyLens, etc.)
    conversation: Optional[str] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DatasetABC(ABC):
    """Pluggable benchmark dataset.

    Subclasses own:
      - data loading (``load``)
      - per-item tool construction (``make_tools``)
      - system prompt (``system_prompt``)
      - scoring (``score``) — default falls back to EM/F1
    """

    name: str
    default_max_steps: int = 12

    @abstractmethod
    def load(self, limit: Optional[int] = None) -> list[Task]:
        """Load dataset items, optionally truncated to ``limit``."""
        ...

    @abstractmethod
    def make_tools(self, task: Task, eag_mode: bool = False) -> list:
        """Return langchain tools bound to this task's context.

        Args:
            task: The task to build tools for.
            eag_mode: If True, data-returning tools produce metadata-only
                observations (EAG mode). If False (default), tools return
                raw results (baseline mode).
        """
        ...

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """System prompt for this dataset's agent."""
        ...

    def score(self, trajectory: "Trajectory", task: Task) -> dict:
        """Score a trajectory against the gold answer.

        Default implementation uses canonical SQuAD EM/F1.
        Override in subclasses for custom metrics (leakage rate, exact match
        on function calls, task success, etc.).

        Returns:
            dict with keys: ``correct`` (bool), ``score`` (float 0–1),
            ``metric`` (str), and any additional detail keys.
        """
        from metrics.em_f1 import score_answer

        result = score_answer(trajectory.answer, task.gold_answer)
        return {
            "correct": result["correct"],
            "score": result["f1"],
            "metric": "em_f1",
            "em": result["em"],
            "f1": result["f1"],
        }
