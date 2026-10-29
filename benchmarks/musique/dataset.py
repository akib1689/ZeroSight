"""MuSiQue adapter.

Loads from local ``data/musique/`` (HF Arrow format via ``load_from_disk``).
Run ``scripts/download_datasets.py`` once to fetch. Each item has 2-20 context
paragraphs. Same tool surface as HotpotQA (BM25 search/lookup). Constructed to
require genuine multi-hop reasoning — single-shot retrieval drops 30 F1.
"""

from pathlib import Path
from typing import Optional

from benchmarks.base import DatasetABC, Task
from benchmarks.registry import register
from tools.qa import make_qa_tools

_DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "musique"

_SYSTEM_PROMPT = """\
You are a multi-hop question answering agent. Answer the question by searching \
and looking up information in the provided context paragraphs.

## Tools
- search(query): Search context paragraphs for relevant information. Returns \
top matching paragraphs with titles.
- lookup(term): Find a specific term in the context paragraphs. Returns \
surrounding text.

## Rules
- Use search first to find relevant paragraphs.
- Use lookup to find specific details within paragraphs.
- These questions require 2-4 reasoning hops — you MUST combine information \
from MULTIPLE paragraphs. A single search will NOT be enough.
- When you know the answer, respond with ONLY the answer value — no tool calls.

## Answer Format
The answer must be ONLY the answer — a single word, name, number, or short \
phrase with no explanation, no full sentences, and no extra context.

Correct format: "yes", "no", "32", "Albert Einstein", "Paris"
Wrong format: "Yes, that is correct", "The answer is Paris because...", \
"Albert Einstein was born in 1879"

Answer with ONLY the fact. Nothing else.
"""


@register("musique")
class MuSiQueDataset(DatasetABC):
    name = "musique"
    default_max_steps = 20

    def __init__(self):
        self._data = None

    def load(self, limit: Optional[int] = None) -> list[Task]:
        if not _DATA_PATH.exists():
            raise FileNotFoundError(
                f"MuSiQue data not found at {_DATA_PATH}. "
                "Run: uv run python scripts/download_datasets.py"
            )

        from datasets import load_from_disk

        ds = load_from_disk(str(_DATA_PATH))
        if limit:
            ds = ds.select(range(min(limit, len(ds))))

        tasks = []
        for item in ds:
            paragraphs = []
            for para in item.get("paragraphs", []):
                paragraphs.append({
                    "title": para.get("title", ""),
                    "paragraph_text": para.get("paragraph_text", ""),
                })

            tasks.append(Task(
                id=item.get("id", ""),
                question=item["question"],
                gold_answer=item["answer"],
                context=paragraphs,
                metadata={"n_paragraphs": len(paragraphs)},
            ))

        return tasks

    def make_tools(self, task: Task, eag_mode: bool = False) -> list:
        return make_qa_tools(task.context)

    @property
    def system_prompt(self) -> str:
        return _SYSTEM_PROMPT
