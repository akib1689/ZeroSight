"""2WikiMultihopQA dataset adapter — multi-hop QA via BM25 search/lookup.

Data: data/2wiki_raw/data_ids/dev.json (official version with answer_id)
       data/2wiki_raw/data_ids/id_aliases.json (Wikidata aliases + demonyms)

Uses the official v1.1 evaluation: for each answer, all Wikidata aliases
and demonyms are acceptable. Scoring uses max_over_answers() across the
full alias set — same as 2wikimultihop_evaluate_v1.1.py.
"""

import json
from pathlib import Path
from typing import Optional

from benchmarks.base import DatasetABC, Task
from benchmarks.registry import register
from metrics.em_f1 import max_over_answers, normalize_answer
from tools.qa import make_qa_tools

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "2wiki_raw" / "data_ids"
_DEV_PATH = _DATA_DIR / "dev.json"
_ALIAS_PATH = _DATA_DIR / "id_aliases.json"

_SYSTEM_PROMPT = """\
You are a multi-hop question answering agent. Answer the user's question by \
searching for relevant information across multiple paragraphs and combining \
evidence.

You have two tools:
- search(query): Search paragraphs by relevance. Returns top-3 matches.
- lookup(term): Find a specific term in the context. Returns the sentence \
containing the term.

You may need to search multiple times with different queries to gather enough \
evidence. Combine information from multiple sources to answer the question.

When you have enough evidence, provide your final answer. The answer must be \
ONLY the answer — a single word, name, number, or short phrase with no \
explanation, no full sentences, and no extra context.

Correct format: "yes", "no", "32", "Albert Einstein", "Paris"
Wrong format: "Yes, that is correct", "The answer is Paris because...", \
"He was born in 1879"

Answer with ONLY the fact. Nothing else."""


def _load_aliases() -> dict[str, set[str]]:
    """Load id_aliases.json → {Q_id: {all normalized aliases + demonyms}}."""
    if not _ALIAS_PATH.exists():
        return {}
    aliases: dict[str, set[str]] = {}
    with open(_ALIAS_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            qid = entry.get("Q_id", "")
            if not qid:
                continue
            names = set()
            for a in entry.get("aliases", []):
                n = normalize_answer(a)
                if n:
                    names.add(n)
            for d in entry.get("demonyms", []):
                n = normalize_answer(d)
                if n:
                    names.add(n)
            aliases[qid] = names
    return aliases


def _build_reverse_alias_map(
    alias_map: dict[str, set[str]],
) -> dict[str, set[str]]:
    """Build reverse lookup: any normalized string → all equivalents."""
    reverse: dict[str, set[str]] = {}
    for qid, names in alias_map.items():
        for name in names:
            reverse.setdefault(name, set()).update(names)
    return reverse


@register("wiki2")
class Wiki2Dataset(DatasetABC):
    name = "wiki2"
    default_max_steps = 16

    def __init__(self):
        self._data = None
        self._alias_map: dict[str, set[str]] | None = None

    def _ensure_loaded(self):
        if self._data is not None:
            return

        if not _DEV_PATH.exists():
            raise FileNotFoundError(
                f"2WikiMultihopQA dev.json not found at {_DEV_PATH}. "
                "The official data with answer_id is required. "
                "Download: https://www.dropbox.com/s/7ep3h8unu2njfxv/data_ids.zip"
            )

        with open(_DEV_PATH) as f:
            self._data = json.load(f)
        self._alias_map = _load_aliases()

    def load(self, limit: Optional[int] = None) -> list[Task]:
        self._ensure_loaded()
        items = self._data
        if limit:
            items = items[:limit]

        tasks = []
        for item in items:
            question = item["question"]
            answer = item["answer"]
            raw_context = item.get("context", [])
            answer_id = item.get("answer_id", "")

            # Build context paragraphs
            context = []
            for ctx_item in raw_context:
                if isinstance(ctx_item, (list, tuple)) and len(ctx_item) >= 2:
                    title = ctx_item[0]
                    sentences = (
                        ctx_item[1]
                        if isinstance(ctx_item[1], list)
                        else [ctx_item[1]]
                    )
                    context.append({"title": title, "sentences": sentences})

            if not context:
                continue

            # Collect acceptable answers: gold + all aliases/demonyms
            gold_answers = {answer}
            if answer_id and self._alias_map and answer_id in self._alias_map:
                # Add normalized alias forms
                for alias_norm in self._alias_map[answer_id]:
                    gold_answers.add(alias_norm)

            tasks.append(
                Task(
                    id=item.get("_id", ""),
                    question=question,
                    gold_answer=answer,
                    context=context,
                    metadata={
                        "type": item.get("type", ""),
                        "answer_id": answer_id,
                        "gold_answers": list(gold_answers),
                    },
                )
            )

        return tasks

    def make_tools(self, task: Task, eag_mode: bool = False) -> list:
        return make_qa_tools(task.context)

    @property
    def system_prompt(self) -> str:
        return _SYSTEM_PROMPT

    def score(self, trajectory, task: Task) -> dict:
        gold_answers = task.metadata.get("gold_answers", [task.gold_answer])
        prediction = trajectory.answer or ""

        em, f1 = max_over_answers(prediction, gold_answers)

        return {
            "correct": em == 1.0,
            "score": f1,
            "metric": "em_f1",
            "em": em,
            "f1": f1,
            "n_gold_answers": len(gold_answers),
        }
