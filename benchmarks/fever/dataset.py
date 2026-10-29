"""FEVER dataset adapter — fact verification via BM25 search/lookup.

Data: data/fever/ (HF dataset from copenlu/fever_gold_evidence, validation split)
Format: claim + evidence sentences + label (SUPPORTS/REFUTES/NOT_ENOUGH INFO)

The agent searches evidence paragraphs, then outputs its verdict.
"""

from pathlib import Path
from typing import Optional

from benchmarks.base import DatasetABC, Task
from benchmarks.registry import register
from tools.qa import make_qa_tools

_DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "fever"

_SYSTEM_PROMPT = """\
You are a fact verification agent. You must determine whether a claim is \
SUPPORTED, REFUTED, or if there is NOT ENOUGH INFO based on the available \
evidence.

You have two tools:
- search(query): Find relevant paragraphs from the evidence corpus. Returns \
top-3 matching paragraphs.
- lookup(term): Find a specific term in the evidence. Returns the sentence \
containing the term.

After examining the evidence, respond with exactly one of:
- SUPPORTS (if the evidence supports the claim)
- REFUTES (if the evidence contradicts the claim)
- NOT ENOUGH INFO (if the evidence is insufficient to verify)

Your final answer must be exactly one of these three labels."""


@register("fever")
class FEVERDataset(DatasetABC):
    name = "fever"

    def __init__(self):
        self._data = None

    def load(self, limit: Optional[int] = None) -> list[Task]:
        if self._data is None:
            if not _DATA_PATH.exists():
                raise FileNotFoundError(
                    f"FEVER data not found at {_DATA_PATH}. "
                    "Run: uv run python scripts/download_datasets.py"
                )
            from datasets import load_from_disk

            self._data = load_from_disk(str(_DATA_PATH))

        items = self._data
        if limit:
            items = items.select(range(min(limit, len(items))))

        tasks = []
        for item in items:
            claim = item["claim"]
            label = item["label"]
            evidence = item.get("evidence", [])

            # Build context paragraphs from evidence
            # Evidence format: [[wiki_page_title, sentence_id, sentence_text], ...]
            title_to_sentences: dict[str, list[str]] = {}
            for ev in evidence:
                if isinstance(ev, list) and len(ev) >= 3:
                    title, _sid, sentence = ev[0], ev[1], ev[2]
                    if title not in title_to_sentences:
                        title_to_sentences[title] = []
                    title_to_sentences[title].append(sentence)

            context = [
                {"title": title, "sentences": sents}
                for title, sents in title_to_sentences.items()
            ]

            # Skip items with no evidence context
            if not context:
                continue

            tasks.append(
                Task(
                    id=str(item.get("id", item.get("original_id", ""))),
                    question=f"Verify this claim: \"{claim}\"",
                    gold_answer=label,
                    context=context,
                    metadata={"claim": claim, "label": label},
                )
            )

        return tasks

    def make_tools(self, task: Task, eag_mode: bool = False) -> list:
        return make_qa_tools(task.context)

    @property
    def system_prompt(self) -> str:
        return _SYSTEM_PROMPT

    def score(self, trajectory, task: Task) -> dict:
        answer = trajectory.answer.strip().upper() if trajectory.answer else ""
        gold = task.gold_answer.strip().upper()
        # Normalize label matching
        if "SUPPORT" in answer and "SUPPORT" in gold:
            correct = True
        elif "REFUTE" in answer and "REFUTE" in gold:
            correct = True
        elif "NOT ENOUGH" in answer and "NOT ENOUGH" in gold:
            correct = True
        else:
            correct = False
        return {
            "correct": correct,
            "score": 1.0 if correct else 0.0,
            "metric": "label_accuracy",
            "predicted": answer,
            "gold": gold,
        }
