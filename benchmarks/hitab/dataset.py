"""HiTab dataset adapter — QA over hierarchical statistical tables.

Data: data/hitab/{test_samples_qualitycheck,test_samples}.jsonl
      data/hitab/tables/hmt/{table_id}.json   (hierarchical matrix tables)

The agent answers natural-language questions by navigating two header trees
(column + row) to locate data cells, reading/aggregating values, and
submitting a final answer. Three test variants are available via the
runner's ``--test-variant`` flag:

  - ``subset`` (default)     — committed 169-question stratified sample
    (``subset.json``): per-label quota ``ceil(10%)`` over the
    ``aggregation`` taxonomy, quality-gated, seed 20260910. Regenerate or
    verify with ``scripts/make_hitab_subset.py`` (``--check``).
  - ``qualitycheck``         — full corrected split (1,584 questions; Oct
    2025 label corrections); the subset's source
  - ``original``             — full split with pre-correction labels, for
    comparability with published numbers

Scoring: canonical EM/F1 with a numeric-equivalence fallback
(metrics/table_qa.py).
"""

import json
from pathlib import Path
from typing import Optional

from benchmarks.base import DatasetABC, Task
from benchmarks.registry import register
from metrics.table_qa import score_table_answer
from tools.hitab import make_hitab_tools

_DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "hitab"
_TEST_VARIANTS = {
    "qualitycheck": "test_samples_qualitycheck.jsonl",
    "original": "test_samples.jsonl",
}
_SUBSET_VARIANT = "subset"
_SUBSET_MANIFEST = Path(__file__).resolve().parent / "subset.json"
_ALL_VARIANTS = {*_TEST_VARIANTS, _SUBSET_VARIANT}

_SYSTEM_PROMPT = """\
You are a question-answering agent working on hierarchical statistical \
tables. Each table has a column-header tree and a row-header tree; every \
tree node annotated with an index labels a data column (top tree) or data \
row (left tree). A node's own row/column often holds the total while its \
children hold the breakdown. A cell is addressed (row, col) in the data \
grid.

## Available Tools

**Structure (metadata):**
- get_table_overview(): Title, both header trees with leaf indices, data \
grid dimensions.
- find_header(query, axis): Find header labels matching a phrase; returns \
the data indices they label. Matching an inner node spans all leaves \
beneath it.

**Data:**
- get_cells(cells): Read cell values by address, e.g. \
[{"row": 3, "col": 0}].
- find_cells(col, op, value): Scan one data column for cells matching a \
condition (==, !=, <, <=, >, >=, contains).
- compare_cells(cells, op, value): Compare specific cells by address, e.g. \
compare_cells(cells=[{"row": 3, "col": 1}, {"row": 3, "col": 2}], \
op="max") — returns the coordinate(s) holding the largest value. op is \
"max"/"min" (no value) or a comparison (==, !=, <, <=, >, >=) against \
value; the matching coordinates are returned. Use it for which-is-larger \
/ which-is-smaller questions.

**Compute:**
- compute(a, op, b): Two-number arithmetic (+ - * /).
- aggregate(op, items): Multi-number aggregation (sum, avg, min, max).

**Submit:**
- submit_answer(answer): Submit your FINAL answer. Do not call other tools \
after submit.

## Workflow
1. Call get_table_overview() to see the header trees.
2. Map question phrases to row/col indices with find_header().
3. Read the cells with get_cells(); use find_cells() for column scans, \
compare_cells() to compare specific cells, and compute/aggregate for \
arithmetic.
4. Call submit_answer() with the final answer (a short value: number, \
name, or yes/no — not a sentence).
5. After submit_answer, respond with a brief text answer — no more tools.

## Rules
- Answer with the exact value(s): "52.1", not "about 52 percent".
- Keep percentages as printed in the table (do not multiply by 100).
"""


_EAG_SYSTEM_PROMPT = """\
You are a question-answering agent working on hierarchical statistical \
tables. Each table has a column-header tree and a row-header tree; every \
tree node annotated with an index labels a data column (top tree) or data \
row (left tree). A node's own row/column often holds the total while its \
children hold the breakdown. A cell is addressed (row, col) in the data \
grid.

IMPORTANT: In EAG mode, get_cells and find_cells return ONLY addresses, \
types, and counts — never values. You locate cells symbolically; the \
harness reads the values locally.

## Available Tools

**Structure (metadata — fully visible):**
- get_table_overview(): Title, both header trees with leaf indices, data \
grid dimensions.
- find_header(query, axis): Find header labels matching a phrase; returns \
the data indices they label.

**Data (addresses + types only — no raw values visible):**
- get_cells(cells): Read cells by address, e.g. [{"row": 3, "col": 0}]. \
Observation lists each cell's type, e.g. (row=3, col=0): <float>.
- find_cells(col, op, value): Scan one data column for cells matching a \
condition. Observations keep matching addresses and the match count.
- compare_cells(cells, op, value): Of 2+ withheld cells, the observation \
names the coordinate(s) holding the extreme (op="max"/"min", e.g. \
"largest: (row=3, col=1)") or satisfying a comparison (==, !=, <, <=, >, \
>= against value — a number or {{ placeholder }}). Map the winning \
address back to its header label with find_header and submit that label \
— this answers which-is-larger questions without any values.

**Local compute (result type only):**
- compute(a, op, b): Two-number arithmetic (+ - * /) over literals or \
{{ placeholder }} references.
- aggregate(op, items): Multi-number aggregation (sum, avg, min, max) over \
literals or {{ placeholder }} references.

**Submit:**
- submit_answer(answer): Submit your FINAL answer. In EAG mode pass a \
{{ placeholder }} reference so the harness fills in the real value. Do not \
call other tools after submit.

## Referencing Values
You never see raw values, so to STATE a result (a number, a name, ...) in \
your submitted answer or final text, embed a placeholder — the harness \
fills it in locally. Syntax:
- {{ get_cells.cells[0].value }}   -> first cell of your MOST RECENT \
get_cells result
- {{ get_cells[1].cells[2].value }} -> an earlier get_cells call \
(0 = first, 1 = second, ...)
- {{ find_cells.cells[R].value }}  -> a matching cell of a find_cells scan
- {{ compute.result }} / {{ aggregate.result }} -> local compute outcomes

Example — question "what percent of french-language workers in eastern \
ontario work in food service?":
  find_header(query="food service", axis="left")  -> row indices [3]
  find_header(query="french-language", axis="top") -> column indices [0]
  get_cells(cells=[{"row": 3, "col": 0}])          -> (row=3, col=0): <float>
  submit_answer(answer="{{ get_cells.cells[0].value }}")
  final text: "The answer is {{ get_cells.cells[0].value }}."

## Workflow
1. get_table_overview() to see the header trees.
2. find_header() to map question phrases onto row/col indices.
3. get_cells() to fetch cells (types only); compare_cells() to find which \
of them is largest/smallest or satisfies a condition; compute/aggregate \
for arithmetic over them via placeholder references.
4. submit_answer() with a placeholder that resolves to the final value.
5. After submit_answer, respond with a brief text answer stating the value \
via placeholder — never end your turn with no text after submit.

## Rules
- You CANNOT see values. Reference them symbolically; never invent a literal.
- Keep percentages as the table stores them (do not scale).
"""


@register("hitab")
class HiTabDataset(DatasetABC):
    name = "hitab"
    default_max_steps = 10

    def __init__(self):
        self._data = None
        self._eag_mode = False
        self._test_variant = _SUBSET_VARIANT
        self._subset_version = None

    def set_test_variant(self, variant: str) -> None:
        """Select the test split: 'subset' (default), 'qualitycheck' or 'original'."""
        if variant not in _ALL_VARIANTS:
            raise ValueError(
                f"Unknown HiTab test variant '{variant}'. "
                f"Choose from {sorted(_ALL_VARIANTS)}"
            )
        self._test_variant = variant

    def _load_subset_items(self) -> list[dict]:
        """Load the committed subset manifest's questions, in manifest order."""
        if not _SUBSET_MANIFEST.exists():
            raise FileNotFoundError(
                f"HiTab subset manifest not found at {_SUBSET_MANIFEST}. "
                "Run: uv run python scripts/make_hitab_subset.py"
            )
        manifest = json.loads(_SUBSET_MANIFEST.read_text(encoding="utf-8"))
        source = manifest.get("source_variant", "qualitycheck")
        if source not in _TEST_VARIANTS:
            raise ValueError(
                f"Subset manifest references unknown source variant '{source}'"
            )
        data_file = _DATA_ROOT / _TEST_VARIANTS[source]
        if not data_file.exists():
            raise FileNotFoundError(
                f"HiTab data not found at {data_file}. "
                "Run: uv run python scripts/download_datasets.py"
            )
        by_id = {}
        with open(data_file, encoding="utf-8") as f:
            for line in f:
                s = json.loads(line)
                if not s.get("question"):
                    continue  # data2text-only records
                by_id[s["id"]] = s
        ids = manifest["ids"]
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise ValueError(
                f"{len(missing)} subset ids missing from {data_file.name}; "
                "regenerate with: uv run python scripts/make_hitab_subset.py"
            )
        self._subset_version = manifest.get("version")
        return [by_id[i] for i in ids]

    def load(self, limit: Optional[int] = None) -> list[Task]:
        if self._test_variant == _SUBSET_VARIANT:
            items = self._load_subset_items()
        else:
            data_file = _DATA_ROOT / _TEST_VARIANTS[self._test_variant]
            if not data_file.exists():
                raise FileNotFoundError(
                    f"HiTab data not found at {data_file}. "
                    "Run: uv run python scripts/download_datasets.py"
                )
            items = []
            with open(data_file, encoding="utf-8") as f:
                for line in f:
                    s = json.loads(line)
                    if not s.get("question"):
                        continue  # data2text-only records
                    items.append(s)
        if limit:
            items = items[:limit]

        tasks = []
        for s in items:
            table_path = _DATA_ROOT / "tables" / "hmt" / f"{s['table_id']}.json"
            if not table_path.exists():
                continue
            answers = s.get("answer") or []
            metadata = {
                "table_id": s["table_id"],
                "table_source": s.get("table_source", ""),
                "table_path": str(table_path),
                "aggregation": ",".join(s.get("aggregation") or []),
                "answer_formulas": s.get("answer_formulas") or [],
                "gold_answers": answers,
                "test_variant": self._test_variant,
            }
            if self._test_variant == _SUBSET_VARIANT:
                metadata["subset_version"] = self._subset_version
            tasks.append(
                Task(
                    id=f"hitab_{s['id']}",
                    question=s["question"],
                    gold_answer=", ".join(str(a) for a in answers),
                    context=[],
                    metadata=metadata,
                )
            )
        return tasks

    def make_tools(self, task: Task, eag_mode: bool = False) -> list:
        self._eag_mode = eag_mode
        return make_hitab_tools(
            table_path=task.metadata["table_path"],
            eag_mode=eag_mode,
        )

    @property
    def system_prompt(self) -> str:
        return _EAG_SYSTEM_PROMPT if self._eag_mode else _SYSTEM_PROMPT

    def score(self, trajectory, task: Task) -> dict:
        # agent.py gives submit_answer args precedence over final text and
        # resolves {{ }} placeholders post-run, so trajectory.answer is the
        # committed, resolved answer.
        answer = (trajectory.answer or "").strip()
        result = score_table_answer(answer, task.metadata.get("gold_answers") or [])

        submitted = None
        for step in reversed(trajectory.steps):
            if step.get("type") == "action" and step.get("action") == "submit_answer":
                submitted = step.get("action_input", {}).get("answer", "")
                break
        detail = {
            "test_variant": task.metadata.get("test_variant", ""),
            "subset_version": task.metadata.get("subset_version"),
            "aggregation": task.metadata.get("aggregation", ""),
            "submitted": submitted,
        }
        return {
            "correct": result["correct"],
            "score": result["score"],
            "metric": result["metric"],
            "em": result["em"],
            "f1": result["f1"],
            "numeric_match": result["numeric_match"],
            **detail,
        }
