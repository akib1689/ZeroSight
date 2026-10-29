"""Spider dataset adapter — text-to-SQL with interactive tool use (SQLite only).

Data: data/spider/dev.json + data/spider/database/{db_id}/{db_id}.sqlite

The agent uses 4 SQL tools (schema, sample rows, execute, submit) to answer
natural-language questions over SQLite databases. Spider has no column
descriptions or domain-knowledge evidence, so the metadata tools are omitted.
Scoring: execution equivalence (run submitted SQL vs gold SQL, compare rows).

Two test variants are available via the runner's ``--test-variant`` flag:

  - ``subset`` (default)    — committed 200-question stratified sample
    (``subset.json``): largest-remainder quota per official Spider
    ``eval_hardness`` stratum (easy 48 / medium 86 / hard 34 / extra 32;
    hardness computed from dev.json's parsed SQL), quality-gated, seed
    20260911. Regenerate or verify with ``scripts/make_spider_subset.py``
    (``--check``).
  - ``full``                — the entire 1,034-question dev set.
"""

import json
from pathlib import Path
from typing import Optional

from benchmarks.base import DatasetABC, Task
from benchmarks.registry import register
from metrics.exec_accuracy import score_execution_accuracy
from tools.sql import make_spider_tools

_DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "spider"
_DATA_FILE = _DATA_ROOT / "dev.json"
_DB_ROOT = _DATA_ROOT / "database"
_SUBSET_VARIANT = "subset"
_FULL_VARIANT = "full"
_ALL_VARIANTS = {_FULL_VARIANT, _SUBSET_VARIANT}
_SUBSET_MANIFEST = Path(__file__).resolve().parent / "subset.json"

_SYSTEM_PROMPT = """\
You are a text-to-SQL agent. Your task is to answer a natural language \
question by writing and executing SQL queries against a SQLite database.

## Available Tools

**Exploration:**
- get_schema(): Get all table definitions (column names, types, primary keys, \
foreign keys).
- get_sample_rows(table_name, n_rows): Get sample rows to understand data \
values and patterns.

**Query & Submit:**
- execute_sql(sql_query): Run a SELECT query and see the results. Use for \
testing.
- submit_sql(sql_query): Submit your FINAL SQL for evaluation. Do not call \
other tools after submit.

## Workflow
1. Call get_schema() to understand the database structure.
2. Optionally call get_sample_rows() to understand data patterns.
3. Write your SQL and test it with execute_sql().
4. If results look wrong, revise and re-test.
5. When confident, call submit_sql() with your final query.
6. After submit_sql, respond with a brief answer — do not call more tools.

## Rules
- Only SELECT queries. INSERT/UPDATE/DELETE are not allowed.
- Always test with execute_sql before submitting.
"""

_EAG_SYSTEM_PROMPT = """\
You are a text-to-SQL agent. Your task is to answer a natural language \
question by writing SQL queries against a SQLite database.

IMPORTANT: In EAG mode, execute_sql and get_sample_rows return ONLY column \
names and row counts — NOT actual data values. You must reason about SQL \
correctness from the schema alone.

## Available Tools

**Metadata:**
- get_schema(): Tables, columns, types, primary keys, foreign keys.

**Data (metadata-only — no raw values visible):**
- get_sample_rows(table_name, n_rows): Returns column names + row count only.
- execute_sql(sql_query): Returns column names + row count only. \
Use to verify your query executes without errors.

**Submit:**
- submit_sql(sql_query): Submit your FINAL SQL for evaluation. submit_sql is \
NOT the end of your turn — after it returns you MUST write a closing text \
answer (see "Referencing Results"). Do not call any other tool after submit.

## Workflow
1. Call get_schema() to understand the database structure.
2. Reason about the correct SQL using schema and column names.
3. Call execute_sql() to verify your query runs without errors. \
(You will see column names and row count, but NOT actual values.)
4. If the query errors, fix and re-test.
5. When confident, call submit_sql() with your final query.
6. MANDATORY FINAL ANSWER: After submit_sql returns, you MUST respond with one \
final text message that STATES the result using a placeholder such as \
{{ execute_sql.col.<alias>[0] }} (see below). Never end your turn immediately \
after submit_sql with no text. Never describe only the method — always state \
the concrete value via a placeholder.

## Referencing Results in Your Final Answer
You never see raw values, so to STATE a computed result (a number, a name, a \
year, ...) in your text answer, embed a placeholder. The harness fills it in \
for the end user locally — you will NEVER see the value yourself. This is the \
only way your answer can contain the actual result.

Your final answer must STATE the answer value via a placeholder — do NOT merely \
describe the method or restate the formula. Compare:
- BAD:  "I counted the employees."            (no value stated)
- GOOD: "There are {{ execute_sql.col.n[0] }} employees."  (value stated)

Syntax — use the column name/alias exactly as written in YOUR query:
- {{ execute_sql.col.<column>[R] }}    -> cell in row R (0-based) of your MOST \
RECENT execute_sql result, in column <column>. For single-row results use R=0.
- {{ execute_sql.col.<column> }}       -> whole column as a list.
- {{ execute_sql[0].col.<column>[R] }} -> an earlier execute_sql call \
(0 = first, 1 = second, ...). Bare execute_sql means the most recent one.

Example — question "how many employees are there?":
  final execute_sql:  SELECT COUNT(*) AS n FROM employee
  your text answer:   "There are {{ execute_sql.col.n[0] }} employees."

Important: always run execute_sql with your final query right before submit_sql \
(so "most recent execute_sql" points at the result you want to cite), then \
reference it by the alias you defined. Use placeholders ONLY in your final text \
answer — never inside tool arguments or SQL.

## Rules
- Only SELECT queries. INSERT/UPDATE/DELETE are not allowed.
- You CANNOT see data values. Reason from schema only.
- ALWAYS write a final text answer after submit_sql that states the result via \
a placeholder. Ending with no text after submit_sql is a failure.
"""


@register("spider")
class SpiderDataset(DatasetABC):
    name = "spider"
    default_max_steps = 8

    def __init__(self):
        self._data = None
        self.eag_mode = False
        self._test_variant = _SUBSET_VARIANT
        self._subset_version = None

    def set_test_variant(self, variant: str) -> None:
        """Select the test split: 'subset' (default) or 'full'."""
        if variant not in _ALL_VARIANTS:
            raise ValueError(
                f"Unknown Spider test variant '{variant}'. "
                f"Choose from {sorted(_ALL_VARIANTS)}"
            )
        self._test_variant = variant

    def _load_full_items(self) -> list[dict]:
        if not _DATA_FILE.exists():
            raise FileNotFoundError(
                f"Spider data not found at {_DATA_FILE}. "
                "Download Spider 1.0 or run: uv run python scripts/download_datasets.py"
            )
        with open(_DATA_FILE) as f:
            return json.load(f)

    def _load_subset_items(self) -> list[dict]:
        """Load the committed subset manifest's questions, in manifest order.

        Manifest ids are indices into dev.json (it has no question id); the
        manifest's `items` array supplies the computed difficulty labels.
        """
        if not _SUBSET_MANIFEST.exists():
            raise FileNotFoundError(
                f"Spider subset manifest not found at {_SUBSET_MANIFEST}. "
                "Run: uv run python scripts/make_spider_subset.py"
            )
        manifest = json.loads(_SUBSET_MANIFEST.read_text(encoding="utf-8"))
        items = self._load_full_items()
        ids = manifest["ids"]
        missing = [i for i in ids if not 0 <= i < len(items)]
        if missing:
            raise ValueError(
                f"{len(missing)} subset ids out of range for {_DATA_FILE.name}; "
                "regenerate with: uv run python scripts/make_spider_subset.py"
            )
        by_index = {rec["index"]: rec for rec in manifest.get("items", [])}
        selected = []
        for i in ids:
            item = dict(items[i])
            rec = by_index.get(i)
            if rec:
                item["difficulty"] = rec.get("difficulty", "")
            selected.append(item)
        self._subset_version = manifest.get("version")
        return selected

    def load(self, limit: Optional[int] = None) -> list[Task]:
        if self._test_variant == _SUBSET_VARIANT:
            items = self._load_subset_items()
        else:
            items = self._load_full_items()

        if limit:
            items = items[:limit]

        tasks = []
        for idx, item in enumerate(items):
            db_id = item["db_id"]
            db_path = _DB_ROOT / db_id / f"{db_id}.sqlite"

            if not db_path.exists():
                continue

            metadata = {
                "db_id": db_id,
                "db_path": str(db_path),
                "difficulty": item.get("difficulty", ""),
                "gold_sql": item.get("query", ""),
                "test_variant": self._test_variant,
            }
            if self._test_variant == _SUBSET_VARIANT:
                metadata["subset_version"] = self._subset_version
            tasks.append(
                Task(
                    id=f"spider_{item.get('question_id', idx)}",
                    question=item["question"],
                    gold_answer=item.get("query", ""),
                    context=[],
                    metadata=metadata,
                )
            )

        return tasks

    def make_tools(self, task: Task, eag_mode: bool = False) -> list:
        self.eag_mode = eag_mode
        return make_spider_tools(
            db_path=task.metadata["db_path"],
            eag_mode=eag_mode,
        )

    @property
    def system_prompt(self) -> str:
        return _EAG_SYSTEM_PROMPT if self.eag_mode else _SYSTEM_PROMPT

    def score(self, trajectory, task: Task) -> dict:
        submissions = [
            s for s in trajectory.steps
            if s.get("action") == "submit_sql"
        ]
        if not submissions:
            return {
                "correct": False,
                "score": 0.0,
                "metric": "execution_accuracy",
                "reason": "no_submission",
            }

        final_sql = submissions[-1]["action_input"].get("sql_query", "")
        if not final_sql.strip():
            return {
                "correct": False,
                "score": 0.0,
                "metric": "execution_accuracy",
                "reason": "empty_sql",
            }

        result = score_execution_accuracy(
            submitted_sql=final_sql,
            gold_sql=task.metadata["gold_sql"],
            db_path=task.metadata["db_path"],
        )
        result["submitted_sql"] = final_sql
        result["test_variant"] = task.metadata.get("test_variant", "")
        result["subset_version"] = task.metadata.get("subset_version")
        return result
