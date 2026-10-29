"""Lindisfarne dataset adapter — QA over multi-format synthetic NHS records.

Corpus: data/lindisfarne/full/records_000{1,2}.jsonl (20,000 records; FHIR
R4 · HL7 v2.5 · SNOMED · GHX procurement). Questions: tasks.json (140-item
seed set, deterministic golds computed by
scripts/generate_lindisfarne_questions.py — see EXPANSION.md).

The agent searches a canonical field vocabulary via find_records, reads raw
per-format JSON via get_record, computes locally with the evaluator tools,
and commits an answer via submit_answer.

Scoring: rule-based match, employees-style, with partial credit:
- numeric golds (including number+unit like "4 years") match the first
  number in the answer (currency/commas stripped) — binary;
- ``true``/``false`` golds match the leading word — binary;
- comma-separated golds match as (normalized) lists, ordered first then as
  sets;
- everything else matches case-insensitively, exactly or as a word-bounded
  substring; leading honorifics (Dr, Mr, Mrs, Ms, Prof, ...) are stripped
  from both sides first so 'Amelia Nair' matches gold 'Ms Amelia Nair';
- verified-empty gold ("no matching records") matches by containment;
- string/list misses fall back to canonical SQuAD F1 (metrics/em_f1.py) as
  the reported ``score`` — ``correct`` stays strict, partial answers earn
  partial credit.

The system prompt is mode-aware: ``make_tools`` records the mode on the
instance (the runner always calls it before reading ``system_prompt``), so
baseline and EAG runs get appropriately-worded instructions without any
runner change.
"""

import json
import re
from pathlib import Path
from typing import Optional

from benchmarks.base import DatasetABC, Task
from benchmarks.registry import register
from metrics.em_f1 import score_answer
from tools.evaluator import make_evaluator_tools
from tools.lindisfarne import make_lindisfarne_tools

_TASKS_PATH = Path(__file__).resolve().parent / "tasks.json"

_COMMON_PROMPT = """\
## Available Tools

**Structure (metadata):**
- get_schema(format?): Canonical field table + corpus overview; with \
format=FHIR|HL7|SNOMED|GHX, the raw record structure of that format.

**Data:**
- find_records(filters, format?, sort_by?, order?, limit?, fields?): \
Search the corpus. filters is a list of {"field", "op", "value"} on \
canonical fields (ops: == != contains starts_with < <= > >=; ISO dates \
compare correctly). ANY-list fields (allergen, med_name, lab_flag, \
contract_ref, ...) match when any item matches; "!=" means no item equals. \
sort_by + order solve oldest/youngest/earliest/latest questions. Returns \
matching record IDs and the total match count.
- get_record(record_id, path?): Fetch one raw record by ID; optional \
dotted path with [N] indices (e.g. "medications[0]", "PID.Address.town") \
trims the display.

**Compute:**
- compare(left, op, right): Boolean comparison of two values (numbers or \
ISO dates).
- compute(a, op, b): Two-number arithmetic (+ - * /).
- count_items(ref): Count a list result.
- filter_items(ref, key, value): Pick matching items from a list result.
- sort_items(ref, key?, order?): Sort a list result (alphabetical names, \
ascending IDs, ...).
- join_items(ref, key?, separator?): Join a list (or one item field, e.g. \
key="name") into one string.

**Submit:**
- submit_answer(answer): Submit your FINAL answer. Do not call other tools \
after submit.

## Workflow
1. Call get_schema() first — it defines the search vocabulary and points \
to per-format structures.
2. Locate records with find_records. Unique keys: name + dob, nhs_number, \
record_id. Read details from raw records with get_record.
3. Count/order/compare locally with the compute tools; submit a short \
final value.

## Answer Conventions
- Names "First Last"; dates YYYY-MM-DD; numbers plain (no separators); \
booleans "true"/"false".
- Name lists: comma-separated, alphabetical. Record-ID lists: \
comma-separated, ascending.
- Contract values exactly as stored (e.g. "£3,074,270").
- If no records match: answer exactly "no matching records" — never guess.
"""

_ROLE = """\
You are a question-answering agent over a corpus of 20,000 synthetic NHS \
records in four formats: FHIR R4, HL7 v2.5, SNOMED, and GHX procurement. \
The same concept is encoded differently per format (patient name is \
patient.name in FHIR, PID.PatientName in HL7, patient_info.full_name in \
SNOMED); find_records works on one canonical vocabulary across formats \
and get_record returns the raw per-format JSON.

"""

_SYSTEM_PROMPT = _ROLE + _COMMON_PROMPT

_EAG_SYSTEM_PROMPT = _ROLE + """\
IMPORTANT: In EAG mode, find_records and get_record disclose ONLY record \
IDs, paths, counts, and types — never values. You work symbolically; the \
harness resolves {{ }} placeholders locally.

""" + _COMMON_PROMPT + """\
## Referencing Values
You never see raw values, so to STATE a result in your submitted answer or \
final text, embed a placeholder — the harness fills it in locally. Syntax:
- {{ find_records.matches[i].<path> }} -> field of match i, navigating the \
record shape shown in the find_records result (lists index as [j], e.g. \
matches[0].allergies[0].allergen; ghx.* for procurement records)
- {{ get_record.<path> }} -> raw field of your MOST RECENT get_record \
result (e.g. get_record.labs[0].performer); use get_record[N].<path> for \
an earlier call (0 = first)
- {{ sort_items.result }} / {{ join_items.result }} / {{ compare.result }} \
-> local compute outcomes

Example — "List the names of patients in Durham diagnosed with X \
(alphabetical)":
  find_records(filters=[{"field": "town", "op": "==", "value": "Durham"},
                        {"field": "diag_label", "op": "==", "value": "X"}])
  sort_items(ref="{{ find_records.matches }}", key="name", order="asc")
  join_items(ref="{{ sort_items.result }}", key="name", separator=", ")
  submit_answer(answer="{{ join_items.result }}")
  final text: "The answer is {{ join_items.result }}."

## Rules
- You CANNOT see values. Reference them symbolically; never invent a \
literal.
- After submit_answer, respond with a brief text answer stating the value \
via placeholder — never end your turn with no text after submit.
"""


@register("lindisfarne")
class LindisfarneDataset(DatasetABC):
    name = "lindisfarne"
    default_max_steps = 16

    _NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
    # Number + unit golds ("4 years", "70 minutes") — scored numerically.
    _UNIT_NUM_RE = re.compile(r"^\s*-?[\d,]+(?:\.\d+)?\s+[A-Za-z]{1,12}\s*$")
    # Leading honorifics stored with clinician/consultant names.
    _HONORIFIC_RE = re.compile(
        r"^(?:(?:dr|mr|mrs|ms|miss|mx|prof|professor|sir|dame|rev|lady|lord)\.?\s+)+",
        re.IGNORECASE,
    )

    def __init__(self):
        self._tasks: Optional[list[Task]] = None
        self._eag_mode = False

    def load(self, limit: Optional[int] = None) -> list[Task]:
        if self._tasks is None:
            if not _TASKS_PATH.exists():
                raise FileNotFoundError(
                    f"Lindisfarne tasks not found at {_TASKS_PATH}. "
                    "See benchmarks/lindisfarne/EXPANSION.md."
                )
            with open(_TASKS_PATH, encoding="utf-8") as f:
                bank = json.load(f)
            self._tasks = [
                Task(
                    id=t["id"],
                    question=t["question"],
                    gold_answer=str(t["gold_answer"]),
                    context=[],
                    metadata={
                        "category": t.get("category", ""),
                        **(t.get("metadata") or {}),
                    },
                )
                for t in bank["tasks"]
            ]
        return self._tasks[:limit] if limit else list(self._tasks)

    def make_tools(self, task: Task, eag_mode: bool = False) -> list:
        # Record mode so system_prompt (read next by the runner) matches.
        self._eag_mode = eag_mode
        tools = make_lindisfarne_tools(eag_mode=eag_mode)
        tools += make_evaluator_tools(eag_mode=eag_mode)
        return tools

    @property
    def system_prompt(self) -> str:
        return _EAG_SYSTEM_PROMPT if self._eag_mode else _SYSTEM_PROMPT

    # ── Scoring ───────────────────────────────────────────────────────────────

    @staticmethod
    def _to_float(s: str) -> Optional[float]:
        s = re.sub(r"[£,%\s]", "", s)
        try:
            return float(s)
        except ValueError:
            return None

    @classmethod
    def _strip_honorifics(cls, s: str) -> str:
        return cls._HONORIFIC_RE.sub("", s.strip()).strip()

    def score(self, trajectory, task: Task) -> dict:
        gold = task.gold_answer.strip()
        answer = (trajectory.answer or "").strip()

        # The agent may end its turn with submit_answer and no final text —
        # the submitted argument is the answer (already placeholder-resolved
        # by the EAG middleware mid-run).
        if not answer:
            for step in reversed(trajectory.steps):
                if step.get("type") == "action" and step.get("action") == "submit_answer":
                    submitted = step.get("action_input", {}).get("answer", "")
                    if submitted:
                        answer = str(submitted).strip()
                        break

        detail = {"category": task.metadata.get("category", "")}
        if not answer:
            return {"correct": False, "score": 0.0, "metric": "lindisfarne", **detail}

        f1_partial = 0.0

        # Verified-empty gold (binary by design — no partial hallucination credit).
        if gold.lower() == "no matching records":
            correct = "no matching records" in answer.lower()
            detail["matched_empty"] = correct
            binary = 1.0 if correct else 0.0
            return {"correct": correct, "score": binary, "em": binary, "f1": binary,
                    "metric": "lindisfarne", **detail}

        gold_num = self._to_float(gold)
        if gold_num is None and self._UNIT_NUM_RE.match(gold):
            m = self._NUM_RE.search(gold)
            gold_num = self._to_float(m.group()) if m else None
        if gold.lower() in ("true", "false"):
            first = answer.rstrip(".!").split()[0].lower() if answer.split() else ""
            correct = first == gold.lower()
            detail["matched_boolean"] = first
        elif gold_num is not None:
            m = self._NUM_RE.search(answer)
            ans_num = self._to_float(m.group()) if m else None
            correct = ans_num is not None and abs(ans_num - gold_num) <= 0.011
            detail["matched_number"] = ans_num
        elif "," in gold:
            # List gold: exact ordered match first, then set-equality fallback.
            def norm(s: str) -> str:
                return self._strip_honorifics(s.strip().casefold().rstrip("."))

            gold_items = [norm(x) for x in gold.split(",") if x.strip()]
            ans_items = [
                norm(x) for x in re.split(r",|\n", answer) if x.strip()
            ]
            ordered = gold_items == ans_items
            as_sets = len(gold_items) == len(ans_items) and set(gold_items) == set(ans_items)
            correct = ordered or as_sets
            detail["matched_list"] = {"ordered": ordered, "as_sets": as_sets}
            if not correct:
                f1_partial = score_answer(answer, gold)["f1"]
        else:
            a = self._strip_honorifics(answer.lower().rstrip("."))
            g = self._strip_honorifics(gold.lower())
            correct = a == g or re.search(rf"\b{re.escape(g)}\b", a) is not None
            detail["matched_string"] = a
            if not correct:
                f1_partial = score_answer(answer, gold)["f1"]

        # Numeric/boolean misses stay binary; string/list misses earn
        # token-overlap F1 as partial credit (correct stays strict).
        f1 = 1.0 if correct else f1_partial
        return {
            "correct": correct,
            "score": f1,
            "em": 1.0 if correct else 0.0,
            "f1": f1,
            "metric": "lindisfarne",
            **detail,
        }
