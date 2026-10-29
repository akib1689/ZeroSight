"""Table-QA scoring — canonical EM/F1 with numeric-equivalence fallback.

HiTab answers are short cell values or arithmetic results: floats (52.1),
ints (95), or strings, often verbalized with formatting the table did not
use ("52.1%", "$1,234", "about 52.1 percent"). Plain SQuAD EM/F1 under- or
over-scores such formatting differences, so scoring runs:

    1. canonical EM/F1 against every acceptable gold rendering
       (the full comma-joined list and each individual gold value);
    2. a numeric fallback — if any gold value and any number in the
       prediction parse to the same float (after stripping commas, currency
       signs, surrounding text, and an optional % suffix), the answer counts
       as correct with score 1.0.

`correct` is EM or numeric match; `score` is max(F1, numeric match).
"""

import re
from typing import Iterable

from metrics.em_f1 import exact_match, f1_score

# First number in a string, tolerating $/€/£ signs, thousands separators,
# and an optional surrounding-word % ("52.1 percent", "(52.1%)").
_NUM_RE = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?")


def _to_float(text: str) -> float | None:
    s = str(text).strip().replace(",", "").replace("$", "")
    try:
        return float(s)
    except ValueError:
        return None


def _numbers_in(text: str) -> list[float]:
    out = []
    for m in _NUM_RE.finditer(str(text)):
        v = _to_float(m.group())
        if v is not None:
            out.append(v)
    return out


def score_table_answer(prediction: str, gold_answers: Iterable) -> dict:
    """Score a table-QA prediction against acceptable gold values.

    Args:
        prediction: the (placeholder-resolved) final answer string.
        gold_answers: acceptable gold values — a list (HiTab ``answer``
            field) or a single scalar. The comma-joined rendering counts as
            an additional acceptable string answer.

    Returns:
        dict: ``correct``, ``score``, ``em``, ``f1``, ``numeric_match``,
        ``metric="table_em_f1"``.
    """
    if not isinstance(gold_answers, (list, tuple)):
        gold_answers = [gold_answers]

    pred = (prediction or "").strip()
    gold_strs: list[str] = []
    gold_nums: list[float] = []

    joined = ", ".join(str(g) for g in gold_answers)
    if joined:
        gold_strs.append(joined)
    for g in gold_answers:
        gold_strs.append(str(g))
        gold_nums.extend(_numbers_in(g))

    if not pred or not gold_strs:
        return {
            "correct": False,
            "score": 0.0,
            "em": 0.0,
            "f1": 0.0,
            "numeric_match": False,
            "metric": "table_em_f1",
        }

    em = max(exact_match(pred, g) for g in gold_strs)
    f1 = max(f1_score(pred, g) for g in gold_strs)

    numeric_match = False
    if gold_nums:
        pred_nums = _numbers_in(pred)
        numeric_match = any(
            abs(p - g) <= 1e-9 * max(1.0, abs(g))
            for p in pred_nums
            for g in gold_nums
        )

    correct = bool(em or numeric_match)
    return {
        "correct": correct,
        "score": max(f1, 1.0 if numeric_match else 0.0),
        "em": em,
        "f1": f1,
        "numeric_match": numeric_match,
        "metric": "table_em_f1",
    }
