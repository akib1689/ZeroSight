"""Canonical SQuAD/HotpotQA EM and F1 metrics.

Normalization order (matches official hotpot_evaluate_v1.py):
    lowercase → remove punctuation → remove articles (a/an/the) → collapse whitespace

EM is exact match after normalization. F1 is token-level precision/recall.
Boolean answers (yes/no/noanswer) are strictly matched — partial F1 is forced
to 0 if one side is boolean and they don't match exactly.

For multiple acceptable answers, the max is taken.

These match the official HotpotQA and 2WikiMultihopQA evaluation scripts:
    https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py
    https://github.com/Alab-NII/2wikimultihop/blob/main/2wikimultihop_evaluate_v1.1.py
"""

import re
import string
import unicodedata
from collections import Counter
from typing import Iterable

_BOOLEAN_ANSWERS = {"yes", "no", "noanswer"}
_PUNCT = set(string.punctuation)


def normalize_answer(text: str) -> str:
    """Normalize a string for comparison.

    Order matches the official HotpotQA script:
        1. Lowercase
        2. Remove punctuation (ASCII + Unicode punctuation categories,
           so en dash "2013–14" matches hyphen "2013-14")
        3. Remove articles (a, an, the)
        4. Collapse whitespace
    """
    text = text.lower().strip()
    # 1. Remove punctuation FIRST (before articles, so word boundaries are intact)
    text = "".join(
        ch for ch in text if ch not in _PUNCT and not unicodedata.category(ch).startswith("P")
    )
    # 2. Remove articles SECOND
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # 3. Collapse whitespace
    text = " ".join(text.split())
    return text


def exact_match(prediction: str, gold: str) -> float:
    """1.0 if normalized prediction equals normalized gold, else 0.0."""
    return float(normalize_answer(prediction) == normalize_answer(gold))


def f1_score(prediction: str, gold: str) -> float:
    """Token-level F1 between normalized prediction and gold.

    Boolean answers (yes/no/noanswer) are strictly matched: if either the
    prediction or gold normalizes to a boolean but they don't match exactly,
    F1 is forced to 0. This matches the official HotpotQA/2WikiMultihopQA
    evaluation scripts.
    """
    normalized_prediction = normalize_answer(prediction)
    normalized_gold = normalize_answer(gold)

    # Strict boolean check — official hotpot_evaluate_v1.py lines 29-33
    if normalized_prediction in _BOOLEAN_ANSWERS and normalized_prediction != normalized_gold:
        return 0.0
    if normalized_gold in _BOOLEAN_ANSWERS and normalized_prediction != normalized_gold:
        return 0.0

    pred_tokens = normalized_prediction.split()
    gold_tokens = normalized_gold.split()

    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def max_over_answers(
    prediction: str,
    gold_answers: Iterable[str],
) -> tuple[float, float]:
    """Max EM and F1 over multiple acceptable gold answers.

    Returns:
        (em, f1) — the best match across all gold answers.
    """
    best_em = 0.0
    best_f1 = 0.0
    for gold in gold_answers:
        em = exact_match(prediction, gold)
        f1 = f1_score(prediction, gold)
        if em > best_em:
            best_em = em
        if f1 > best_f1:
            best_f1 = f1
    return best_em, best_f1


def score_answer(prediction: str, gold: str) -> dict:
    """Score a single prediction against a gold answer.

    Args:
        prediction: Agent's answer string.
        gold: Gold answer string.

    Returns:
        Dict with keys: ``em`` (float 0/1), ``f1`` (float 0-1),
        ``correct`` (bool, True if EM==1).
    """
    em = exact_match(prediction, gold)
    f1 = f1_score(prediction, gold)
    return {
        "em": em,
        "f1": f1,
        "correct": em == 1.0,
    }
