#!/usr/bin/env python3
"""Generate the committed Spider `subset` manifest (benchmarks/spider/subset.json).

Selection spec (frozen, seed 20260911):
  - Source: data/spider/dev.json (1,034 questions)
  - Difficulty: dev.json ships no labels; strata come from the official Spider
    hardness function (taoyds/spider evaluation.py::Evaluator.eval_hardness)
    computed over each item's parsed `sql` field
    -> easy 248 / medium 446 / hard 174 / extra 166
  - Budget: exactly 200 questions, apportioned across hardness strata by
    largest remainder (proportional to source counts)
    -> easy 48 / medium 86 / hard 34 / extra 32
  - Strata sampled rare-first (extra, hard, easy, medium)
  - Gates: db sqlite present, non-empty question and gold SQL, parsed sql
    present, unique normalized question text, max
    ceil(200 * db count / 1034) + 3 questions per database, short questions
    preferred

Manifest `ids` are indices into dev.json (it has no question id); the `items`
array carries (db_id, difficulty, question) for auditing.

Usage:
    uv run python scripts/make_spider_subset.py           # (re)write manifest
    uv run python scripts/make_spider_subset.py --check   # determinism guard
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_FILE = REPO_ROOT / "data" / "spider" / "dev.json"
DB_ROOT = REPO_ROOT / "data" / "spider" / "database"
MANIFEST_FILE = REPO_ROOT / "benchmarks" / "spider" / "subset.json"

SEED = 20260911
BUDGET = 200
DB_SLACK = 3
MAX_QUESTION_LEN = 150

_WHERE_OPS = ("not", "between", "=", ">", "<", ">=", "<=", "!=", "in", "like", "is", "exists")
_AGG_OPS = ("none", "max", "min", "count", "sum", "avg")


def _nested(sql: dict) -> list:
    nested = []
    for cond_unit in sql["from"]["conds"][::2] + sql["where"][::2] + sql["having"][::2]:
        if type(cond_unit[3]) is dict:
            nested.append(cond_unit[3])
        if type(cond_unit[4]) is dict:
            nested.append(cond_unit[4])
    if sql["intersect"] is not None:
        nested.append(sql["intersect"])
    if sql["except"] is not None:
        nested.append(sql["except"])
    if sql["union"] is not None:
        nested.append(sql["union"])
    return nested


def _count_agg(units: list) -> int:
    none_agg = _AGG_OPS.index("none")
    return len([unit for unit in units if unit[0] != none_agg])


def _count_component1(sql: dict) -> int:
    count = 0
    if len(sql["where"]) > 0:
        count += 1
    if len(sql["groupBy"]) > 0:
        count += 1
    if len(sql["orderBy"]) > 0:
        count += 1
    if sql["limit"] is not None:
        count += 1
    if len(sql["from"]["table_units"]) > 0:
        count += len(sql["from"]["table_units"]) - 1

    ao = sql["from"]["conds"][1::2] + sql["where"][1::2] + sql["having"][1::2]
    count += len([token for token in ao if token == "or"])
    cond_units = sql["from"]["conds"][::2] + sql["where"][::2] + sql["having"][::2]
    count += len(
        [cond_unit for cond_unit in cond_units if cond_unit[1] == _WHERE_OPS.index("like")]
    )
    return count


def _count_component2(sql: dict) -> int:
    return len(_nested(sql))


def _count_others(sql: dict) -> int:
    count = 0
    agg_count = _count_agg(sql["select"][1])
    agg_count += _count_agg(sql["where"][::2])
    agg_count += _count_agg(sql["groupBy"])
    if len(sql["orderBy"]) > 0:
        agg_count += _count_agg(
            [unit[1] for unit in sql["orderBy"][1] if unit[1]]
            + [unit[2] for unit in sql["orderBy"][1] if unit[2]]
        )
    agg_count += _count_agg(sql["having"])
    if agg_count > 1:
        count += 1

    if len(sql["select"][1]) > 1:
        count += 1
    if len(sql["where"]) > 1:
        count += 1
    if len(sql["groupBy"]) > 1:
        count += 1
    return count


def eval_hardness(sql: dict) -> str:
    """Official Spider hardness (taoyds/spider evaluation.py::Evaluator.eval_hardness)."""
    comp1 = _count_component1(sql)
    comp2 = _count_component2(sql)
    others = _count_others(sql)

    if comp1 <= 1 and others == 0 and comp2 == 0:
        return "easy"
    if (others <= 2 and comp1 <= 1 and comp2 == 0) or (
        comp1 <= 2 and others < 2 and comp2 == 0
    ):
        return "medium"
    if (
        (others > 2 and comp1 <= 2 and comp2 == 0)
        or (2 < comp1 <= 3 and others <= 2 and comp2 == 0)
        or (comp1 <= 1 and others == 0 and comp2 <= 1)
    ):
        return "hard"
    return "extra"


def apportion(counts: Counter, budget: int) -> dict:
    """Largest-remainder apportionment of `budget` across `counts` keys."""
    total = sum(counts.values())
    raw = {k: budget * v / total for k, v in counts.items()}
    base = {k: math.floor(r) for k, r in raw.items()}
    remaining = budget - sum(base.values())
    by_frac = sorted(raw, key=lambda k: (-(raw[k] - base[k]), k))
    for k in by_frac[:remaining]:
        base[k] += 1
    return base


def sane(rec: dict) -> bool:
    db_path = DB_ROOT / rec["db_id"] / f"{rec['db_id']}.sqlite"
    return (
        bool(rec.get("question"))
        and bool((rec.get("query") or "").strip())
        and isinstance(rec.get("sql"), dict)
        and db_path.exists()
    )


def build_manifest() -> dict:
    rng = random.Random(SEED)
    items = json.loads(SOURCE_FILE.read_text(encoding="utf-8"))
    recs = [{**item, "_index": i} for i, item in enumerate(items)]

    db_pop = Counter(r["db_id"] for r in recs)
    db_cap = {
        db: math.ceil(BUDGET * n / len(recs)) + DB_SLACK for db, n in db_pop.items()
    }
    strata = Counter(eval_hardness(r["sql"]) for r in recs)
    quota = apportion(strata, BUDGET)

    def pref_key(rec: dict):
        return (len(rec["question"]) > MAX_QUESTION_LEN, rng.random())

    selected: list[dict] = []
    used_db: Counter = Counter()
    seen_q: set = set()

    def take(candidates: list[dict], n: int) -> int:
        got = 0
        viable = [c for c in candidates if sane(c)]
        for rec in sorted(viable, key=pref_key):
            q = " ".join(rec["question"].lower().split())
            if q in seen_q:
                continue
            if used_db[rec["db_id"]] >= db_cap[rec["db_id"]]:
                continue
            selected.append(rec)
            seen_q.add(q)
            used_db[rec["db_id"]] += 1
            got += 1
            if got == n:
                break
        return got

    for lab in sorted(quota, key=lambda lab: (strata[lab], lab)):  # rare-first
        take([r for r in recs if eval_hardness(r["sql"]) == lab], quota[lab])

    assert len(selected) == BUDGET, f"selected {len(selected)} != budget {BUDGET}"
    coverage = Counter(eval_hardness(r["sql"]) for r in selected)
    for lab, q in quota.items():
        assert coverage[lab] == q, f"{lab}: coverage {coverage[lab]} != quota {q}"

    quota_order = sorted(quota, key=lambda lab: (-strata[lab], lab))
    ids_by_difficulty = {lab: [] for lab in quota_order}
    for rec in selected:
        ids_by_difficulty[eval_hardness(rec["sql"])].append(rec["_index"])

    share = BUDGET / len(recs)
    return {
        "version": 1,
        "created": date.today().isoformat(),
        "seed": SEED,
        "source_file": str(SOURCE_FILE.relative_to(REPO_ROOT)),
        "n_questions": len(selected),
        "formula": (
            f"largest-remainder apportionment of a {BUDGET}-question budget across "
            "official Spider eval_hardness strata (proportional to source counts); "
            "strata sampled rare-first"
        ),
        "gates": [
            "db sqlite file present",
            "non-empty question and gold SQL",
            "parsed sql field present",
            "unique normalized question text",
            f"max ceil({share:.4g} * db count) + {DB_SLACK} questions per database",
            "short questions preferred",
        ],
        "quotas": {lab: quota[lab] for lab in quota_order},
        "ids": [rec["_index"] for rec in selected],
        "ids_by_difficulty": ids_by_difficulty,
        "items": [
            {
                "index": rec["_index"],
                "db_id": rec["db_id"],
                "difficulty": eval_hardness(rec["sql"]),
                "question": rec["question"],
            }
            for rec in selected
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate/verify the Spider subset manifest")
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed manifest matches a fresh deterministic regeneration",
    )
    args = parser.parse_args()

    manifest = build_manifest()
    if args.check:
        committed = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
        for m in (manifest, committed):
            m.pop("created", None)
        if manifest != committed:
            print(f"MISMATCH: {MANIFEST_FILE} differs from regeneration", file=sys.stderr)
            sys.exit(1)
        print(f"OK: {MANIFEST_FILE.name} matches regeneration ({manifest['n_questions']} questions)")
        return

    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {MANIFEST_FILE} ({manifest['n_questions']} questions, seed {SEED})")


if __name__ == "__main__":
    main()
