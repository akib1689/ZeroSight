#!/usr/bin/env python3
"""Generate the committed BIRD `subset` manifest (benchmarks/bird/subset.json).

Selection spec (frozen, seed 20260911):
  - Source: data/bird/minidev/MINIDEV/mini_dev_sqlite.json (500 questions)
  - Exclusions: question_ids listed in benchmarks/bird/minidev_offenders.csv
    are dropped before any quota math (manual audit: gold SQL provably wrong,
    or unit/denominator/scope defects — see the CSV's severity column)
  - Budget: per-database proportional (unitary-method) representation of the
    clean pool: quota(db) = ceil(clean_db_count / clean_total * 195).
    Ceilings push the total slightly above 200 — with the current clean pool
    of 428 questions the manifest holds 201 questions:
    formula_1 27, superhero 24, card_games 21, student_club 20,
    european_football_2 20, thrombosis_prediction 19, codebase_community 19,
    toxicology 16, financial 13, california_schools 13,
    debit_card_specializing 9
  - Gates: db sqlite present, non-empty question and gold SQL, unique
    (question_id, db_id), unique normalized question text, short questions
    preferred

Manifest `ids` are indices into the source file (question_id is not unique
in mini_dev_sqlite.json); the `items` array carries (question_id, db_id,
difficulty, question) for auditing.

v2 (2026-09-12): added the minidev offender exclusion gate and replaced
v1's difficulty-strata largest-remainder apportionment (59/100/41) with
per-database unitary-method quotas. Results produced against a v1 manifest
are not comparable with v2 runs.

Usage:
    uv run python scripts/make_bird_subset.py           # (re)write manifest
    uv run python scripts/make_bird_subset.py --check   # determinism guard
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_FILE = REPO_ROOT / "data" / "bird" / "minidev" / "MINIDEV" / "mini_dev_sqlite.json"
DB_ROOT = SOURCE_FILE.parent / "dev_databases"
MANIFEST_FILE = REPO_ROOT / "benchmarks" / "bird" / "subset.json"
EXCLUSION_FILE = REPO_ROOT / "benchmarks" / "bird" / "minidev_offenders.csv"

SEED = 20260911
SUBSET_BASE = 195  # proportional base; ceil rounds the total slightly above 200
MAX_QUESTION_LEN = 150
MANIFEST_VERSION = 2


def load_exclusions() -> set[int]:
    """question_ids excluded per the minidev audit (CSV is the filter)."""
    if not EXCLUSION_FILE.exists():
        raise FileNotFoundError(
            f"exclusion list missing: {EXCLUSION_FILE} — the offender CSV is "
            "required to build the subset deterministically"
        )
    with EXCLUSION_FILE.open(newline="", encoding="utf-8") as fh:
        return {int(row["question_id"]) for row in csv.DictReader(fh)}


def db_quota(clean_db_counts: Counter, clean_total: int) -> dict:
    """Unitary-method proportional quota per database (ceil)."""
    return {
        db: math.ceil(n / clean_total * SUBSET_BASE)
        for db, n in clean_db_counts.items()
    }


def sane(rec: dict) -> bool:
    db_path = DB_ROOT / rec["db_id"] / f"{rec['db_id']}.sqlite"
    return (
        bool(rec.get("question"))
        and bool((rec.get("SQL") or "").strip())
        and db_path.exists()
    )


def build_manifest() -> dict:
    rng = random.Random(SEED)
    items = json.loads(SOURCE_FILE.read_text(encoding="utf-8"))
    recs = [{**item, "_index": i} for i, item in enumerate(items)]

    excluded = load_exclusions()
    recs = [r for r in recs if r["question_id"] not in excluded]

    db_pop = Counter(r["db_id"] for r in recs)
    quota = db_quota(db_pop, len(recs))
    for db, q in quota.items():
        assert db_pop[db] >= q, f"{db}: clean pool {db_pop[db]} < quota {q}"

    def pref_key(rec: dict):
        return (len(rec["question"]) > MAX_QUESTION_LEN, rng.random())

    selected: list[dict] = []
    seen_pair: set = set()
    seen_q: set = set()

    def take(candidates: list[dict], n: int) -> int:
        got = 0
        viable = [c for c in candidates if sane(c)]
        for rec in sorted(viable, key=pref_key):
            pair = (rec["question_id"], rec["db_id"])
            if pair in seen_pair:
                continue
            q = " ".join(rec["question"].lower().split())
            if q in seen_q:
                continue
            selected.append(rec)
            seen_pair.add(pair)
            seen_q.add(q)
            got += 1
            if got == n:
                break
        return got

    for db in sorted(quota):
        take([r for r in recs if r["db_id"] == db], quota[db])

    total = sum(quota.values())
    assert len(selected) == total, f"selected {len(selected)} != quota total {total}"
    assert not {r["question_id"] for r in selected} & excluded, (
        "excluded question leaked into selection"
    )
    coverage = Counter(r["db_id"] for r in selected)
    for db, q in quota.items():
        assert coverage[db] == q, f"{db}: coverage {coverage[db]} != quota {q}"

    ids_by_difficulty = {}
    for rec in selected:
        ids_by_difficulty.setdefault(rec["difficulty"], []).append(rec["_index"])

    return {
        "version": MANIFEST_VERSION,
        "created": date.today().isoformat(),
        "seed": SEED,
        "source_file": str(SOURCE_FILE.relative_to(REPO_ROOT)),
        "exclusion_file": str(EXCLUSION_FILE.relative_to(REPO_ROOT)),
        "excluded_question_ids": len(excluded),
        "clean_pool": len(recs),
        "n_questions": len(selected),
        "formula": (
            f"drop minidev offender question_ids, then per-database unitary-method "
            f"quotas: ceil(clean_db_count / {len(recs)} * {SUBSET_BASE}); ceilings "
            f"make the total {total} (slightly above 200)"
        ),
        "gates": [
            "question_id not listed in benchmarks/bird/minidev_offenders.csv",
            "db sqlite file present",
            "non-empty question and gold SQL",
            "unique (question_id, db_id)",
            "unique normalized question text",
            "short questions preferred",
        ],
        "quotas": {db: quota[db] for db in sorted(quota)},
        "ids": [rec["_index"] for rec in selected],
        "ids_by_difficulty": ids_by_difficulty,
        "items": [
            {
                "index": rec["_index"],
                "question_id": rec["question_id"],
                "db_id": rec["db_id"],
                "difficulty": rec["difficulty"],
                "question": rec["question"],
            }
            for rec in selected
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate/verify the BIRD subset manifest")
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
