#!/usr/bin/env python3
"""Generate the committed HiTab `subset` manifest (benchmarks/hitab/subset.json).

Selection spec (frozen, seed 20260910):
  - Source: data/hitab/test_samples_qualitycheck.jsonl (1,584 questions)
  - Quota per aggregation label: ceil(0.1 * label_count), multi-label aware
    -> 169 questions total, every label covered
  - Labels sampled rare-first, then by descending population, `none` last
  - Pure-label records preferred, so per-label coverage == quota exactly
  - Gates: non-empty question/answer, linked_cells present, unique
    (table_id, sentence_id), unique normalized question text, max 2
    questions per table, single-value answers preferred, short answers
    preferred

Usage:
    uv run python scripts/make_hitab_subset.py           # (re)write manifest
    uv run python scripts/make_hitab_subset.py --check   # determinism guard
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
SOURCE_FILE = REPO_ROOT / "data" / "hitab" / "test_samples_qualitycheck.jsonl"
MANIFEST_FILE = REPO_ROOT / "benchmarks" / "hitab" / "subset.json"

SEED = 20260910
SOURCE_VARIANT = "qualitycheck"
MAX_PER_TABLE = 2
RARE_LABEL_POP = 10  # labels with population <= this are sampled first

GATES = [
    "non-empty question and answer",
    "linked_cells present",
    "unique (table_id, sentence_id)",
    "unique normalized question text",
    f"max {MAX_PER_TABLE} questions per table",
    "single-value answers preferred, then short answers",
]


def labels_of(record: dict) -> tuple:
    return tuple(record.get("aggregation") or [])


def sane(record: dict) -> bool:
    return (
        bool(record.get("question"))
        and bool(record.get("answer"))
        and record.get("linked_cells") is not None
    )


def build_manifest() -> dict:
    rng = random.Random(SEED)

    def pref_key(record: dict):
        answer = record.get("answer") or [""]
        return (
            len(answer) != 1,
            len(str(answer[0])) > 40,
            len(record["question"]) > 140,
            rng.random(),
        )

    records = [json.loads(line) for line in SOURCE_FILE.read_text(encoding="utf-8").splitlines()]
    pop = Counter()
    for s in records:
        for lab in set(labels_of(s)):
            pop[lab] += 1
    quota = {lab: math.ceil(0.1 * pop[lab]) for lab in pop}

    rare = sorted(lab for lab in pop if pop[lab] <= RARE_LABEL_POP)
    rest = sorted(
        (lab for lab in pop if pop[lab] > RARE_LABEL_POP and lab != "none"),
        key=lambda lab: -pop[lab],
    )
    order = rare + rest + ["none"]

    selected: list[dict] = []
    sel_ids: set[str] = set()
    used_pair: set[tuple] = set()
    used_table: Counter = Counter()
    seen_q: set[str] = set()

    def take(candidates: list[dict], n: int) -> int:
        got = 0
        viable = [c for c in candidates if c["id"] not in sel_ids and sane(c)]
        for s in sorted(viable, key=pref_key):
            pair = (s["table_id"], s.get("sentence_id"))
            if pair in used_pair:
                continue
            q = " ".join(s["question"].lower().split())
            if q in seen_q:
                continue
            if used_table[s["table_id"]] >= MAX_PER_TABLE:
                continue
            selected.append(s)
            sel_ids.add(s["id"])
            used_pair.add(pair)
            used_table[s["table_id"]] += 1
            seen_q.add(q)
            got += 1
            if got == n:
                break
        return got

    for lab in order:
        pure = [s for s in records if set(labels_of(s)) == {lab}]
        got = take(pure, quota[lab])
        if got < quota[lab]:  # fallback: multi-label records containing lab
            multi = [s for s in records if lab in labels_of(s)]
            take(multi, quota[lab] - got)

    coverage = Counter()
    for s in selected:
        for lab in set(labels_of(s)):
            coverage[lab] += 1
    assert len(selected) == sum(quota.values()), (
        f"selected {len(selected)} != quota sum {sum(quota.values())}"
    )
    for lab in quota:
        assert coverage[lab] == quota[lab], f"{lab}: coverage {coverage[lab]} != quota {quota[lab]}"

    quota_order = sorted(quota, key=lambda lab: (-pop[lab], lab))
    ids_by_label = {lab: [] for lab in quota_order}
    for s in selected:
        for lab in quota_order:
            if lab in labels_of(s):
                ids_by_label[lab].append(s["id"])

    return {
        "version": 1,
        "created": date.today().isoformat(),
        "seed": SEED,
        "source_variant": SOURCE_VARIANT,
        "source_file": str(SOURCE_FILE.relative_to(REPO_ROOT)),
        "n_questions": len(selected),
        "formula": (
            "ceil(0.1 * per-label count) over aggregation labels "
            "(multi-label aware); rare labels sampled first, pure-label records preferred"
        ),
        "gates": GATES,
        "quotas": {lab: quota[lab] for lab in quota_order},
        "ids": [s["id"] for s in selected],
        "ids_by_label": ids_by_label,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate/verify the HiTab subset manifest")
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
