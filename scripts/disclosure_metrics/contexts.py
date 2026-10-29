"""Per-dataset value inventories and observation classification."""

from __future__ import annotations

import json
import sqlite3
import time
from collections import OrderedDict
from pathlib import Path

from .common import (
    CARDINALITY_RE,
    HITAB_CELL_LINE_RE,
    HITAB_HANDLE_RE,
    LIND_HANDLE_RE,
    MIN_GENERIC_LEN,
    NON_VALUE_KEYS,
    ROOT,
    STRICT_PATHS,
    TYPE_PLACEHOLDER_RE,
    _count_tokens,
    count_token_occurrences,
    flatten_leaves,
    match_inventory,
    norm,
    strip_columns_block,
    strip_eag_failure_lines,
)


class DatasetContext:
    """Per-dataset value inventory + observation classification."""

    name = ""
    has_handles = False

    def __init__(self) -> None:
        self._inv_cache: dict[
            tuple[str, str], tuple[list[tuple[str, str]], list[tuple[str, str]]]
        ] = {}

    def prepare(self, runs_root: Path, cell_dirs: list[Path]) -> None:
        """Bulk preparation across all runs of the dataset (called once)."""

    def prepare_run(self, run_key: str, config: dict) -> None:
        """Per-run hook: receive the run's config (test variant, etc.)."""

    def _pairs(
        self, item: dict, run_key: str
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        """Cached (strict, broad) (original, normalized) value pairs per task.

        Keyed by run as well as task: lindisfarne inventories cover the records
        a specific trajectory touched, and the same task id recurs across
        modes and model cells with different trajectories.
        """
        cached = self._inv_cache.get((run_key, item["id"]))
        if cached is None:
            cached = self._build_inventory(item)
            self._inv_cache[(run_key, item["id"])] = cached
        return cached

    def _build_inventory(
        self, item: dict
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        raise NotImplementedError

    def classify_values(
        self, item: dict, text: str, run_key: str
    ) -> tuple[int, bool]:
        raise NotImplementedError

    def handles(self, item: dict, text: str) -> list[str]:
        return []

    def preprocess(self, text: str) -> str:
        """Remove non-value content classes before value matching.

        Cardinality spans (match/row/cell counts) are the cardinality class,
        never values -- e.g. 'Rows returned: 100' must not let a sampled value
        '100' collide with the count.
        """
        return CARDINALITY_RE.sub(" ", text)

    def disclosed_records(
        self, item: dict, hay_norm: str, run_key: str
    ) -> set[str]:
        """Record IDs with >=1 question-excluded identifying value in the text."""
        return set()

    def arg_value_collision(self, item: dict, args_norm: str, run_key: str) -> bool:
        """Model-emitted-args channel: any question-excluded store value present."""
        pairs, _ = self._pairs(item, run_key)
        return any(nv and count_token_occurrences(args_norm, nv) for _, nv in pairs)

    @staticmethod
    def _exclude(pairs: list[tuple[str, str]], question: str) -> list[tuple[str, str]]:
        nq = norm(question)
        return [
            (orig, nv)
            for orig, nv in pairs
            if not (len(nv) >= 4 and nv in nq)
        ]


class LindisfarneData(DatasetContext):
    name = "lindisfarne"
    has_handles = True

    def prepare(self, runs_root: Path, cell_dirs: list[Path]) -> None:
        touched: set[str] = set()
        for cell in cell_dirs:
            for run_file in sorted((runs_root / cell).glob("*/run-*.json")):
                data = json.loads(run_file.read_text())
                for item in data["results"]:
                    blob = json.dumps(item.get("steps", []))
                    touched.update(LIND_HANDLE_RE.findall(blob))
        corpus_dir = ROOT / "data" / "lindisfarne" / "full"
        files = sorted(corpus_dir.glob("records_*.jsonl"))
        if not files:
            raise SystemExit(f"lindisfarne corpus not found under {corpus_dir}")
        self._strict: dict[str, list[tuple[str, str]]] = {}
        self._broad: dict[str, list[tuple[str, str]]] = {}
        for path in files:
            with path.open() as fh:
                for line in fh:
                    rec = json.loads(line)
                    rid = rec.get("id", "")
                    if rid not in touched:
                        continue
                    leaves: list[tuple[str, str]] = []
                    flatten_leaves(rec, leaves)
                    s_vals: list[tuple[str, str]] = []
                    b_vals: list[tuple[str, str]] = []
                    for key, val in leaves:
                        if key in NON_VALUE_KEYS or len(val) < MIN_GENERIC_LEN:
                            continue
                        b_vals.append((val, norm(val)))
                        if key in STRICT_PATHS and not val.startswith("REC-"):
                            s_vals.append((val, norm(val)))
                    self._strict[rid] = s_vals
                    self._broad[rid] = b_vals
        print(f"[lindisfarne] loaded values for {len(self._strict)} touched records")

    def _task_records(self, item: dict) -> set[str]:
        blob = json.dumps(item.get("steps", []))
        return set(LIND_HANDLE_RE.findall(blob))

    def _build_inventory(
        self, item: dict
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        strict: list[tuple[str, str]] = []
        broad: list[tuple[str, str]] = []
        for rid in self._task_records(item):
            strict.extend(self._strict.get(rid, []))
            broad.extend(self._broad.get(rid, []))
        q = item.get("question", "")
        return self._exclude(strict, q), self._exclude(broad, q)

    def classify_values(self, item: dict, text: str, run_key: str) -> tuple[int, bool]:
        strict_pairs, _ = self._pairs(item, run_key)
        hay = norm(text)
        toks = match_inventory(strict_pairs, hay)
        return toks, toks > 0

    def disclosed_records(self, item: dict, hay_norm: str, run_key: str) -> set[str]:
        """Record IDs with >=1 question-excluded identifying value in the text."""
        strict_pairs, _ = self._pairs(item, run_key)
        inv_nv = {nv for _, nv in strict_pairs}
        found: set[str] = set()
        for rid in self._task_records(item):
            for _, nv in self._strict.get(rid, []):
                if nv and nv in inv_nv and count_token_occurrences(hay_norm, nv):
                    found.add(rid)
                    break
        return found

    def handles(self, item: dict, text: str) -> list[str]:
        q = norm(item.get("question", ""))
        return [h for h in LIND_HANDLE_RE.findall(text) if norm(h) not in q]


class HitabData(DatasetContext):
    name = "hitab"
    has_handles = True

    def prepare(self, runs_root: Path, cell_dirs: list[Path]) -> None:  # noqa: ARG002
        source = ROOT / "data" / "hitab" / "test_samples_qualitycheck.jsonl"
        self._id_to_table: dict[str, str] = {}
        with source.open() as fh:
            for line in fh:
                row = json.loads(line)
                self._id_to_table[row["id"]] = row["table_id"]
        self._table_cache: dict[str, list[tuple[str, str]]] = {}

    def _cells(self, table_id: str) -> list[tuple[str, str]]:
        if table_id in self._table_cache:
            return self._table_cache[table_id]
        path = ROOT / "data" / "hitab" / "tables" / "hmt" / f"{table_id}.json"
        table = json.loads(path.read_text())
        leaves: list[tuple[str, str]] = []
        flatten_leaves(table, leaves)
        vals = sorted({
            v
            for _, v in leaves
            if v and v != "<TOP>" and not TYPE_PLACEHOLDER_RE.match(v)
        })
        pairs = [(v, norm(v)) for v in vals]
        pairs = [(o, nv) for o, nv in pairs if nv]
        self._table_cache[table_id] = pairs
        return pairs

    def _build_inventory(
        self, item: dict
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        iid = item["id"].removeprefix("hitab_")
        pairs = self._cells(self._id_to_table[iid])
        kept = self._exclude(pairs, item.get("question", ""))
        return kept, kept

    def classify_values(self, item: dict, text: str, run_key: str) -> tuple[int, bool]:
        # Structured parse: "(row=R, col=C): VALUE" lines carry exact cells.
        toks = 0
        parsed = 0
        for m in HITAB_CELL_LINE_RE.finditer(text):
            parsed += 1
            raw = m.group(3).strip()
            if not raw or TYPE_PLACEHOLDER_RE.match(raw):
                continue
            toks += _count_tokens(raw)
        if parsed:
            return toks, toks > 0
        # Fallback: generic matching against the task's table inventory.
        pairs, _ = self._pairs(item, run_key)
        hay = norm(text)
        toks = 0
        for orig, nv in pairs:
            if not nv:
                continue
            if len(nv) >= MIN_GENERIC_LEN or not nv.isdigit():
                n = count_token_occurrences(hay, nv)
                if n:
                    toks += n * _count_tokens(orig)
        return toks, toks > 0

    def handles(self, item: dict, text: str) -> list[str]:
        return HITAB_HANDLE_RE.findall(text)


class SqlData(DatasetContext):
    """bird/spider: full per-database distinct-value inventory.

    Every non-null, non-empty value of every table/column becomes a needle
    (normalized, >= MIN_GENERIC_LEN chars), deduplicated per database with
    the shortest original kept for token accounting. Detection joins
    consecutive haystack tokens into windows of 1..maxn tokens and looks
    each up in the needle set -- equivalent to the standalone-token
    semantics of count_token_occurrences, but O(tokens * maxn) per
    observation instead of O(needles). Question exclusion is applied to
    matches on the fly (same rule as DatasetContext._exclude), so no
    per-task inventory copy is built. Inventories are LRU-cached per
    database (largest measured: financial, ~1.1M needles, ~0.5GB).
    """

    has_handles = False
    # Must hold a dataset's full working set (bird: 11, spider: 19
    # databases) -- a smaller LRU thrashes because runs walk databases
    # in cluster order. Peak memory ~1-2GB (financial is the largest).
    _CACHE_SIZE = 32

    def __init__(self, name: str, db_root: Path, id_to_db: dict[str, str]) -> None:
        super().__init__()
        self.name = name
        self._db_root = db_root
        self._id_to_db = id_to_db
        # db_id -> (normalized -> shortest original, max needle tokens)
        self._db_cache: OrderedDict[str, tuple[dict[str, str], int]] = OrderedDict()

    def _db_for(self, item: dict, run_key: str) -> str:
        return self._id_to_db[item["id"]]

    def _full_db(self, db_id: str) -> tuple[dict[str, str], int]:
        if db_id in self._db_cache:
            self._db_cache.move_to_end(db_id)
            return self._db_cache[db_id]
        db_path = self._db_root / db_id / f"{db_id}.sqlite"
        if not db_path.exists():
            raise SystemExit(f"{self.name} database not found: {db_path}")
        t0 = time.perf_counter()
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        # Some spider databases store non-UTF-8 text (e.g. 'Selin G..Lseren')
        # that crashes the default strict decoder; decode leniently so the
        # inventory is total. Replacement chars are stripped by norm().
        con.text_factory = lambda b: b.decode("utf-8", errors="replace")
        try:
            tables = [
                r[0]
                for r in con.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                    " AND name NOT LIKE 'sqlite_%'"
                )
            ]
            raw_seen: set[str] = set()
            nv_to_orig: dict[str, str] = {}
            for tbl in tables:
                try:
                    cols = [r[1] for r in con.execute(f'PRAGMA table_info("{tbl}")')]
                except sqlite3.Error:
                    continue
                for col in cols:
                    try:
                        cur = con.execute(
                            f'SELECT "{col}" FROM "{tbl}" WHERE "{col}" IS NOT NULL'
                            f' AND "{col}" != \'\''
                        )
                    except sqlite3.Error:
                        continue
                    while True:
                        batch = cur.fetchmany(50_000)
                        if not batch:
                            break
                        for (v,) in batch:
                            s = str(v)
                            if s in raw_seen:
                                continue
                            raw_seen.add(s)
                            nv = norm(s)
                            if not nv or len(nv) < MIN_GENERIC_LEN:
                                continue
                            if nv not in nv_to_orig or len(s) < len(nv_to_orig[nv]):
                                nv_to_orig[nv] = s
        finally:
            con.close()
        maxn = max((nv.count(" ") + 1 for nv in nv_to_orig), default=1)
        if maxn > 20:
            print(
                f"[{self.name}/{db_id}] warning: longest value has {maxn} tokens;"
                " window scan scales with it",
                flush=True,
            )
        self._db_cache[db_id] = (nv_to_orig, maxn)
        while len(self._db_cache) > self._CACHE_SIZE:
            self._db_cache.popitem(last=False)
        print(
            f"[{self.name}/{db_id}] full inventory: {len(nv_to_orig):,} values"
            f" in {time.perf_counter() - t0:.1f}s",
            flush=True,
        )
        return self._db_cache[db_id]

    @staticmethod
    def _window_matches(
        nv_to_orig: dict[str, str], maxn: int, hay_norm: str, q_norm: str
    ) -> set[str]:
        """Needles present as standalone token runs; question-excluded dropped."""
        tokens = hay_norm.split()
        if not tokens:
            return set()
        matched: set[str] = set()
        for n in range(1, maxn + 1):
            for i in range(len(tokens) - n + 1):
                key = " ".join(tokens[i : i + n])
                if key in nv_to_orig and key not in matched:
                    if not (len(key) >= 4 and key in q_norm):
                        matched.add(key)
        return matched

    def classify_values(self, item: dict, text: str, run_key: str) -> tuple[int, bool]:
        db_id = self._db_for(item, run_key)
        nv_to_orig, maxn = self._full_db(db_id)
        hay = norm(text)
        matched = self._window_matches(
            nv_to_orig, maxn, hay, norm(item.get("question", ""))
        )
        toks = sum(
            count_token_occurrences(hay, nv) * _count_tokens(nv_to_orig[nv])
            for nv in matched
        )
        return toks, bool(matched)

    def arg_value_collision(self, item: dict, args_norm: str, run_key: str) -> bool:
        db_id = self._db_for(item, run_key)
        nv_to_orig, maxn = self._full_db(db_id)
        return bool(
            self._window_matches(
                nv_to_orig, maxn, args_norm, norm(item.get("question", ""))
            )
        )

    def preprocess(self, text: str) -> str:
        """Drop non-value content classes before value matching.

        Cardinality spans (match/row/cell counts) are the cardinality class,
        never values -- e.g. 'Rows returned: 100' must not let a store value
        '100' collide with the count. The 'Columns:' header block and
        '[EAG] Query failed:' diagnostic lines are metadata/driver classes
        (see strip_columns_block / strip_eag_failure_lines), never row
        content; baseline observations carry neither, so baseline matching
        is untouched.
        """
        text = super().preprocess(text)
        return strip_eag_failure_lines(strip_columns_block(text))


class SpiderData(SqlData):
    """spider: variant-aware id -> database mapping.

    Spider ids are enumerate positions over the loaded item list, and the
    position-to-database assignment differs between the committed subset
    (manifest order) and the full dev set. The active variant is recorded
    per run via ``prepare_run`` and resolved when the inventory is built.
    """

    def __init__(self, db_root: Path, mappings: dict[str, dict[str, str]]) -> None:
        super().__init__("spider", db_root, {})
        self._mappings = mappings
        self._active_variant: dict[str, str] = {}

    def prepare_run(self, run_key: str, config: dict) -> None:
        self._active_variant[run_key] = config.get("test_variant", "full")

    def _db_for(self, item: dict, run_key: str) -> str:
        variant = self._active_variant.get(run_key, "full")
        mapping = self._mappings[variant]
        if item["id"] not in mapping:
            raise SystemExit(
                f"spider id {item['id']} not in '{variant}' mapping "
                f"({len(mapping)} entries)"
            )
        return mapping[item["id"]]


def build_bird_context() -> SqlData:
    data_file = ROOT / "data" / "bird" / "minidev" / "MINIDEV" / "mini_dev_sqlite.json"
    items = json.loads(data_file.read_text())
    id_to_db = {f"bird_{it['question_id']}": it["db_id"] for it in items}
    return SqlData("bird", data_file.parent / "dev_databases", id_to_db)


def build_spider_context() -> "SpiderData":
    items = json.loads((ROOT / "data" / "spider" / "dev.json").read_text())
    full_map = {f"spider_{i}": it["db_id"] for i, it in enumerate(items)}
    manifest_path = ROOT / "benchmarks" / "spider" / "subset.json"
    subset_map: dict[str, str] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        # Spider dev.json has no question id: the runner assigns spider_{idx}
        # by enumerate position over the LOADED item list. Subset runs load
        # manifest['ids'] in manifest order, so spider_{idx} is
        # dev.json[ids[idx]] -- NOT dev.json[idx].
        subset_map = {
            f"spider_{i}": items[dev_idx]["db_id"]
            for i, dev_idx in enumerate(manifest["ids"])
        }
    return SpiderData(
        ROOT / "data" / "spider" / "database",
        {"full": full_map, "subset": subset_map},
    )
