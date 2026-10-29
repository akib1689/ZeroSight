"""Shared primitives: normalization, token counting, patterns, matching."""

from __future__ import annotations

import re
from pathlib import Path

try:
    import litellm

    def _count_tokens(text: str) -> int:
        if not text:
            return 0
        try:
            return litellm.token_counter(text=text)
        except Exception:
            return len(text) // 4

except ImportError:
    def _count_tokens(text: str) -> int:
        return len(text) // 4


ROOT = Path(__file__).resolve().parent.parent.parent
_PUNCT = re.compile(r"[^a-z0-9]+")


def norm(s: str) -> str:
    """Lowercase, map every non-alphanumeric run to a single space, strip."""
    return _PUNCT.sub(" ", s.lower()).strip()


def count_token_occurrences(hay_norm: str, lit_norm: str) -> int:
    """Count standalone-token occurrences of a normalized literal."""
    if not lit_norm or lit_norm not in hay_norm:
        return 0
    return len(re.findall(rf"(?<!\S){re.escape(lit_norm)}(?!\S)", hay_norm))


# ---------------------------------------------------------------------------
# Cardinality + handle patterns (matched on the original observation text)
# ---------------------------------------------------------------------------
CARDINALITY_RE = re.compile(
    r"\d+\s+match(?:es)?\b"
    r"|\(\d+\s+of\s+\d+\)"
    r"|\d+\s+rows?\s+returned"
    r"|Rows\s+returned:\s*\d+"
    r"|\d+\s+cell\(s\)"
    r"|\d+\s+matched\b"
)
LIND_HANDLE_RE = re.compile(r"REC-\d{8}")
HITAB_HANDLE_RE = re.compile(r"\(\s*rows?\s*=\s*\d+\s*,\s*cols?\s*=\s*\d+\s*\)")
HITAB_CELL_LINE_RE = re.compile(
    r"^\(\s*rows?\s*=\s*(\d+)\s*,\s*cols?\s*=\s*(\d+)\s*\)\s*:\s*(.+)$",
    re.MULTILINE,
)
TYPE_PLACEHOLDER_RE = re.compile(r"^<[^<>]+>$")

# lindisfarne leaf-path suffixes treated as *identifying* values (the strict
# tier used for VVF/RD); everything else counts only in the broad tier.
STRICT_PATHS = {
    "name", "dob", "nhs_number", "ni_number", "postcode", "phone", "email",
    "line1", "town", "gp", "trust", "clinician", "clinicianemail", "ward",
    "specialty", "code", "description", "test",
}
NON_VALUE_KEYS = {
    "resourcetype", "profile", "versionid", "lastupdated", "gender", "country",
    "severity", "status", "frequency", "route", "duration", "type",
}
MIN_GENERIC_LEN = 3

# EAG SQL observations label the output-column block with a "Columns:" line;
# unaliased computed columns make SQLite echo the full SQL expression as the
# column name, so the block can wrap across many lines. The block is metadata
# (model-emitted SQL reflected back), never row content; strip it wholesale.
_COLUMNS_BLOCK_RE = re.compile(r"^Columns:.*?(?=^Rows returned:|\Z)", re.M | re.S)


def strip_columns_block(text: str) -> str:
    """Remove the whole EAG 'Columns:' header block (multiline-safe)."""
    return _COLUMNS_BLOCK_RE.sub("", text)


# EAG data tools render execution failures as '[EAG] Query failed: ...'
# diagnostics: SQLite driver text plus (at most) an echo of the model's own
# SQL -- never row content, since executed values live only in local
# artifacts. Driver boilerplate tokens collide with ordinary store values
# ('error', 'union', 'token', 'only' all exist as literals in the frozen
# cells), so the line is stripped wholesale before value matching. Baseline
# observations carry no '[EAG]' prefix, so this cannot touch baseline
# matching. Contrast lindisfarne: its tool-error echoes DO carry
# post-resolution record values and are deliberately NOT stripped -- the
# full-corpus oracle counts them as genuine violations.
_EAG_FAILURE_LINE_RE = re.compile(r"^\[EAG\] Query failed:.*$", re.M)


def strip_eag_failure_lines(text: str) -> str:
    """Remove EAG '[EAG] Query failed:' diagnostic lines (single-line)."""
    return _EAG_FAILURE_LINE_RE.sub("", text)


def flatten_leaves(obj, out: list[tuple[str, str]], key: str = "") -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten_leaves(v, out, str(k).lower())
    elif isinstance(obj, list):
        for v in obj:
            flatten_leaves(v, out, key)
    elif obj is None or isinstance(obj, bool):
        return
    else:
        out.append((key, str(obj)))


def match_inventory(pairs: list[tuple[str, str]], hay_norm: str) -> int:
    """Token cost of inventory literals found in a normalized haystack."""
    toks = 0
    for orig, nv in pairs:
        if nv and nv in hay_norm:
            n = count_token_occurrences(hay_norm, nv)
            toks += n * _count_tokens(orig)
    return toks
