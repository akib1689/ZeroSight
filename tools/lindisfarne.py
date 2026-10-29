"""Lindisfarne tools — multi-format NHS record QA (FHIR R4 / HL7 v2.5 / SNOMED / GHX).

Corpus: ``data/lindisfarne/full/records_000{1,2}.jsonl`` — 20,000 synthetic NHS
records (FHIR R4 · HL7 v2.5 · SNOMED · GHX procurement). Questions live in
``benchmarks/lindisfarne/tasks.json``; golds were computed by
``scripts/generate_lindisfarne_questions.py`` over a *normalized* view of the
corpus. ``_normalize()`` below is a port of the generator's ``normalize()``
— KEEP IN SYNC (the generator stays the source of truth for golds).

Two data layers:

  - Canonical view (``find_records``): every record projected onto one field
    vocabulary (name, dob, town, diag_label, allergen, contract_ref, ...) so
    filters and sorts work uniformly across formats. "Any-element" fields
    (allergen, med_name, lab_flag, contract_ref, ...) match when ANY item in
    the underlying list matches; ``!=`` means NO item equals the value.
  - Raw view (``get_record``): the original per-format JSON, navigated with
    dotted paths + ``[N]`` indices (e.g. ``medications[0].name``,
    ``PID.Address.town``). Placeholders always navigate the full record from
    the root; the ``path`` argument trims the displayed output only.

Tool tiers (``exposure_type``):

  Metadata:   get_schema — canonical field table + per-format raw structure
  Data:       find_records, get_record
  Evaluation: submit_answer
  (+ the 7 evaluator compute tools from tools/evaluator.py, added by the
   dataset adapter)

Leak rules for EAG redaction (mirrors tools/hitab.py): observations disclose
addresses (record IDs, paths — the model chose them), counts, and types —
never values. ``find_records`` echoes filter fields/ops but never filter
values (they may be placeholder-resolved mid-run) and embeds a schema-safe
shape block per format present in the result. Placeholder paths navigate
those nested shapes (and get_record's raw structures) — NOT the flat
canonical filter vocabulary in get_schema's table.
"""

import json
import re
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from tools.evaluator import _as_number, _type_name
from tools.sql import EAGBaseTool, ToolResult

_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "lindisfarne" / "full"
_CHUNKS = ("records_0001.jsonl", "records_0002.jsonl")

_MAX_LIMIT = 50
_MAX_FIELDS_SHOWN = 8
_MAX_JSON_CHARS = 6000


# ── Canonical view (port of the generator's normalize) ───────────────────────


def _d(x: Any) -> dict:
    return x if isinstance(x, dict) else {}


def _l(x: Any) -> list:
    return x if isinstance(x, list) else []


def _dpart(ts: Any) -> str | None:
    return str(ts)[:10] if ts else None


def _normalize(r: dict) -> dict:
    """Project a raw record onto the canonical field vocabulary.

    Mirrors ``normalize()`` in scripts/generate_lindisfarne_questions.py —
    keep the two in sync (golds are computed over that view).
    """
    fmt = r.get("format")
    n = {
        "format": fmt, "name": None, "dob": None, "nhs_number": None, "gender": None,
        "town": None, "postcode": None, "gp": None, "trust": None,
        "encounter_type": None, "encounter_date": None, "ward": None, "clinician": None,
        "specialty": None, "diag_code": None, "diag_label": None, "diag_severity": None,
        "diag_status": None, "diag_onset": None, "diag_type": None,
        "lab_test": None, "med_code": None, "proc_code": None,
        "medications": [], "labs": [], "allergies": [], "procedures": [], "alerts": [],
    }
    if fmt == "FHIR":
        p, en, dg = _d(r.get("patient")), _d(r.get("encounter")), _d(r.get("diagnosis"))
        n["record_id"] = r.get("id")
        n.update(
            name=p.get("name"), dob=_dpart(p.get("dob")), nhs_number=p.get("nhs_number"),
            gender=p.get("gender"), town=_d(p.get("address")).get("town"),
            postcode=_d(p.get("address")).get("postcode"), gp=p.get("gp"),
            trust=p.get("trust"), encounter_type=en.get("type"),
            encounter_date=_dpart(en.get("date")), ward=en.get("ward"),
            clinician=en.get("clinician"), specialty=en.get("specialty"),
            diag_code=dg.get("code"), diag_label=dg.get("description"),
            diag_severity=dg.get("severity"), diag_status=dg.get("status"),
            diag_onset=_dpart(dg.get("onset")),
        )
        n["medications"] = [
            {"name": m.get("name"), "status": m.get("status"),
             "start_date": _dpart(m.get("startDate"))}
            for m in _l(r.get("medications")) if isinstance(m, dict)]
        n["labs"] = [
            {"test": lab.get("test"), "value": lab.get("value"), "date": _dpart(lab.get("date")),
             "status": lab.get("status"), "performer": lab.get("performer")}
            for lab in _l(r.get("labs")) if isinstance(lab, dict)]
        n["allergies"] = [
            {"allergen": a.get("allergen"), "reaction": a.get("reaction"),
             "severity": a.get("severity"), "confirmed": a.get("confirmed")}
            for a in _l(r.get("allergies")) if isinstance(a, dict)]
        n["procedures"] = [
            {"name": pr.get("name"), "date": _dpart(pr.get("date")), "status": pr.get("status")}
            for pr in _l(r.get("procedures")) if isinstance(pr, dict)]
        n["alerts"] = [a for a in _l(r.get("alerts")) if isinstance(a, str)]
    elif fmt == "HL7":
        pid, pv1, dg1 = _d(r.get("PID")), _d(r.get("PV1")), _d(r.get("DG1"))
        n["record_id"] = r.get("HL7MessageID")
        n.update(
            name=pid.get("PatientName"), dob=_dpart(pid.get("DateOfBirth")),
            nhs_number=pid.get("NHSNumber"), gender=pid.get("Gender"),
            town=_d(pid.get("Address")).get("town"),
            postcode=_d(pid.get("Address")).get("postcode"),
            trust=_d(r.get("MSH")).get("SendingFacility"), encounter_type=pv1.get("VisitType"),
            encounter_date=_dpart(pv1.get("VisitDate")), ward=pv1.get("Ward"),
            clinician=pv1.get("Consultant"), diag_code=dg1.get("DiagnosisCode"),
            diag_label=dg1.get("DiagnosisDesc"), diag_type=dg1.get("DiagnosisType"),
        )
        n["medications"] = [
            {"name": m.get("MedicationName"), "prescribed_by": m.get("PrescribedBy"),
             "start_date": None}
            for m in _l(r.get("RX1")) if isinstance(m, dict)]
        n["labs"] = [
            {"test": lab.get("test"), "value": lab.get("value"), "date": _dpart(lab.get("date")),
             "flag": lab.get("flag")}
            for lab in _l(r.get("OBX")) if isinstance(lab, dict)]
        n["allergies"] = [
            {"allergen": a.get("allergen"), "reaction": a.get("reaction"),
             "severity": a.get("severity"), "confirmed": a.get("confirmed")}
            for a in _l(r.get("AL1")) if isinstance(a, dict)]
    elif fmt == "SNOMED":
        pi, en, sc = (
            _d(r.get("patient_info")), _d(r.get("clinical_encounter")), _d(r.get("SNOMED_codes")))
        n["record_id"] = r.get("record_id")
        n.update(
            name=pi.get("full_name"), dob=_dpart(pi.get("dob")), nhs_number=pi.get("nhs_number"),
            town=_d(pi.get("address")).get("town"), postcode=_d(pi.get("address")).get("postcode"),
            gp=pi.get("gp_surgery"), encounter_type=en.get("type"),
            encounter_date=_dpart(en.get("date")), ward=en.get("ward"),
            clinician=en.get("clinician"),
            diag_code=sc.get("diagnosis"), diag_label=sc.get("diag_label"),
            lab_test=sc.get("lab"), med_code=sc.get("medication"), proc_code=sc.get("procedure"),
        )
        n["alerts"] = [a for a in _l(r.get("alerts")) if isinstance(a, str)]
    elif fmt == "GHX":
        sup, tr, pr, wp = (
            _d(r.get("SUPPLIER")), _d(r.get("TRUST")), _d(r.get("PROJECT")),
            _d(r.get("WORKPLAN")))
        n["record_id"] = r.get("record_id")
        n["ghx"] = {
            "company": sup.get("CompanyName"), "contact_name": sup.get("ContactName"),
            "contact_role": sup.get("ContactRole"), "trust_name": tr.get("TrustName"),
            "procurement_ref": tr.get("ProcurementRef"), "cohort_size": tr.get("CohortSize"),
            "data_scope": tr.get("DataScope"),
            "project_ref": pr.get("ProjectRef"), "bid_ref": pr.get("BidRef"),
            "bid_date": _dpart(pr.get("BidDate")), "phase": pr.get("Phase"),
            "lead_clinician": pr.get("LeadClinician"),
            "workplan_ref": wp.get("WorkplanRef"), "milestones": wp.get("Milestones"),
            "start_date": _dpart(wp.get("StartDate")),
            "planned_golive": _dpart(wp.get("PlannedGoLive")),
            "contracts": [
                {"ref": c.get("ContractRef"), "date": _dpart(c.get("ContractDate")),
                 "value": c.get("Value"), "duration": c.get("Duration"),
                 "sla": c.get("SLA"), "status": c.get("Status")}
                for c in _l(r.get("CONTRACT")) if isinstance(c, dict)],
        }
    else:
        n["record_id"] = None
    return n


# ── Corpus ────────────────────────────────────────────────────────────────────


class _Corpus:
    """Raw + normalized corpus, indexed by record ID."""

    def __init__(self, raw_records: list[dict]):
        self.raw_by_id: dict[str, dict] = {}
        self.norm: list[dict] = []
        self.norm_by_id: dict[str, dict] = {}
        self.format_counts: dict[str, int] = {}
        self._first_of_format: dict[str, dict] = {}
        for raw in raw_records:
            n = _normalize(raw)
            rid = n.get("record_id")
            if not rid:
                continue
            self.raw_by_id[rid] = raw
            self.norm.append(n)
            self.norm_by_id[rid] = n
            fmt = n.get("format") or "?"
            self.format_counts[fmt] = self.format_counts.get(fmt, 0) + 1
            self._first_of_format.setdefault(fmt, raw)
        if len(self.raw_by_id) != len(raw_records):
            raise AssertionError("lindisfarne corpus: record ids not unique")
        self._field_formats: dict[str, list[str]] | None = None

    def first_of_format(self, fmt: str) -> dict | None:
        return self._first_of_format.get(fmt)

    def field_formats(self) -> dict[str, list[str]]:
        """For each canonical field, the formats that populate it (cached)."""
        if self._field_formats is None:
            known = _KNOWN_FIELDS
            acc: dict[str, set[str]] = {f: set() for f in known}
            for rec in self.norm:
                fmt = rec.get("format") or "?"
                for f in known:
                    if _field_values(rec, f):
                        acc[f].add(fmt)
            self._field_formats = {
                f: sorted(v) for f, v in acc.items() if v}
        return self._field_formats


_CORPUS: _Corpus | None = None


def _load_corpus() -> _Corpus:
    global _CORPUS
    if _CORPUS is None:
        raw_records: list[dict] = []
        try:
            for chunk in _CHUNKS:
                with open(_DATA_DIR / chunk, encoding="utf-8") as f:
                    for line in f:
                        raw_records.append(json.loads(line))
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Lindisfarne corpus not found under {_DATA_DIR}. "
                "Run: uv run python scripts/generate_lindisfarne_questions.py "
                "(it auto-downloads the corpus)"
            ) from e
        _CORPUS = _Corpus(raw_records)
    return _CORPUS


# ── Canonical field table ─────────────────────────────────────────────────────

_SCALAR_FIELDS = frozenset({
    "name", "dob", "nhs_number", "gender", "town", "postcode", "gp", "trust",
    "encounter_type", "encounter_date", "ward", "clinician", "specialty",
    "diag_code", "diag_label", "diag_severity", "diag_status", "diag_onset",
    "diag_type", "med_code", "proc_code",
})
_GHX_SCALARS = frozenset({
    "company", "contact_name", "contact_role", "trust_name", "procurement_ref",
    "cohort_size", "data_scope", "project_ref", "bid_ref", "bid_date", "phase",
    "lead_clinician", "workplan_ref", "milestones", "start_date", "planned_golive",
})

# (group, field, type, description) — drives get_schema + validation.
# Descriptions stay minimal: self-evident fields carry no note; ANY-list
# semantics are conveyed by the "*" marker (see _ANY_FIELDS) in get_schema's
# legend; value-format hints (REC-XXXXXXXX, ...) are kept — the model needs
# them to construct filter values.
_CANONICAL_FIELDS: list[tuple[str, str, str, str]] = [
    ("identity", "record_id", "str", "REC-XXXXXXXX; unique"),
    ("patient", "name", "str", "'First Last'"),
    ("patient", "dob", "date", "birth date"),
    ("patient", "nhs_number", "str", "unique, 10 digits"),
    ("patient", "gender", "str", ""),
    ("patient", "town", "str", ""),
    ("patient", "postcode", "str", ""),
    ("patient", "gp", "str", ""),
    ("patient", "trust", "str", ""),
    ("encounter", "encounter_type", "str", ""),
    ("encounter", "encounter_date", "date", ""),
    ("encounter", "ward", "str", ""),
    ("encounter", "clinician", "str", ""),
    ("encounter", "specialty", "str", ""),
    ("diagnosis", "diag_code", "str", ""),
    ("diagnosis", "diag_label", "str", "1:1 with diag_code"),
    ("diagnosis", "diag_severity", "str", ""),
    ("diagnosis", "diag_status", "str", ""),
    ("diagnosis", "diag_onset", "date", ""),
    ("diagnosis", "diag_type", "str", ""),
    ("medication", "med_name", "str", ""),
    ("medication", "med_status", "str", ""),
    ("medication", "med_code", "str", "SNOMED code"),
    ("lab", "lab_test", "str", ""),
    ("lab", "lab_flag", "str", ""),
    ("lab", "lab_value", "str", ""),
    ("allergy", "allergen", "str", ""),
    ("allergy", "allergy_severity", "str", ""),
    ("allergy", "allergy_confirmed", "bool", ""),
    ("procedure", "proc_name", "str", ""),
    ("procedure", "proc_code", "str", "SNOMED code"),
    ("safety", "alert", "str", ""),
    ("procurement", "company", "str", ""),
    ("procurement", "contact_name", "str", ""),
    ("procurement", "contact_role", "str", ""),
    ("procurement", "trust_name", "str", "procurement trust"),
    ("procurement", "procurement_ref", "str", "PROC-XXXXX"),
    ("procurement", "cohort_size", "int", ""),
    ("procurement", "data_scope", "str", ""),
    ("procurement", "project_ref", "str", "PXXXXXX"),
    ("procurement", "bid_ref", "str", "BID-XXXXXXX"),
    ("procurement", "bid_date", "date", ""),
    ("procurement", "phase", "str", ""),
    ("procurement", "lead_clinician", "str", ""),
    ("procurement", "workplan_ref", "str", "WP-XXXXXXXX"),
    ("procurement", "milestones", "int", "count"),
    ("procurement", "start_date", "date", ""),
    ("procurement", "planned_golive", "date", ""),
    ("procurement", "contract_ref", "str", "CON-XXXXXXXXX"),
    ("procurement", "contract_value", "str", "e.g. £3,074,270"),
    ("procurement", "contract_status", "str", ""),
    ("procurement", "contract_date", "date", ""),
    ("procurement", "contract_duration", "str", ""),
]

# Fields backed by underlying lists (or a scalar + list, like lab_test): they
# match when ANY item matches, and '!=' means no item equals the value.
_ANY_FIELDS = frozenset({
    "med_name", "med_status", "lab_test", "lab_flag", "lab_value",
    "allergen", "allergy_severity", "allergy_confirmed", "proc_name",
    "alert", "contract_ref", "contract_value", "contract_status",
    "contract_date", "contract_duration",
})

_FORMAT_INITIALS = {"FHIR": "F", "HL7": "H", "SNOMED": "S", "GHX": "G"}

_KNOWN_FIELDS = (
    {"record_id", "format"} | _SCALAR_FIELDS | _GHX_SCALARS
    | {f for _, f, _, _ in _CANONICAL_FIELDS}
)
_KNOWN_OPS = ("==", "!=", "contains", "starts_with", "<", "<=", ">", ">=")
_FORMATS = ("FHIR", "HL7", "SNOMED", "GHX")

_FORMAT_NOTES = {
    "FHIR": ("Richest format: allergies, procedures, medication details "
             "(dose/route/start), specialty, GP, trust."),
    "HL7": ("v2.5 ADT-style segments (MSH/PID/PV1/DG1/RX1/OBX/AL1); no GP, "
            "no procedures, labs carry a flag instead of status."),
    "SNOMED": ("Thin: codes (diagnosis/medication/procedure) + one lab + "
               "alerts; no gender, allergies, procedures, or med list."),
    "GHX": ("Procurement: SUPPLIER / TRUST / PROJECT / WORKPLAN segments + "
            "CONTRACT[] list; no patient fields."),
}


def _field_values(rec: dict, field: str) -> list:
    """All values of a canonical field for one record (any-element semantics).

    Scalar fields yield 0 or 1 values; list-backed fields yield every item's
    value. Unknown fields yield [] (validated earlier).
    """
    if field == "record_id":
        v = rec.get("record_id")
        return [v] if v else []
    if field == "format":
        v = rec.get("format")
        return [v] if v else []
    if field in _SCALAR_FIELDS:
        v = rec.get(field)
        return [v] if v is not None else []
    if field in _GHX_SCALARS:
        v = _d(rec.get("ghx")).get(field)
        return [v] if v is not None else []
    if field == "med_name":
        return [m.get("name") for m in rec.get("medications", []) if m.get("name") is not None]
    if field == "med_status":
        return [m.get("status") for m in rec.get("medications", []) if m.get("status") is not None]
    if field == "lab_test":
        vals = [] if rec.get("lab_test") is None else [rec["lab_test"]]
        labs = [lb.get("test") for lb in rec.get("labs", []) if lb.get("test") is not None]
        return vals + labs
    if field == "lab_flag":
        return [lb.get("flag") for lb in rec.get("labs", []) if lb.get("flag") is not None]
    if field == "lab_value":
        return [lb.get("value") for lb in rec.get("labs", []) if lb.get("value") is not None]
    if field == "allergen":
        return [a.get("allergen") for a in rec.get("allergies", []) if a.get("allergen")]
    if field == "allergy_severity":
        return [a.get("severity") for a in rec.get("allergies", []) if a.get("severity")]
    if field == "allergy_confirmed":
        return [a.get("confirmed") for a in rec.get("allergies", [])
                if a.get("confirmed") is not None]
    if field == "proc_name":
        return [p.get("name") for p in rec.get("procedures", []) if p.get("name")]
    if field == "alert":
        return list(rec.get("alerts", []))
    if field.startswith("contract_"):
        key = field[len("contract_"):]
        ghx = _d(rec.get("ghx"))
        return [c.get(key) for c in ghx.get("contracts", []) if c.get(key) is not None]
    return []


# ── Filter / sort semantics ───────────────────────────────────────────────────


def _boolish(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v).strip().lower()


def _eq(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return _boolish(a) == _boolish(b)
    an, bn = _as_number(a), _as_number(b)
    if an is not None and bn is not None:
        return an == bn
    return str(a).strip().lower() == str(b).strip().lower()


def _cmp(a: Any, b: Any) -> int:
    an, bn = _as_number(a), _as_number(b)
    if an is not None and bn is not None:
        return (an > bn) - (an < bn)
    a_s, b_s = str(a).strip().lower(), str(b).strip().lower()
    return (a_s > b_s) - (a_s < b_s)


def _record_matches(rec: dict, cond: dict) -> bool:
    field, op, target = cond["field"], cond["op"], cond["value"]
    vals = _field_values(rec, field)
    if op == "!=":
        # No item equals the target (records with no values for the field
        # also match — e.g. "not allergic to X" includes no-allergy records).
        return not any(_eq(v, target) for v in vals)
    if not vals:
        return False
    if op == "==":
        return any(_eq(v, target) for v in vals)
    if op == "contains":
        t = str(target).lower()
        return any(t in str(v).lower() for v in vals)
    if op == "starts_with":
        t = str(target).lower()
        return any(str(v).lower().startswith(t) for v in vals)
    return any({
        "<": _cmp(v, target) < 0,
        "<=": _cmp(v, target) <= 0,
        ">": _cmp(v, target) > 0,
        ">=": _cmp(v, target) >= 0,
    }[op] for v in vals)


def _sort_records(recs: list[dict], field: str, order: str) -> list[dict]:
    def sort_key(r: dict):
        vals = _field_values(r, field)
        v = next((x for x in vals if x is not None), None)
        if v is None:
            return (1, 0, "")
        n = None if isinstance(v, bool) else _as_number(v)
        if n is not None:
            return (0, n, "")
        return (0, 0, str(v).lower())

    ranked = sorted(recs, key=sort_key)
    non_null = [r for r in ranked if _field_values(r, field)]
    null = [r for r in ranked if not _field_values(r, field)]
    if order == "desc":
        non_null.reverse()
    return non_null + null


# ── Path walking + shape rendering ────────────────────────────────────────────

_PATH_TOKEN_RE = re.compile(r"\.([A-Za-z_][A-Za-z0-9_-]*)|\[(\d+)\]")


def _walk_path(obj: Any, path: str) -> tuple[Any, str | None]:
    """Walk ``obj`` by a dotted/bracketed path. Returns (value, error)."""
    if path and path[0] not in ".[":
        path = "." + path
    pos, cur = 0, obj
    while pos < len(path):
        m = _PATH_TOKEN_RE.match(path, pos)
        if not m:
            return None, f"cannot parse path at '{path[pos:]}' — use .key and [N] segments"
        if m.group(1) is not None:
            if not isinstance(cur, dict):
                return None, f"segment '{m.group(1)}' needs an object, got {_type_name(cur)}"
            if m.group(1) not in cur:
                keys = sorted(str(k) for k in cur.keys())
                return None, (
                    f"no key '{m.group(1)}' — available keys: "
                    f"{', '.join(keys) if keys else 'none'}"
                )
            cur = cur[m.group(1)]
        else:
            if not isinstance(cur, list):
                return None, f"segment [{m.group(2)}] needs a list, got {_type_name(cur)}"
            idx = int(m.group(2))
            if not 0 <= idx < len(cur):
                return None, f"index [{idx}] out of range — list has {len(cur)} item(s)"
            cur = cur[idx]
        pos = m.end()
    return cur, None


def _render_shape(value: Any, depth: int = 0, max_depth: int = 5) -> str:
    """Schema-safe rendering: keys, list lengths, and types — never values."""
    if isinstance(value, dict):
        if not value:
            return "{} (empty)"
        if depth >= max_depth:
            return "{...}"
        inner = ",\n".join(
            f"{'  ' * (depth + 1)}{k}: {_render_shape(v, depth + 1, max_depth)}"
            for k, v in value.items()
        )
        return "{\n" + inner + "\n" + "  " * depth + "}"
    if isinstance(value, list):
        if not value:
            return "[] (empty)"
        return f"[{len(value)} items, each {_render_shape(value[0], depth, max_depth)}]"
    return f"<{_type_name(value)}>"


def _populated_view(rec: dict) -> dict:
    """Copy of a record keeping only keys with content.

    None scalars and empty lists/dicts are dropped — the EAG shape blocks
    show what the matched records actually populate. Omitted keys still
    exist on the artifact (they resolve to null), so this is a hint, not
    the full key map.
    """
    return {
        k: v for k, v in rec.items()
        if v is not None and v != [] and v != {}
    }


def _match_shape_blocks(matches: list[dict]) -> list[str]:
    """One schema-safe shape block per format present in ``matches``.

    Uses the first match of each format as the exemplar; ``_render_shape``
    prints keys, lengths, and types only — never values. Placeholder paths
    navigate these nested structures (the norm artifact), NOT the flat
    canonical filter vocabulary used by get_schema.
    """
    blocks: list[str] = []
    seen: set[str] = set()
    for m in matches:
        fmt = str(m.get("format") or "?")
        if fmt in seen:
            continue
        seen.add(fmt)
        blocks.append(f"{fmt} shape:\n{_render_shape(_populated_view(m))}")
    return blocks


def _display(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


# ── Input schemas ─────────────────────────────────────────────────────────────


class _FilterCond(BaseModel):
    field: str = Field(..., description="Canonical field name (see get_schema)")
    op: str = Field(
        ..., description='One of: ==, !=, contains, starts_with, <, <=, >, >= '
                         "(dates ISO YYYY-MM-DD; != means NO item equals)"
    )
    # Permissive: EAG middleware may resolve {{ }} placeholders to any type.
    value: Any = Field(..., description="Value to compare against — literal or {{ placeholder }}")


class _FindRecordsInput(BaseModel):
    filters: list[_FilterCond] = Field(
        ..., min_length=1, description="All conditions are AND-ed",
    )
    format: str | None = Field(
        None, description='Restrict to one format: "FHIR", "HL7", "SNOMED", or "GHX"',
    )
    sort_by: str | None = Field(
        None, description="Canonical field to sort by (e.g. dob for oldest/youngest)",
    )
    order: str = Field("asc", description='"asc" (default) or "desc"')
    limit: int = Field(
        20, ge=1, le=_MAX_LIMIT,
        description=f"Max matches returned (default 20, max {_MAX_LIMIT}); "
                    "the total count is always reported",
    )
    fields: list[str] | None = Field(
        None, description="Canonical fields to show in the result table "
                          "(default: record_id, format, name + filter/sort fields)",
    )


class _GetRecordInput(BaseModel):
    record_id: str = Field(..., description="Record identifier, e.g. REC-00007380")
    path: str | None = Field(
        None, description="Optional dotted path with [N] indices that trims the "
                          "displayed output (e.g. 'medications[0]', 'PID.Address.town'); "
                          "{{ }} references always navigate the full record",
    )


class _GetSchemaInput(BaseModel):
    format: str | None = Field(
        None, description='Optional: "FHIR", "HL7", "SNOMED", or "GHX" for that '
                          "format's raw record structure; omit for the canonical "
                          "field table + corpus overview",
    )


class _SubmitAnswerInput(BaseModel):
    # Permissive on purpose: EAG middleware resolves {{ }} placeholders to
    # native types before execution; a str-only schema would fail pydantic
    # validation and echo the resolved value back to the model (the exact
    # leak EAG must prevent).
    answer: str | float | int | bool = Field(
        ..., description="Final answer — a literal in baseline mode, or a "
                         "{{ placeholder }} reference (e.g. \"{{ join_items.result }}\") "
                         "in EAG mode",
    )


# ── Metadata tool ─────────────────────────────────────────────────────────────


class GetSchemaTool(EAGBaseTool):
    """Canonical field table + corpus overview, or a format's raw structure."""

    name: str = "get_schema"
    description: str = (
        "Get the corpus schema. Without arguments: corpus overview (formats, "
        "counts, uniqueness keys) and the canonical field table that "
        "find_records filters/sorts on — grouped by domain, with types and "
        "which formats populate each field. With format='FHIR'|'HL7'|"
        "'SNOMED'|'GHX': the raw record structure of that format (keys and "
        "types only) as returned by get_record. Start here."
    )
    args_schema: type[BaseModel] = _GetSchemaInput
    exposure_type: str = "metadata"

    def validate(self, **kwargs) -> tuple[bool, str]:
        fmt = kwargs.get("format")
        if fmt is not None and str(fmt).upper() not in _FORMATS:
            return False, f"format must be one of {list(_FORMATS)} or omitted"
        return True, ""

    def execute(self, **kwargs) -> ToolResult:
        corpus = _load_corpus()
        fmt = str(kwargs["format"]).upper() if kwargs.get("format") else None

        if fmt is not None:
            sample = corpus.first_of_format(fmt)
            if sample is None:
                return ToolResult(output=f"No {fmt} records in corpus.", error="empty format")
            tree = _render_shape(sample)
            out = (
                f"{fmt} raw record structure (keys/types only — navigate with "
                f"get_record paths):\n{tree}\n\nNote: {_FORMAT_NOTES[fmt]}"
            )
            return ToolResult(
                output=out,
                row_count=1,
                data={"format": fmt, "raw_tree": tree, "note": _FORMAT_NOTES[fmt]},
            )

        counts = corpus.format_counts
        total = sum(counts.values())
        populated = corpus.field_formats()

        def _pop(field: str) -> str:
            return "".join(
                _FORMAT_INITIALS[f] for f in _FORMATS if f in populated.get(field, [])
            ) or "-"

        lines = [
            f"Corpus: {total} records — " + " · ".join(
                f"{f}: {counts.get(f, 0)}" for f in _FORMATS),
            "Unique keys: record_id (REC-XXXXXXXX) · nhs_number (10 digits) · "
            "name + dob",
            "Legend: F=FHIR H=HL7 S=SNOMED G=GHX · * = ANY-list field: matches "
            "if ANY item matches; '!=' = no item matches (records lacking the "
            "field match '!=' too) · dates YYYY-MM-DD — < <= > >= and sort "
            "compare correctly",
            "These flat fields drive find_records filters/sorts only. {{ }} "
            "placeholder paths instead navigate the nested record shapes "
            "echoed by find_records / get_record results; per-format raw "
            "structures: get_schema(format=...).",
            "",
        ]
        cur_group = None
        for group, field, ftype, desc in _CANONICAL_FIELDS:
            if group != cur_group:
                cur_group = group
                lines.append(f"{group}:")
            marker = "*" if field in _ANY_FIELDS else ""
            row = f"  {field + marker:<20} {ftype:<5} {_pop(field):<4}"
            if desc:
                row += f" {desc}"
            lines.append(row.rstrip())
        lines.append("  (record_id and format are also filterable)")
        return ToolResult(
            output="\n".join(lines),
            row_count=len(_CANONICAL_FIELDS),
            data={
                "total": total,
                "format_counts": counts,
                "fields": [
                    {"group": g, "field": f, "type": t, "populated_by": populated.get(f, []),
                     "description": d}
                    for g, f, t, d in _CANONICAL_FIELDS
                ],
            },
        )


# ── Data tools ────────────────────────────────────────────────────────────────


class FindRecordsTool(EAGBaseTool):
    """Filter/sort the corpus on canonical fields."""

    name: str = "find_records"
    description: str = (
        "Search the record corpus. All filters are AND-ed on canonical fields "
        '(see get_schema), e.g. find_records(filters=[{"field": "town", "op": "==", '
        '"value": "Durham"}, {"field": "diag_label", "op": "==", "value": '
        '"Rheumatoid arthritis"}]). Ops: == != contains starts_with < <= > >= '
        "(ISO dates compare correctly). ANY-list fields (allergen, med_name, "
        "contract_ref, ...) match if any item matches. sort_by + order solve "
        "superlatives (e.g. sort_by='dob', order='asc', limit=1 → oldest). "
        "Returns matching record IDs; the total count is always reported."
    )
    args_schema: type[BaseModel] = _FindRecordsInput
    exposure_type: str = "data"

    @staticmethod
    def _norm_filters(filters: Any) -> list[dict]:
        """Accept both dicts and pydantic _FilterCond instances (langchain
        validates args into models; the EAG middleware passes plain dicts)."""
        out = []
        for f in filters or []:
            if hasattr(f, "model_dump"):
                f = f.model_dump()
            out.append(dict(f))
        return out

    def validate(self, **kwargs) -> tuple[bool, str]:
        for f in self._norm_filters(kwargs.get("filters")):
            if f.get("field") not in _KNOWN_FIELDS:
                return False, (
                    f"Unknown filter field '{f.get('field')}' — see get_schema for "
                    "the canonical field table"
                )
            if f.get("op") not in _KNOWN_OPS:
                return False, f"op must be one of {list(_KNOWN_OPS)}"
        fmt = kwargs.get("format")
        if fmt is not None and str(fmt).upper() not in _FORMATS:
            return False, f"format must be one of {list(_FORMATS)} or omitted"
        if kwargs.get("sort_by") is not None and kwargs["sort_by"] not in _KNOWN_FIELDS:
            return False, f"Unknown sort_by field '{kwargs.get('sort_by')}' — see get_schema"
        if str(kwargs.get("order", "asc")).lower() not in ("asc", "desc"):
            return False, "order must be 'asc' or 'desc'"
        for fld in kwargs.get("fields") or []:
            if fld not in _KNOWN_FIELDS:
                return False, f"Unknown fields entry '{fld}' — see get_schema"
        return True, ""

    def execute(self, **kwargs) -> ToolResult:
        corpus = _load_corpus()
        filters = self._norm_filters(kwargs.get("filters"))
        fmt = str(kwargs["format"]).upper() if kwargs.get("format") else None
        sort_by = kwargs.get("sort_by")
        order = str(kwargs.get("order", "asc")).lower()
        limit = int(kwargs.get("limit", 20))
        req_fields = kwargs.get("fields")

        recs = corpus.norm
        if fmt:
            recs = [r for r in recs if r.get("format") == fmt]
        for cond in filters:
            recs = [r for r in recs if _record_matches(r, cond)]
        total = len(recs)
        if sort_by:
            recs = _sort_records(recs, sort_by, order)
        shown = recs[:limit]

        if req_fields:
            cols = ["record_id"] + [f for f in req_fields if f != "record_id"]
        else:
            cols = ["record_id", "format", "name"]
            cols += [c["field"] for c in filters if c["field"] not in cols]
            if sort_by and sort_by not in cols:
                cols.append(sort_by)
        cols = cols[:_MAX_FIELDS_SHOWN]

        header = " | ".join(cols)
        rows = [
            " | ".join("; ".join(_display(v) for v in _field_values(r, c)) for c in cols)
            for r in shown
        ]
        note = f" — showing first {len(shown)}" if total > len(shown) else ""
        out = f"{total} match(es){note}\n{header}\n" + "\n".join(rows)

        return ToolResult(
            output=out,
            row_count=total,
            data={
                "filters": [{"field": c["field"], "op": c["op"]} for c in filters],
                "format": fmt,
                "sort_by": sort_by,
                "order": order,
                "match_count": total,
                "matches": [dict(r) for r in shown],
            },
        )

    def generate_metadata(self, result: ToolResult) -> str:
        if result.error:
            return f"[EAG] find_records failed — {result.output}"
        d = result.data or {}
        matches = d.get("matches", [])
        total = d.get("match_count", len(matches))
        flt = "; ".join(
            f"{f['field']} {f['op']} <withheld>" for f in d.get("filters", [])
        ) or "no filters"
        scope = f"; format {d['format']}" if d.get("format") else ""
        lines = [f"[EAG] find_records → {total} match(es) on {flt}{scope}."]
        if matches:
            ids = ", ".join(str(m.get("record_id")) for m in matches)
            ref_note = "" if len(matches) == total else \
                " (only these are referenceable)"
            lines.append(f"record IDs ({len(matches)} of {total}){ref_note}: {ids}")
            lines.extend(_match_shape_blocks(matches))
            lines.append(
                "Values withheld. Reference match i (0 = first listed) as "
                "{{ find_records.matches[i].<path> }} using the shapes above — "
                "lists index as [j], e.g. "
                "{{ find_records.matches[0].medications[0].name }}. Keys "
                "omitted from a shape are empty/absent on that format."
            )
        else:
            lines.append("record IDs: none.")
        return "\n".join(lines)


class GetRecordTool(EAGBaseTool):
    """Fetch one raw record by ID (optional path trims the display)."""

    name: str = "get_record"
    description: str = (
        "Fetch one raw record by record ID, e.g. get_record(record_id="
        "'REC-00007380'). Optional path trims the displayed output, e.g. "
        "path='medications[0]' or 'PID.Address.town'. Raw structures are "
        "format-specific (see get_schema(format=...)). Placeholder references "
        "always navigate the FULL record from its root."
    )
    args_schema: type[BaseModel] = _GetRecordInput
    exposure_type: str = "data"
    _view: Any = PrivateAttr(default=None)
    _last_id: str = PrivateAttr(default="")

    def execute(self, **kwargs) -> ToolResult:
        corpus = _load_corpus()
        rid = str(kwargs.get("record_id", "")).strip()
        raw = corpus.raw_by_id.get(rid)
        if raw is None:
            return ToolResult(
                output=f"Unknown record_id '{rid}' — use find_records to locate records first.",
                error="unknown record",
            )
        path = kwargs.get("path")
        view = raw
        if path:
            view, err = _walk_path(raw, str(path))
            if err is not None:
                return ToolResult(output=f"Bad path '{path}': {err}", error="bad path")
        self._view = view
        self._last_id = rid

        fmt = raw.get("format", "?")
        dump = json.dumps(view, ensure_ascii=False, indent=1)
        if len(dump) > _MAX_JSON_CHARS:
            dump = dump[:_MAX_JSON_CHARS] + "\n… (truncated — narrow with path=)"
        out = f"record {rid} ({fmt}):\n{dump}"
        return ToolResult(
            output=out,
            row_count=1,
            data=raw,  # artifact = FULL raw record; placeholders navigate from root
        )

    def generate_metadata(self, result: ToolResult) -> str:
        if result.error:
            return f"[EAG] get_record failed — {result.output}"
        view = self._view if self._view is not None else (result.data or {})
        return (
            f"[EAG] get_record {self._last_id} — structure only (values withheld):\n"
            f"{_render_shape(view)}"
        )


# ── Evaluation tool ───────────────────────────────────────────────────────────


class SubmitAnswerTool(EAGBaseTool):
    """Submit the final answer for evaluation."""

    name: str = "submit_answer"
    description: str = (
        "Submit your final answer. Call this when you know the answer (a "
        "literal in baseline mode; a {{ placeholder }} reference in EAG "
        'mode, e.g. "{{ join_items.result }}"). After calling submit_answer, '
        "respond with a brief text answer — do not call any more tools."
    )
    args_schema: type[BaseModel] = _SubmitAnswerInput
    exposure_type: str = "evaluation"

    def execute(self, **kwargs) -> ToolResult:
        answer = kwargs.get("answer")
        answer = answer if isinstance(answer, str) else str(answer)
        return ToolResult(
            output=(
                "Answer submitted successfully. It will be evaluated against "
                "the gold answer."
            ),
            data={"submitted": answer},
        )


# ── Factory ───────────────────────────────────────────────────────────────────


def make_lindisfarne_tools(eag_mode: bool = False) -> list[BaseTool]:
    """The 4 lindisfarne-domain tools (get_schema, find_records, get_record,
    submit_answer). The dataset adapter adds the 7 evaluator compute tools."""
    return [
        GetSchemaTool(eag_mode=eag_mode),
        FindRecordsTool(eag_mode=eag_mode),
        GetRecordTool(eag_mode=eag_mode),
        SubmitAnswerTool(eag_mode=eag_mode),
    ]
