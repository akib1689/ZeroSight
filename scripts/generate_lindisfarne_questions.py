"""Generate the 140-question curated seed set for the Lindisfarne benchmark.

Questions are computed against the Lindisfarne M1 full-tier corpus (chunks
records_0001-0002.jsonl, 20,000 synthetic NHS records), so every gold answer
is derived from the data and machine-verified at generation time:

- point lookups resolve to exactly one record (NHS number, name+DOB, record ID)
- filter/aggregate golds are recomputed over the full 20k corpus
- superlatives assert a unique winner (no ties at the extreme)
- unanswerable questions assert ZERO matching records corpus-wide
- multihop questions span 2-5 hops (filter/superlative -> list selection ->
  attribute -> count), with difficulty rising gradually

Design constraints (structured-tool answerability, EAG-safe):
- no dosage/route/clinical-plausibility questions (synthetic data, not validated)
- no medication code<->name translation (generator maps them inconsistently)
- diagnosis code<->label IS 1:1 across formats (verified) and is used
- answers: strings as stored; lists comma-separated (names alphabetical,
  record IDs ascending); dates YYYY-MM-DD; booleans true/false; numbers plain;
  verified-empty questions answer "no matching records"

Output: benchmarks/lindisfarne/tasks.json
Data:   data/lindisfarne/full/records_000{1,2}.jsonl (auto-download if missing)
"""

import argparse
import datetime as dt
import json
import random
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "lindisfarne" / "full"
CHUNKS = ["records_0001.jsonl", "records_0002.jsonl"]
OUT_PATH = BASE_DIR / "benchmarks" / "lindisfarne" / "tasks.json"
HF_BASE = "https://huggingface.co/datasets/Toridion/lindisfarne-m1/resolve/main/full"

EXPECTED = {
    "lookup": 20, "filter": 15, "multihop": 20, "aggregate": 15,
    "superlative": 15, "temporal": 15, "cross_format": 10,
    "procurement": 10, "unanswerable": 10, "safety": 10,
}
SEED = 20260907


def _d(x):
    return x if isinstance(x, dict) else {}


def _l(x):
    return x if isinstance(x, list) else []


def dpart(ts):
    return str(ts)[:10] if ts else None


def normalize(r):
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
            name=p.get("name"), dob=dpart(p.get("dob")), nhs_number=p.get("nhs_number"),
            gender=p.get("gender"), town=_d(p.get("address")).get("town"),
            postcode=_d(p.get("address")).get("postcode"), gp=p.get("gp"),
            trust=p.get("trust"), encounter_type=en.get("type"),
            encounter_date=dpart(en.get("date")), ward=en.get("ward"),
            clinician=en.get("clinician"), specialty=en.get("specialty"),
            diag_code=dg.get("code"), diag_label=dg.get("description"),
            diag_severity=dg.get("severity"), diag_status=dg.get("status"),
            diag_onset=dpart(dg.get("onset")),
        )
        n["medications"] = [
            {"name": m.get("name"), "status": m.get("status"), "start_date": dpart(m.get("startDate"))}
            for m in _l(r.get("medications")) if isinstance(m, dict)]
        n["labs"] = [
            {"test": lab.get("test"), "value": lab.get("value"), "date": dpart(lab.get("date")),
             "status": lab.get("status"), "performer": lab.get("performer")}
            for lab in _l(r.get("labs")) if isinstance(lab, dict)]
        n["allergies"] = [
            {"allergen": a.get("allergen"), "reaction": a.get("reaction"),
             "severity": a.get("severity"), "confirmed": a.get("confirmed")}
            for a in _l(r.get("allergies")) if isinstance(a, dict)]
        n["procedures"] = [
            {"name": pr.get("name"), "date": dpart(pr.get("date")), "status": pr.get("status")}
            for pr in _l(r.get("procedures")) if isinstance(pr, dict)]
        n["alerts"] = [a for a in _l(r.get("alerts")) if isinstance(a, str)]
    elif fmt == "HL7":
        pid, pv1, dg1 = _d(r.get("PID")), _d(r.get("PV1")), _d(r.get("DG1"))
        n["record_id"] = r.get("HL7MessageID")
        n.update(
            name=pid.get("PatientName"), dob=dpart(pid.get("DateOfBirth")),
            nhs_number=pid.get("NHSNumber"), gender=pid.get("Gender"),
            town=_d(pid.get("Address")).get("town"), postcode=_d(pid.get("Address")).get("postcode"),
            trust=_d(r.get("MSH")).get("SendingFacility"), encounter_type=pv1.get("VisitType"),
            encounter_date=dpart(pv1.get("VisitDate")), ward=pv1.get("Ward"),
            clinician=pv1.get("Consultant"), diag_code=dg1.get("DiagnosisCode"),
            diag_label=dg1.get("DiagnosisDesc"), diag_type=dg1.get("DiagnosisType"),
        )
        n["medications"] = [
            {"name": m.get("MedicationName"), "prescribed_by": m.get("PrescribedBy"),
             "start_date": None}
            for m in _l(r.get("RX1")) if isinstance(m, dict)]
        n["labs"] = [
            {"test": lab.get("test"), "value": lab.get("value"), "date": dpart(lab.get("date")),
             "flag": lab.get("flag")}
            for lab in _l(r.get("OBX")) if isinstance(lab, dict)]
        n["allergies"] = [
            {"allergen": a.get("allergen"), "reaction": a.get("reaction"),
             "severity": a.get("severity"), "confirmed": a.get("confirmed")}
            for a in _l(r.get("AL1")) if isinstance(a, dict)]
    elif fmt == "SNOMED":
        pi, en, sc = _d(r.get("patient_info")), _d(r.get("clinical_encounter")), _d(r.get("SNOMED_codes"))
        n["record_id"] = r.get("record_id")
        n.update(
            name=pi.get("full_name"), dob=dpart(pi.get("dob")), nhs_number=pi.get("nhs_number"),
            town=_d(pi.get("address")).get("town"), postcode=_d(pi.get("address")).get("postcode"),
            gp=pi.get("gp_surgery"), encounter_type=en.get("type"),
            encounter_date=dpart(en.get("date")), ward=en.get("ward"), clinician=en.get("clinician"),
            diag_code=sc.get("diagnosis"), diag_label=sc.get("diag_label"),
            lab_test=sc.get("lab"), med_code=sc.get("medication"), proc_code=sc.get("procedure"),
        )
        n["alerts"] = [a for a in _l(r.get("alerts")) if isinstance(a, str)]
    elif fmt == "GHX":
        sup, tr, pr, wp = _d(r.get("SUPPLIER")), _d(r.get("TRUST")), _d(r.get("PROJECT")), _d(r.get("WORKPLAN"))
        n["record_id"] = r.get("record_id")
        n["ghx"] = {
            "company": sup.get("CompanyName"), "contact_name": sup.get("ContactName"),
            "contact_role": sup.get("ContactRole"), "trust_name": tr.get("TrustName"),
            "procurement_ref": tr.get("ProcurementRef"), "cohort_size": tr.get("CohortSize"),
            "project_ref": pr.get("ProjectRef"), "bid_date": dpart(pr.get("BidDate")),
            "phase": pr.get("Phase"), "lead_clinician": pr.get("LeadClinician"),
            "workplan_ref": wp.get("WorkplanRef"), "milestones": wp.get("Milestones"),
            "planned_golive": dpart(wp.get("PlannedGoLive")),
            "contracts": [
                {"ref": c.get("ContractRef"), "date": dpart(c.get("ContractDate")),
                 "value": c.get("Value"), "duration": c.get("Duration"),
                 "sla": c.get("SLA"), "status": c.get("Status")}
                for c in _l(r.get("CONTRACT")) if isinstance(c, dict)],
        }
    else:
        n["record_id"] = None
    return n


def ensure_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for chunk in CHUNKS:
        path = DATA_DIR / chunk
        if path.exists():
            continue
        url = f"{HF_BASE}/{chunk}"
        print(f"downloading {url} ...")
        tmp = path.with_suffix(".tmp")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(path)


def load_corpus():
    records = []
    for chunk in CHUNKS:
        with open(DATA_DIR / chunk, encoding="utf-8") as f:
            for line in f:
                records.append(normalize(json.loads(line)))
    return records


class Corpus:
    def __init__(self, records):
        self.records = records
        self.by_nhs = {}
        self.by_nd = {}
        self.by_rec = {}
        for r in records:
            if r["nhs_number"]:
                self.by_nhs[r["nhs_number"]] = r
            if r["name"] and r["dob"]:
                self.by_nd[(r["name"], r["dob"])] = r
            if r["record_id"]:
                self.by_rec[r["record_id"]] = r
        if len(self.by_rec) != len(records):
            raise AssertionError("record ids not unique")
        if len(self.by_nhs) != sum(1 for r in records if r["nhs_number"]):
            raise AssertionError("nhs numbers not unique")

    def nhs(self, nhs):
        r = self.by_nhs.get(nhs)
        assert r is not None, f"nhs {nhs} not found"
        return r

    def nd(self, name, dob):
        r = self.by_nd.get((name, dob))
        assert r is not None, f"name+dob {name}/{dob} not found"
        return r

    def rec(self, rid):
        r = self.by_rec.get(rid)
        assert r is not None, f"record {rid} not found"
        return r

    def where(self, **kw):
        out = self.records
        for k, v in kw.items():
            if k == "allergen":
                out = [r for r in out if any(a["allergen"] == v for a in r["allergies"])]
            elif k == "lab_flag":
                out = [r for r in out if any(lab.get("flag") == v for lab in r["labs"])]
            elif k == "alert":
                out = [r for r in out if v in r["alerts"]]
            else:
                out = [r for r in out if r.get(k) == v]
        return out

    def key_phrase(self, r, kind):
        if kind == "rec":
            return f"record {r['record_id']}"
        if kind == "nhs":
            return f"the patient with NHS number {r['nhs_number']}"
        return f"{r['name']} (date of birth {r['dob']})"


def gold_names(recs):
    return ", ".join(sorted(r["name"] for r in recs))


def gold_ids(recs):
    return ", ".join(sorted(r["record_id"] for r in recs))


def gender_phrase(g):
    if g == "Prefer not to say":
        return "patients whose recorded gender is 'Prefer not to say'"
    return f"{g.lower()} patients"


def pick(rng, pool, n, constraint=lambda x: True, attempts=4000):
    pool = [x for x in pool if constraint(x)]
    chosen, seen = [], set()
    while len(chosen) < n and attempts > 0:
        attempts -= 1
        item = rng.choice(pool)
        key = json.dumps(item, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            chosen.append(item)
    if len(chosen) < n:
        raise AssertionError(f"pool exhausted: need {n}, got {len(chosen)}")
    return chosen


def build_lookup(C, rng):
    tasks = []
    fhir = [r for r in C.records if r["format"] == "FHIR" and r["nhs_number"] and r["gp"]]
    hl7 = [r for r in C.records if r["format"] == "HL7" and r["nhs_number"]]
    snomed = [r for r in C.records if r["format"] == "SNOMED" and r["nhs_number"] and r["gp"]]

    specs = [
        ("gp", "At which GP surgery is {key} registered?", fhir),
        ("ward", "In which ward was {key} seen?", fhir),
        ("specialty", "Under which specialty was {key} seen?", fhir),
        ("clinician", "Who was the treating clinician for {key}?", fhir),
        ("diag_severity", "What is the severity of the diagnosis recorded for {key}?", fhir),
        ("lab_first", "Which lab test was recorded for {key}?", fhir),
        ("trust", "Which NHS trust sent the admission message for {key}?", hl7),
        ("clinician", "Who was the consultant responsible for {key}?", hl7),
        ("diag_type", "What diagnosis type is recorded for {key}?", hl7),
        ("med_first", "Which medication was prescribed for {key}?", hl7),
        ("lab_test", "Which lab test is recorded for {key}?", snomed),
        ("med_code", "What SNOMED medication code is recorded for {key}?", snomed),
    ]
    field_rotation = [s for s in specs for _ in (0,)] * 2
    key_plan = (["rec"] * 5) + (["nhs"] * 7) + (["nd"] * 8)
    rng.shuffle(key_plan)
    used_recs = set()
    for i in range(20):
        field, template, pool = field_rotation[i % len(field_rotation)]
        for _ in range(500):
            r = rng.choice(pool)
            if r["record_id"] in used_recs:
                continue
            if field == "lab_first":
                if len(r["labs"]) != 1:
                    continue
                gold, val = r["labs"][0]["test"], r["labs"][0]["test"]
            elif field == "med_first":
                if len(r["medications"]) != 1:
                    continue
                gold, val = r["medications"][0]["name"], r["medications"][0]["name"]
            else:
                val = r.get(field)
                if not val:
                    continue
                gold = val
            used_recs.add(r["record_id"])
            kind = key_plan[i]
            q = template.format(key=C.key_phrase(r, kind))
            tasks.append({
                "category": "lookup", "question": q, "gold_answer": str(gold),
                "metadata": {"record_ids": [r["record_id"]], "format": r["format"],
                             "field": field, "key": kind},
            })
            break
        else:
            raise AssertionError(f"lookup spec {field} unsatisfiable")
    return tasks


def build_filter(C, rng):
    tasks = []
    td_counts = Counter((r["town"], r["diag_label"]) for r in C.records if r["town"] and r["diag_label"])
    td_pool = [{"town": t, "diag": d, "n": n} for (t, d), n in td_counts.items() if 4 <= n <= 8]
    for spec in pick(rng, td_pool, 6):
        recs = C.where(town=spec["town"], diag_label=spec["diag"])
        tasks.append({
            "category": "filter",
            "question": f"List the names of the patients in {spec['town']} "
                        f"diagnosed with {spec['diag']}.",
            "gold_answer": gold_names(recs),
            "metadata": {"record_ids": sorted(r["record_id"] for r in recs),
                         "formats": sorted({r["format"] for r in recs}), "size": len(recs)},
        })
    for spec in pick(rng, td_pool, 2):
        recs = C.where(town=spec["town"], diag_label=spec["diag"])
        tasks.append({
            "category": "filter",
            "question": f"List the record IDs of the {spec['diag']} patients in {spec['town']}.",
            "gold_answer": gold_ids(recs),
            "metadata": {"record_ids": sorted(r["record_id"] for r in recs), "size": len(recs)},
        })
    dtg_counts = Counter(
        (r["town"], r["diag_label"], r["gender"])
        for r in C.records if r["town"] and r["diag_label"] and r["gender"])
    dtg_pool = [k for k, v in dtg_counts.items() if 2 <= v <= 4]
    for town, diag, gender in pick(rng, dtg_pool, 2):
        recs = C.where(town=town, diag_label=diag, gender=gender)
        tasks.append({
            "category": "filter",
            "question": f"Which {gender_phrase(gender)} in {town} are diagnosed with "
                        f"{diag}?",
            "gold_answer": gold_names(recs),
            "metadata": {"record_ids": sorted(r["record_id"] for r in recs), "size": len(recs)},
        })
    ats_counts = Counter(
        (r["town"], a["allergen"], a["severity"])
        for r in C.records for a in r["allergies"] if r["town"])
    ats_pool = [k for k, v in ats_counts.items() if 4 <= v <= 12]
    for town, allergen, severity in pick(rng, ats_pool, 2):
        recs = [r for r in C.records
                if r["town"] == town
                and any(a["allergen"] == allergen and a["severity"] == severity
                       for a in r["allergies"])]
        tasks.append({
            "category": "filter",
            "question": f"List the patients in {town} with a {severity.lower()} "
                        f"{allergen} allergy.",
            "gold_answer": gold_names(recs),
            "metadata": {"record_ids": sorted(r["record_id"] for r in recs), "size": len(recs)},
        })
    tey_counts = Counter(
        (r["town"], r["encounter_type"], r["encounter_date"][:4])
        for r in C.records if r["town"] and r["encounter_type"] and r["encounter_date"])
    tey_pool = [k for k, v in tey_counts.items() if 2 <= v <= 8]
    town, etype, year = pick(rng, tey_pool, 1)[0]
    recs = [r for r in C.records if r["town"] == town and r["encounter_type"] == etype
            and (r["encounter_date"] or "")[:4] == year]
    tasks.append({
        "category": "filter",
        "question": f"Which patients in {town} had a {etype.lower()} encounter in {year}?",
        "gold_answer": gold_names(recs),
        "metadata": {"record_ids": sorted(r["record_id"] for r in recs), "size": len(recs)},
    })
    flag_counts = Counter(
        (lab.get("flag"), lab["test"])
        for r in C.records if r["format"] == "HL7" for lab in r["labs"])
    flag_pool = [k for k, v in flag_counts.items()
                 if k[0] in ("Critical High", "Critical Low") and 5 <= v <= 15]
    flag, test = pick(rng, flag_pool, 1)[0]
    recs = [r for r in C.records if r["format"] == "HL7"
            and any(lab.get("flag") == flag and lab["test"] == test for lab in r["labs"])]
    tasks.append({
        "category": "filter",
        "question": f"Which patients had a {flag.lower()} flag on their {test} lab result?",
        "gold_answer": gold_names(recs),
        "metadata": {"record_ids": sorted(r["record_id"] for r in recs), "size": len(recs)},
    })
    we_counts = Counter(
        (r["ward"], r["encounter_type"]) for r in C.records
        if r["ward"] and r["encounter_type"])
    we_pool = [k for k, v in we_counts.items() if 5 <= v <= 15]
    ward, etype = pick(rng, we_pool, 1)[0]
    recs = C.where(ward=ward, encounter_type=etype)
    tasks.append({
        "category": "filter",
        "question": f"Which patients were on {ward} for a {etype.lower()} visit?",
        "gold_answer": gold_names(recs),
        "metadata": {"record_ids": sorted(r["record_id"] for r in recs), "size": len(recs)},
    })
    return tasks


def extreme_in_group(C, town, diag, fmt=None, old=True, need=None):
    recs = C.where(town=town, diag_label=diag)
    if fmt:
        recs = [r for r in recs if r["format"] == fmt]
    if len(recs) < 3:
        return None
    dobs = sorted(r["dob"] for r in recs if r["dob"])
    if dobs[0] == dobs[1] or dobs[-1] == dobs[-2]:
        return None
    rec = min(recs, key=lambda r: r["dob"]) if old else max(recs, key=lambda r: r["dob"])
    if need and not need(rec):
        return None
    return rec


def sel_allergy(rec, allergen):
    hits = [a for a in rec["allergies"] if a["allergen"] == allergen]
    assert len(hits) == 1, f"allergen {allergen} x{len(hits)} in {rec['record_id']}"
    return hits[0]


def sel_lab(rec, test):
    hits = [lab for lab in rec["labs"] if lab["test"] == test]
    assert len(hits) == 1, f"lab {test} x{len(hits)} in {rec['record_id']}"
    return hits[0]


def sel_proc(rec, name):
    hits = [p for p in rec["procedures"] if p["name"] == name]
    assert len(hits) == 1, f"proc {name} x{len(hits)} in {rec['record_id']}"
    return hits[0]


def sel_med(rec, name):
    hits = [m for m in rec["medications"] if m["name"] == name]
    assert len(hits) == 1, f"med {name} x{len(hits)} in {rec['record_id']}"
    return hits[0]


def build_multihop(C, rng):
    tasks = []

    def add(hops, q, gold, rec, note):
        tasks.append({
            "category": "multihop", "question": q, "gold_answer": str(gold),
            "metadata": {"record_ids": [rec["record_id"]], "hops": hops,
                         "format": rec["format"], "evidence": note},
        })

    fhir1 = [r for r in C.records if r["format"] == "FHIR" and r["nhs_number"]
             and len(r["medications"]) == 1 and len(r["labs"]) == 1]
    hl7m = [r for r in C.records if r["format"] == "HL7" and r["nhs_number"]
            and len(r["medications"]) == 1]
    snomed = [r for r in C.records if r["format"] == "SNOMED" and r["nhs_number"]]

    r = pick(rng, snomed, 1)[0]
    add(2, f"What is the SNOMED diagnosis code for {C.key_phrase(r, 'nd')}?",
        r["diag_code"], r, "name+dob -> record -> diag_code")

    r = pick(rng, fhir1, 1)[0]
    add(2, f"Which medication is prescribed for {C.key_phrase(r, 'nhs')}?",
        r["medications"][0]["name"], r, "nhs -> record -> medication name")

    r = pick(rng, hl7m, 1)[0]
    add(2, f"Who prescribed the medication for {C.key_phrase(r, 'rec')}?",
        r["medications"][0]["prescribed_by"], r, "record -> RX1 -> prescribed_by")

    r = pick(rng, fhir1, 1)[0]
    add(2, f"Which laboratory performed the lab test for {C.key_phrase(r, 'nd')}?",
        r["labs"][0]["performer"], r, "name+dob -> record -> lab performer")

    r = pick(rng, fhir1, 1)[0]
    add(2, f"On what date was the diagnosis for {C.key_phrase(r, 'nhs')} first recorded "
           f"(the onset date)?", r["diag_onset"], r, "nhs -> record -> onset date")

    al_rec = [r for r in C.records if r["nhs_number"] and r["allergies"]]
    for _ in range(500):
        r = rng.choice(al_rec)
        a = rng.choice(r["allergies"])
        key = rng.choice(["nhs", "nd", "rec"])
        if key == "nd" and not (r["name"] and r["dob"]):
            continue
        add(3, f"What is the severity of the {a['allergen']} allergy recorded for "
               f"{C.key_phrase(r, key)}?", a["severity"], r,
            f"key -> record -> allergy[{a['allergen']}] -> severity")
        break
    for _ in range(500):
        r = rng.choice(al_rec)
        a = rng.choice(r["allergies"])
        key = rng.choice(["nhs", "nd", "rec"])
        if key == "nd" and not (r["name"] and r["dob"]):
            continue
        add(3, f"What was the reaction to the {a['allergen']} allergy for "
               f"{C.key_phrase(r, key)}?", a["reaction"], r,
            f"key -> record -> allergy[{a['allergen']}] -> reaction")
        break
    for _ in range(500):
        r = rng.choice([x for x in C.records if x["format"] == "FHIR" and x["nhs_number"] and x["labs"]])
        lab = rng.choice(r["labs"])
        add(3, f"What is the status of the {lab['test']} lab result for "
               f"{C.key_phrase(r, 'nd')}?", lab["status"], r,
            f"name+dob -> record -> lab[{lab['test']}] -> status")
        break
    for _ in range(500):
        r = rng.choice([x for x in C.records if x["format"] == "HL7" and x["labs"]])
        lab = rng.choice(r["labs"])
        if not lab.get("flag"):
            continue
        add(3, f"What flag was raised on the {lab['test']} result in {C.key_phrase(r, 'rec')}?",
             lab["flag"], r, f"record -> OBX[{lab['test']}] -> flag")
        break
    for _ in range(500):
        r = rng.choice([x for x in C.records if x["format"] == "FHIR" and x["nhs_number"] and x["labs"]])
        lab = rng.choice(r["labs"])
        add(3, f"Which laboratory performed the {lab['test']} test for {C.key_phrase(r, 'nhs')}?",
             lab["performer"], r, f"nhs -> record -> lab[{lab['test']}] -> performer")
        break
    for _ in range(500):
        r = rng.choice([x for x in C.records if x["format"] == "FHIR" and x["nhs_number"] and x["procedures"]])
        p = rng.choice(r["procedures"])
        add(3, f"What is the status of the {p['name']} procedure for {C.key_phrase(r, 'nhs')}?",
             p["status"], r, f"nhs -> record -> procedure[{p['name']}] -> status")
        break

    td_counts = Counter((r["town"], r["diag_label"]) for r in C.records if r["town"] and r["diag_label"])
    td_pool = [k for k, v in td_counts.items() if v >= 6]

    def group_pick(fmt=None, need=None, avoid=()):
        for _ in range(3000):
            town, diag = rng.choice(td_pool)
            if (town, diag) in avoid:
                continue
            use_old = rng.choice([True, False])
            rec = extreme_in_group(C, town, diag, fmt=fmt, old=use_old, need=need)
            if rec is not None:
                return town, diag, rec, ("oldest" if use_old else "youngest")
        raise AssertionError("no 4-hop group found")

    town, diag, rec, who = group_pick(
        need=lambda r: any(a["confirmed"] == "Yes" for a in r["allergies"]) and r["allergies"])
    a = rng.choice([x for x in rec["allergies"] if x["confirmed"] == "Yes"])
    add(4, f"What is the severity of the {a['allergen']} allergy for the {who} patient with "
           f"{diag} in {town}?", a["severity"], rec,
        f"town+diag filter -> {who} -> allergy[{a['allergen']}] -> severity")

    town, diag, rec, who = group_pick(
        need=lambda r: r["labs"] and all(lab.get("performer") for lab in r["labs"]))
    lab = rng.choice(rec["labs"])
    add(4, f"Which laboratory performed the {lab['test']} test for the {who} patient with "
           f"{diag} in {town}?", lab["performer"], rec,
        f"town+diag filter -> {who} -> lab[{lab['test']}] -> performer")

    town, diag, rec, who = group_pick(need=lambda r: r["procedures"])
    p = rng.choice(rec["procedures"])
    add(4, f"What is the status of the {p['name']} procedure for the {who} patient with "
           f"{diag} in {town}?", p["status"], rec,
        f"town+diag filter -> {who} -> procedure[{p['name']}] -> status")

    town, diag, rec, who = group_pick(fmt="SNOMED", need=lambda r: r.get("diag_code"))
    add(4, f"What is the SNOMED diagnosis code for the {who} SNOMED-format patient with "
           f"{diag} in {town}?", rec["diag_code"], rec,
        "town+diag+format filter -> extreme -> diag_code")

    town, diag, rec, who = group_pick(
        fmt="HL7", need=lambda r: len(r["medications"]) == 1 and r["medications"][0].get("prescribed_by"))
    m = rec["medications"][0]
    add(4, f"Who prescribed {m['name']} for the {who} HL7 patient with {diag} in {town}?",
        m["prescribed_by"], rec, f"town+diag+format filter -> {who} -> RX1 -> prescribed_by")

    town, diag, rec, who = group_pick(
        need=lambda r: r["allergies"] and any(a["confirmed"] == "No" for a in r["allergies"]))
    n_unconf = sum(1 for a in rec["allergies"] if a["confirmed"] == "No")
    add(5, f"How many of the {who} {diag} patient's allergies in {town} are unconfirmed?",
        n_unconf, rec,
        f"town+diag filter -> {who} -> allergies -> confirmed=No -> count (={n_unconf})")

    town, diag, rec, who = group_pick(
        need=lambda r: r["allergies"] and any(a["confirmed"] == "Yes" for a in r["allergies"]))
    n_conf = sum(1 for a in rec["allergies"] if a["confirmed"] == "Yes")
    add(5, f"How many confirmed allergies does the {who} patient with {diag} in {town} have?",
        n_conf, rec, f"town+diag filter -> {who} -> allergies -> confirmed=Yes -> count (={n_conf})")

    town, diag, rec, who = group_pick(
        need=lambda r: r["allergies"] and any(a["confirmed"] == "Yes" for a in r["allergies"]))
    confirmed = sorted(a["allergen"] for a in rec["allergies"] if a["confirmed"] == "Yes")
    add(5, f"List the confirmed allergens for the {who} patient with {diag} in {town}.",
        ", ".join(confirmed), rec,
        f"town+diag filter -> {who} -> allergies -> confirmed=Yes -> names ({len(confirmed)})")

    al_town_pool = sorted({(r["town"]) for r in C.records if r["town"]})
    for _ in range(3000):
        town = rng.choice(al_town_pool)
        allergen = rng.choice(sorted({a["allergen"] for r in C.records for a in r["allergies"]}))
        cands = [r for r in C.where(town=town, allergen=allergen)
                 if r["gp"] and r["dob"]
                 and any(a["allergen"] == allergen and a["confirmed"] == "Yes" for a in r["allergies"])]
        if len(cands) < 3:
            continue
        dobs = sorted(r["dob"] for r in cands)
        if dobs[0] == dobs[1]:
            continue
        rec = min(cands, key=lambda r: r["dob"])
        add(5, f"At which GP surgery is the oldest patient in {town} with a confirmed "
               f"{allergen} allergy registered?", rec["gp"], rec,
            "town+allergen+confirmed filter -> oldest -> gp")
        break
    else:
        raise AssertionError("no gp 5-hop found")

    hop_counts = Counter(t["metadata"]["hops"] for t in tasks)
    assert hop_counts == Counter({2: 5, 3: 6, 4: 5, 5: 4}), hop_counts
    return tasks


def build_aggregate(C, rng):
    tasks = []
    td_counts = Counter((r["town"], r["diag_label"]) for r in C.records if r["town"] and r["diag_label"])
    td_pool = [k for k, v in td_counts.items() if 4 <= v <= 20]
    for town, diag in pick(rng, td_pool, 4):
        n = len(C.where(town=town, diag_label=diag))
        tasks.append({
            "category": "aggregate",
            "question": f"How many patients in {town} are diagnosed with {diag}?",
            "gold_answer": str(n),
            "metadata": {"town": town, "diag": diag, "count": n},
        })
    dg_counts = Counter((r["diag_label"], r["gender"]) for r in C.records if r["diag_label"] and r["gender"])
    dg_pool = [k for k, v in dg_counts.items() if v >= 4]
    for diag, gender in pick(rng, dg_pool, 3):
        n = len(C.where(diag_label=diag, gender=gender))
        tasks.append({
            "category": "aggregate",
            "question": f"How many {gender_phrase(gender)} have a diagnosis of {diag}?",
            "gold_answer": str(n),
            "metadata": {"diag": diag, "gender": gender, "count": n},
        })
    ty_counts = Counter(
        (r["encounter_type"], r["encounter_date"][:4])
        for r in C.records if r["encounter_type"] and r["encounter_date"])
    ty_pool = [k for k, v in ty_counts.items() if v >= 4]
    for etype, year in pick(rng, ty_pool, 2):
        recs = [r for r in C.records
                if r["encounter_type"] == etype and (r["encounter_date"] or "")[:4] == year]
        tasks.append({
            "category": "aggregate",
            "question": f"How many {etype.lower()} encounters took place in {year}?",
            "gold_answer": str(len(recs)),
            "metadata": {"encounter_type": etype, "year": year, "count": len(recs)},
        })
    ghx = [r["ghx"] for r in C.records if r["format"] == "GHX"]
    phase = rng.choice(sorted({g["phase"] for g in ghx}))
    tasks.append({
        "category": "aggregate",
        "question": f"How many GHX procurement records are in the {phase} phase?",
        "gold_answer": str(sum(1 for g in ghx if g["phase"] == phase)),
        "metadata": {"phase": phase},
    })
    status = rng.choice(sorted({c["status"] for g in ghx for c in g["contracts"]}))
    tasks.append({
        "category": "aggregate",
        "question": f"How many GHX contracts have the status '{status}'?",
        "gold_answer": str(sum(1 for g in ghx for c in g["contracts"] if c["status"] == status)),
        "metadata": {"contract_status": status},
    })
    asc_counts = Counter(
        (a["allergen"], a["severity"], a["confirmed"])
        for r in C.records for a in r["allergies"])
    asc_pool = [
        {"a": al, "s": s} for (al, s, c), n in asc_counts.items()
        if c == "Yes" and s in ("Life-threatening", "Severe") and n >= 4]
    for spec in pick(rng, asc_pool, 2):
        allergen, severity = spec["a"], spec["s"]
        n = sum(1 for r in C.records
                if any(a["allergen"] == allergen and a["severity"] == severity
                       and a["confirmed"] == "Yes" for a in r["allergies"]))
        sev = "life-threatening" if severity == "Life-threatening" else severity.lower()
        tasks.append({
            "category": "aggregate",
            "question": f"How many patients have a confirmed {sev} {allergen} allergy?",
            "gold_answer": str(n),
            "metadata": {"allergen": allergen, "severity": severity, "count": n},
        })
    flag_counts = Counter(
        (lab.get("flag"), lab["test"])
        for r in C.records if r["format"] == "HL7" for lab in r["labs"])
    flag_pool = [
        {"f": f, "t": t} for (f, t), n in flag_counts.items()
        if f in ("Critical High", "Critical Low") and n >= 4]
    for spec in pick(rng, flag_pool, 2):
        flag, test = spec["f"], spec["t"]
        n = len([r for r in C.records if r["format"] == "HL7"
                 and any(lab.get("flag") == flag and lab["test"] == test for lab in r["labs"])])
        tasks.append({
            "category": "aggregate",
            "question": f"How many HL7 patients had a {flag.lower()} flag on their {test} "
                        f"lab result?",
            "gold_answer": str(n),
            "metadata": {"flag": flag, "test": test, "count": n},
        })
    return tasks


def build_superlative(C, rng):
    tasks = []
    groups = defaultdict(list)
    for r in C.records:
        if r["diag_label"] and r["dob"]:
            groups[r["diag_label"]].append(r)
    uniq_old, uniq_young = [], []
    for lbl, rs in groups.items():
        dobs = sorted(r["dob"] for r in rs)
        if len(dobs) >= 20 and dobs[0] != dobs[1]:
            uniq_old.append(lbl)
        if len(dobs) >= 20 and dobs[-1] != dobs[-2]:
            uniq_young.append(lbl)
    for lbl in pick(rng, uniq_old, 3):
        rec = min(groups[lbl], key=lambda r: r["dob"])
        tasks.append({
            "category": "superlative",
            "question": f"Who is the oldest patient diagnosed with {lbl}?",
            "gold_answer": rec["name"],
            "metadata": {"record_ids": [rec["record_id"]], "dob": rec["dob"], "diag": lbl},
        })
    for lbl in pick(rng, uniq_young, 3):
        rec = max(groups[lbl], key=lambda r: r["dob"])
        tasks.append({
            "category": "superlative",
            "question": f"Who is the youngest patient diagnosed with {lbl}?",
            "gold_answer": rec["name"],
            "metadata": {"record_ids": [rec["record_id"]], "dob": rec["dob"], "diag": lbl},
        })
    td_counts = Counter((r["town"], r["diag_label"]) for r in C.records if r["town"] and r["diag_label"])
    td_pool = [k for k, v in td_counts.items() if 4 <= v <= 12]
    picked = pick(rng, td_pool, 3)
    for i, (town, diag) in enumerate(picked):
        recs = C.where(town=town, diag_label=diag)
        dated = [r for r in recs if r["encounter_date"]]
        dates = sorted(r["encounter_date"] for r in dated)
        if dates[0] == dates[1] or dates[-1] == dates[-2]:
            raise AssertionError(f"encounter tie in {town}/{diag}")
        if i < 2:
            rec = min(dated, key=lambda r: r["encounter_date"])
            tasks.append({
                "category": "superlative",
                "question": f"Which patient with {diag} in {town} had the earliest recorded "
                            f"encounter?",
                "gold_answer": rec["name"],
                "metadata": {"record_ids": [rec["record_id"]],
                             "encounter_date": rec["encounter_date"]},
            })
        else:
            rec = max(dated, key=lambda r: r["encounter_date"])
            tasks.append({
                "category": "superlative",
                "question": f"Which patient with {diag} in {town} had the most recent recorded "
                            f"encounter?",
                "gold_answer": rec["name"],
                "metadata": {"record_ids": [rec["record_id"]],
                             "encounter_date": rec["encounter_date"]},
            })
    ghx = [r["ghx"] for r in C.records if r["format"] == "GHX"]
    cohorts = sorted(g["cohort_size"] or 0 for g in ghx)
    assert cohorts[-1] != cohorts[-2], "cohort max tie"
    gmax = max(ghx, key=lambda g: g["cohort_size"] or 0)
    tasks.append({
        "category": "superlative",
        "question": "Which NHS trust has the largest patient cohort size across all GHX "
                    "procurement records?",
        "gold_answer": gmax["trust_name"],
        "metadata": {"cohort_size": gmax["cohort_size"], "company": gmax["company"]},
    })

    def value_num(v):
        m = re.match(r"£([\d,]+)", v or "")
        return int(m.group(1).replace(",", "")) if m else None

    vals = sorted(v for v in (value_num(c["value"]) for g in ghx for c in g["contracts"]) if v)
    assert vals[-1] != vals[-2], "contract value tie"
    top_ref, top_val = None, None
    for g in ghx:
        for c in g["contracts"]:
            v = value_num(c["value"])
            if v == vals[-1]:
                top_ref, top_val = c["ref"], c["value"]
    tasks.append({
        "category": "superlative",
        "question": "Which contract has the highest value across all GHX records? "
                    "Give the contract reference.",
        "gold_answer": top_ref,
        "metadata": {"value": top_val},
    })
    phases = sorted({g["phase"] for g in ghx})
    for _ in range(200):
        phase = rng.choice(phases)
        recs = [g for g in ghx if g["phase"] == phase and g["planned_golive"]]
        if len(recs) < 3:
            continue
        ds = sorted(g["planned_golive"] for g in recs)
        if ds[-1] == ds[-2]:
            continue
        g = max(recs, key=lambda g: g["planned_golive"])
        tasks.append({
            "category": "superlative",
            "question": f"What is the most recent planned go-live date among {phase}-phase "
                        f"workplans?",
            "gold_answer": g["planned_golive"],
            "metadata": {"phase": phase, "workplan_ref": g["workplan_ref"]},
        })
        break
    else:
        raise AssertionError("no unique phase golive")
    mode_done = False
    for field, label in (("company", "supplier company"), ("trust_name", "NHS trust")):
        counts = Counter(g[field] for g in ghx)
        top = counts.most_common(2)
        if top[0][1] != top[1][1]:
            tasks.append({
                "category": "superlative",
                "question": f"Which {label} is associated with the greatest number of GHX "
                            f"procurement records?",
                "gold_answer": top[0][0],
                "metadata": {field: top[0][0], "records": top[0][1]},
            })
            mode_done = True
            break
    if not mode_done:
        golives = sorted(g["planned_golive"] for g in ghx if g["planned_golive"])
        assert golives[0] != golives[1], "earliest golive tie"
        g = min((x for x in ghx if x["planned_golive"]), key=lambda x: x["planned_golive"])
        tasks.append({
            "category": "superlative",
            "question": "What is the earliest planned go-live date across all GHX workplans?",
            "gold_answer": g["planned_golive"],
            "metadata": {"workplan_ref": g["workplan_ref"]},
        })
    town_diags = defaultdict(Counter)
    for r in C.records:
        if r["town"] and r["diag_label"]:
            town_diags[r["town"]][r["diag_label"]] += 1
    uniq_mode_towns = []
    for town, ctr in town_diags.items():
        top = ctr.most_common(2)
        if len(top) == 2 and top[0][1] >= 8 and top[0][1] != top[1][1]:
            uniq_mode_towns.append((town, top[0][0], top[0][1]))
    for town, diag, n in pick(rng, uniq_mode_towns, 2):
        tasks.append({
            "category": "superlative",
            "question": f"What is the most common diagnosis among patients in {town}?",
            "gold_answer": diag,
            "metadata": {"town": town, "count": n},
        })
    return tasks


def event_set(r):
    ev = {}
    if r["diag_onset"]:
        ev["the diagnosis onset"] = r["diag_onset"]
    if r["encounter_date"]:
        ev[f"the {r['encounter_type'].lower()} encounter"] = r["encounter_date"]
    for m in r["medications"]:
        if m.get("start_date"):
            ev[f"the start of {m['name']}"] = m["start_date"]
    for p in r["procedures"]:
        if p.get("date"):
            ev[f"the {p['name']}"] = p["date"]
    for lab in r["labs"]:
        if lab.get("date"):
            ev[f"the {lab['test']} lab"] = lab["date"]
    return ev


def build_temporal(C, rng):
    tasks = []
    pool = [r for r in C.records if r["format"] == "FHIR" and r["nhs_number"] and r["dob"]]
    used = set()
    answers_mix = Counter()
    made = 0
    while made < 10:
        r = rng.choice(pool)
        ev = event_set(r)
        keys = list(ev)
        if len(keys) < 2 or r["record_id"] in used:
            continue
        a, b = rng.sample(keys, 2)
        if ev[a] == ev[b]:
            continue
        truth = "true" if ev[a] < ev[b] else "false"
        if answers_mix[truth] >= 5:
            continue
        used.add(r["record_id"])
        answers_mix[truth] += 1
        made += 1
        tasks.append({
            "category": "temporal",
            "question": f"For {C.key_phrase(r, rng.choice(['nhs', 'nd']))}, was {a} "
                        f"before {b}?",
            "gold_answer": truth,
            "metadata": {"record_ids": [r["record_id"]], "dates": {a: ev[a], b: ev[b]},
                         "pair": [a, b]},
        })
    made = 0
    while made < 5:
        r = rng.choice(pool)
        ev = event_set(r)
        keys = list(ev)
        if len(keys) < 2 or r["record_id"] in used:
            continue
        a, b = rng.sample(keys, 2)
        if ev[a] == ev[b]:
            continue
        used.add(r["record_id"])
        earlier = a if ev[a] < ev[b] else b
        made += 1
        tasks.append({
            "category": "temporal",
            "question": f"For {C.key_phrase(r, rng.choice(['nhs', 'nd']))}, which came "
                        f"first: {a} or {b}?",
            "gold_answer": earlier,
            "metadata": {"record_ids": [r["record_id"]], "dates": {a: ev[a], b: ev[b]},
                         "pair": [a, b]},
        })
    return tasks


def build_cross_format(C, rng):
    tasks = []
    td_counts = Counter((r["town"], r["diag_label"]) for r in C.records if r["town"] and r["diag_label"])
    td_pool = [k for k, v in td_counts.items() if 4 <= v <= 15]
    for town, diag in pick(rng, td_pool, 4):
        recs = C.where(town=town, diag_label=diag)
        tasks.append({
            "category": "cross_format",
            "question": f"How many patients in {town} are diagnosed with {diag} across all "
                        f"message formats combined?",
            "gold_answer": str(len(recs)),
            "metadata": {"formats": sorted({r["format"] for r in recs}), "count": len(recs)},
        })
    for fmt in ("FHIR", "HL7", "SNOMED"):
        diag_pool = [r["diag_label"] for r in C.records if r["format"] == fmt and r["diag_label"]]
        diag = rng.choice(sorted(set(diag_pool)))
        n = len(C.where(format=fmt, diag_label=diag))
        tasks.append({
            "category": "cross_format",
            "question": f"How many {fmt} records have a diagnosis of {diag}?",
            "gold_answer": str(n),
            "metadata": {"format": fmt, "diag": diag, "count": n},
        })
    labels = sorted({r["diag_label"] for r in C.records if r["diag_label"]})
    for diag in rng.sample(labels, 2):
        codes = {r["diag_code"] for r in C.records if r["diag_label"] == diag and r["diag_code"]}
        assert len(codes) == 1, f"code not 1:1 for {diag}"
        tasks.append({
            "category": "cross_format",
            "question": f"Which SNOMED diagnosis code corresponds to the diagnosis "
                        f"'{diag}' used consistently across formats?",
            "gold_answer": codes.pop(),
            "metadata": {"diag": diag},
        })
    for _ in range(2000):
        town, diag = rng.choice(td_pool)
        fmts = sorted({r["format"] for r in C.where(town=town, diag_label=diag)})
        if fmts == ["FHIR", "HL7", "SNOMED"]:
            tasks.append({
                "category": "cross_format",
                "question": f"For patients in {town} diagnosed with {diag}, which message "
                            f"formats occur? Answer with a comma-separated list.",
                "gold_answer": "FHIR, HL7, SNOMED",
                "metadata": {"town": town, "diag": diag},
            })
            break
    else:
        raise AssertionError("no all-three-formats town+diag")
    assert len(tasks) == 10
    return tasks


def build_procurement(C, rng):
    tasks = []
    ghx_recs = [r for r in C.records if r["format"] == "GHX"]
    used = set()

    def fresh():
        for _ in range(500):
            r = rng.choice(ghx_recs)
            if r["record_id"] not in used and r["ghx"]["contracts"]:
                used.add(r["record_id"])
                return r
        raise AssertionError("ghx pool exhausted")

    r = fresh()
    c = r["ghx"]["contracts"][0]
    tasks.append((f"What is the value of contract {c['ref']}?", c["value"],
                  {"contract_ref": c["ref"]}))
    r = fresh()
    c = r["ghx"]["contracts"][0]
    tasks.append((f"What is the status of contract {c['ref']}?", c["status"],
                  {"contract_ref": c["ref"]}))
    r = fresh()
    c = r["ghx"]["contracts"][0]
    tasks.append((f"What is the contract duration agreed in {c['ref']}?", c["duration"],
                  {"contract_ref": c["ref"]}))
    r = fresh()
    g = r["ghx"]
    tasks.append((f"Which NHS trust issued procurement {g['procurement_ref']}?",
                  g["trust_name"], {"procurement_ref": g["procurement_ref"]}))
    r = fresh()
    g = r["ghx"]
    tasks.append((f"What is the patient cohort size for procurement {g['procurement_ref']}?",
                  str(g["cohort_size"]), {"procurement_ref": g["procurement_ref"]}))
    r = fresh()
    g = r["ghx"]
    tasks.append((f"What phase is project {g['project_ref']} in?", g["phase"],
                  {"project_ref": g["project_ref"]}))
    r = fresh()
    g = r["ghx"]
    tasks.append((f"Who is the lead clinician for project {g['project_ref']}?",
                  g["lead_clinician"], {"project_ref": g["project_ref"]}))
    r = fresh()
    g = r["ghx"]
    tasks.append((f"How many milestones are in workplan {g['workplan_ref']}?",
                  str(g["milestones"]), {"workplan_ref": g["workplan_ref"]}))
    r = fresh()
    g = r["ghx"]
    tasks.append((f"What is the planned go-live date for workplan {g['workplan_ref']}?",
                  g["planned_golive"], {"workplan_ref": g["workplan_ref"]}))
    ghx = [r["ghx"] for r in ghx_recs]
    cohorts = sorted(g["cohort_size"] or 0 for g in ghx)
    assert cohorts[-1] != cohorts[-2]
    gmax = max(ghx, key=lambda g: g["cohort_size"] or 0)
    tasks.append(("Which company is the supplier for the procurement with the largest "
                  "patient cohort size?", gmax["company"],
                  {"cohort_size": gmax["cohort_size"], "trust": gmax["trust_name"]}))
    return [{
        "category": "procurement", "question": q, "gold_answer": str(gold),
        "metadata": meta,
    } for q, gold, meta in tasks]


def build_unanswerable(C, rng):
    tasks = []
    towns = sorted({r["town"] for r in C.records if r["town"]})
    genders = sorted({r["gender"] for r in C.records if r["gender"]})
    diags = sorted({r["diag_label"] for r in C.records if r["diag_label"]})
    dtg = Counter((r["diag_label"], r["town"], r["gender"])
                  for r in C.records if r["diag_label"] and r["town"] and r["gender"])
    td = Counter((r["diag_label"], r["town"]) for r in C.records if r["diag_label"] and r["town"])
    dg = Counter((r["diag_label"], r["gender"]) for r in C.records if r["diag_label"] and r["gender"])
    empty_dtg = [(d, t, g) for d in diags for t in towns for g in genders
                 if (d, t, g) not in dtg and (d, t) in td and (d, g) in dg]
    for diag, town, gender in pick(rng, empty_dtg, 3):
        assert not C.where(diag_label=diag, town=town, gender=gender)
        tasks.append({
            "category": "unanswerable",
            "question": f"List the record IDs of {gender_phrase(gender)} in {town} "
                        f"diagnosed with {diag}.",
            "gold_answer": "no matching records",
            "metadata": {"verified_empty": [f"diag={diag}", f"town={town}", f"gender={gender}"]},
        })
    sd = Counter((r["specialty"], r["diag_label"])
                 for r in C.records if r["specialty"] and r["diag_label"])
    specialties = sorted({s for s, _ in sd})
    empty_sd = [(s, d) for s in specialties for d in diags if (s, d) not in sd]
    for spec, diag in pick(rng, empty_sd, 2):
        assert not C.where(specialty=spec, diag_label=diag)
        tasks.append({
            "category": "unanswerable",
            "question": f"Which patients were seen under the specialty {spec} with a "
                        f"diagnosis of {diag}?",
            "gold_answer": "no matching records",
            "metadata": {"verified_empty": [f"specialty={spec}", f"diag={diag}"]},
        })
    wd = Counter((r["ward"], r["diag_label"]) for r in C.records if r["ward"] and r["diag_label"])
    wards = sorted({w for w, _ in wd})
    empty_wd = [(w, d) for w in wards for d in diags if (w, d) not in wd]
    for ward, diag in pick(rng, empty_wd, 2):
        assert not C.where(ward=ward, diag_label=diag)
        tasks.append({
            "category": "unanswerable",
            "question": f"Which patients on {ward} have a diagnosis of {diag}?",
            "gold_answer": "no matching records",
            "metadata": {"verified_empty": [f"ward={ward}", f"diag={diag}"]},
        })
    ghx = [r["ghx"] for r in C.records if r["format"] == "GHX"]
    ct = Counter((g["company"], g["trust_name"]) for g in ghx)
    empty_ct = [(c, t) for c in sorted({g["company"] for g in ghx})
                for t in sorted({g["trust_name"] for g in ghx}) if (c, t) not in ct]
    company, trust = pick(rng, empty_ct, 1)[0]
    assert not [g for g in ghx if g["company"] == company and g["trust_name"] == trust]
    tasks.append({
        "category": "unanswerable",
        "question": f"List the GHX procurement records where {company} supplied "
                    f"{trust}.",
        "gold_answer": "no matching records",
        "metadata": {"verified_empty": [f"company={company}", f"trust={trust}"]},
    })
    all_nhs = {r["nhs_number"] for r in C.records if r["nhs_number"]}
    fake = 1872446700
    while str(fake) in all_nhs:
        fake += 7
    tasks.append({
        "category": "unanswerable",
        "question": f"What is the GP surgery of the patient with NHS number {fake}?",
        "gold_answer": "no matching records",
        "metadata": {"verified_empty": [f"nhs={fake} (not in corpus)"]},
    })
    all_refs = {c["ref"] for g in ghx for c in g["contracts"]}
    fake_ref = "CON-999999999"
    while fake_ref in all_refs:
        fake_ref = f"CON-{int(fake_ref.split('-')[1]) - 13}"
    tasks.append({
        "category": "unanswerable",
        "question": f"What is the value of contract {fake_ref}?",
        "gold_answer": "no matching records",
        "metadata": {"verified_empty": [f"ref={fake_ref} (not in corpus)"]},
    })
    return tasks


def build_safety(C, rng):
    tasks = []
    with_alg = [r for r in C.records if r["nhs_number"] and r["dob"] and r["allergies"]]
    with_alerts = [r for r in C.records if r["nhs_number"] and r["dob"] and r["alerts"] is not None]
    used = set()
    made = 0
    while made < 3:
        r = rng.choice(with_alg)
        sev = rng.choice(["Life-threatening", "Severe", "Moderate"])
        has = any(a["severity"] == sev and a["confirmed"] == "Yes" for a in r["allergies"])
        if r["record_id"] in used:
            continue
        used.add(r["record_id"])
        made += 1
        sev_l = "life-threatening" if sev == "Life-threatening" else sev.lower()
        tasks.append({
            "category": "safety",
            "question": f"Does {C.key_phrase(r, 'nd')} have any confirmed {sev_l} "
                        f"allergies?",
            "gold_answer": "true" if has else "false",
            "metadata": {"record_ids": [r["record_id"]],
                         "confirmed_severities": sorted(
                             a["severity"] for a in r["allergies"] if a["confirmed"] == "Yes")},
        })
    made = 0
    while made < 2:
        r = rng.choice(with_alerts)
        if r["record_id"] in used:
            continue
        used.add(r["record_id"])
        made += 1
        tasks.append({
            "category": "safety",
            "question": f"How many safety alert flags are recorded for "
                        f"{C.key_phrase(r, 'nhs')}?",
            "gold_answer": str(len(r["alerts"])),
            "metadata": {"record_ids": [r["record_id"]], "alerts": list(r["alerts"])},
        })
    alert_pool = sorted({a for r in C.records for a in r["alerts"]})
    made = 0
    want = ["true", "false"]
    rng.shuffle(want)
    while made < 2:
        r = rng.choice(with_alerts)
        alert = rng.choice(alert_pool)
        has = alert in r["alerts"]
        if r["record_id"] in used:
            continue
        if ("true" if has else "false") != want[made]:
            continue
        used.add(r["record_id"])
        made += 1
        tasks.append({
            "category": "safety",
            "question": f"Is {C.key_phrase(r, 'nd')} flagged with the alert "
                        f"'{alert}'?",
            "gold_answer": "true" if has else "false",
            "metadata": {"record_ids": [r["record_id"]], "alerts": list(r["alerts"])},
        })
    made = 0
    while made < 2:
        r = rng.choice(with_alg)
        n = sum(1 for a in r["allergies"] if a["confirmed"] == "Yes")
        if r["record_id"] in used or n == 0:
            continue
        used.add(r["record_id"])
        made += 1
        tasks.append({
            "category": "safety",
            "question": f"How many of the recorded allergies for {C.key_phrase(r, 'nhs')} "
                        f"are confirmed?",
            "gold_answer": str(n),
            "metadata": {"record_ids": [r["record_id"]], "confirmed": n,
                         "total_allergies": len(r["allergies"])},
        })
    made = 0
    while made < 1:
        cands = [r for r in with_alg
                 if sum(1 for a in r["allergies"] if a["severity"] == "Life-threatening") == 1]
        if not cands:
            raise AssertionError("no single-LT record")
        r = rng.choice(cands)
        if r["record_id"] in used:
            continue
        alg = [a["allergen"] for a in r["allergies"] if a["severity"] == "Life-threatening"][0]
        used.add(r["record_id"])
        made += 1
        tasks.append({
            "category": "safety",
            "question": f"Which allergen is recorded as life-threatening for "
                        f"{C.key_phrase(r, 'nd')}?",
            "gold_answer": alg,
            "metadata": {"record_ids": [r["record_id"]]},
        })
    return tasks


def build_all():
    ensure_data()
    records = load_corpus()
    print(f"corpus: {len(records)} records")
    C = Corpus(records)
    rng = random.Random(SEED)
    tasks = []
    for builder in (build_lookup, build_filter, build_multihop, build_aggregate,
                    build_superlative, build_temporal, build_cross_format,
                    build_procurement, build_unanswerable, build_safety):
        part = builder(C, random.Random(rng.randrange(1 << 30)))
        assert len(part) == EXPECTED[builder.__name__.replace("build_", "")], \
            f"{builder.__name__}: {len(part)} != {EXPECTED[builder.__name__.replace('build_', '')]}"
        tasks.extend(part)
        print(f"  {builder.__name__:24s} {len(part):3d}")
    questions = [t["question"] for t in tasks]
    assert len(set(questions)) == len(questions), "duplicate questions"
    for i, t in enumerate(tasks, 1):
        assert t["gold_answer"] and isinstance(t["gold_answer"], str)
        t["id"] = f"lind-{i:03d}"
    return tasks


CATEGORIES_DOC = {
    "lookup": "single attribute of a uniquely-keyed record (record ID / NHS number / name+DOB)",
    "filter": "one or two conditions; answer is a list (names alphabetical or record IDs ascending)",
    "multihop": "2-5 hops: key/filter/superlative -> record -> list selection -> attribute -> count; "
                "difficulty rises with hop count (see metadata.hops)",
    "aggregate": "counts over filtered groups, computed over the full 20k corpus",
    "superlative": "oldest/youngest/earliest/latest/highest with a machine-verified unique winner",
    "temporal": "before/after ordering and which-came-first over dated events within a record",
    "cross_format": "questions spanning FHIR R4 / HL7 v2.5 / SNOMED encodings of the same concept",
    "procurement": "GHX procurement records: contracts, projects, workplans, cohorts",
    "unanswerable": "verified-empty questions (zero matching records) — hallucination resistance",
    "safety": "allergy confirmation/severity and safety-alert flags",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="regenerate and diff against the existing tasks.json")
    args = parser.parse_args()

    tasks = build_all()
    doc = {
        "dataset": "lindisfarne",
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "source": "HuggingFace Toridion/lindisfarne-m1, full tier, chunks records_0001.jsonl + "
                  "records_0002.jsonl (20,000 synthetic NHS records; 6,798 FHIR R4, 6,720 HL7 v2.5, "
                  "5,658 SNOMED, 824 GHX)",
        "answer_format": "names 'First Last'; dates YYYY-MM-DD; numbers plain (no separators); "
                         "booleans 'true'/'false'; name lists comma-separated alphabetical; "
                         "record-ID lists comma-separated ascending (REC-XXXXXXXX); "
                         "contract values exactly as stored (e.g. '£3,074,270'); "
                         "verified-empty questions: 'no matching records'",
        "counts": {c: sum(1 for t in tasks if t["category"] == c) for c in CATEGORIES_DOC},
        "multihop_hops": "a hop = one resolution step: (1) resolve key or filter to record(s), "
                         "(2) read a list/nested field, (3) select an item inside it, "
                         "(4) read its attribute, (5) filter/count over it. metadata.hops records "
                         "the intended depth (2-5).",
        "categories": CATEGORIES_DOC,
        "tasks": [
            {"id": t["id"], "question": t["question"], "gold_answer": t["gold_answer"],
             "category": t["category"], "metadata": t["metadata"]}
            for t in tasks
        ],
    }
    if args.check:
        existing = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        if existing["tasks"] == doc["tasks"]:
            print("CHECK OK: tasks identical to benchmarks/lindisfarne/tasks.json")
        else:
            diffs = sum(1 for a, b in zip(existing["tasks"], doc["tasks"]) if a != b)
            print(f"CHECK FAILED: {diffs} task(s) differ")
            sys.exit(1)
        return
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT_PATH} ({len(tasks)} tasks)")
    by_cat = Counter(t["category"] for t in tasks)
    for cat in CATEGORIES_DOC:
        print(f"  {cat:14s} {by_cat[cat]:3d}")
    hops = Counter(t["metadata"].get("hops") for t in tasks if t["category"] == "multihop")
    print(f"  multihop hops: {dict(sorted(hops.items(), key=lambda x: (x[0] is None, x[0])))}")


if __name__ == "__main__":
    main()
