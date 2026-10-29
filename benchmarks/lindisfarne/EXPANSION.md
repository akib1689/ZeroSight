# Lindisfarne question bank — expansion guide (140 → ~800)

`tasks.json` holds the 140-question seed set, generated and verified by
`scripts/generate_lindisfarne_questions.py` (deterministic; `--check` re-derives
and diffs it against the file). Corpus scope: `data/lindisfarne/full/records_000{1,2}.jsonl`
(20,000 records; FHIR R4 6,798 · HL7 v2.5 6,720 · SNOMED 5,658 · GHX 824).

## Target distribution (800)

| Category | Seed | Target | How to scale |
|---|---|---|---|
| lookup | 20 | 150 | widen the `(field, format)` spec list; every record is a fresh key candidate |
| filter | 15 | 100 | draw more cells from the verified pools below (cell sizes are the constraint) |
| multihop | 20 | 150 | add templates per hop tier; keep the 2:3:4:5 ratio roughly 5:6:5:4 → 30:45:40:35 |
| aggregate | 15 | 100 | any filter family × count; no size cap needed |
| superlative | 15 | 75 | unique-winner assertions already gate everything |
| temporal | 15 | 75 | FHIR records all qualify; keep the 5/5 true/false balance per batch of 10 |
| cross_format | 10 | 50 | (town, diag) × 54 labels; per-format counts; code↔label questions |
| procurement | 10 | 50 | 824 GHX records × ~10 fields each; refs are unique keys |
| unanswerable | 10 | 50 | verified-empty pools: 476 (diag,town,gender), 38 (specialty,diag), 71 (ward,diag), 10 (company,trust) + fabricated keys |
| safety | 10 | 50 | 10,130 confirmed-allergy records; 15 alert types |

## Rules that keep quality up (enforced by the generator)

1. **Golds are computed, never typed.** Every answer is derived by running the
   equivalent filter over the full 20k corpus at generation time. Keep it that way.
2. **Uniqueness assertions.** Point lookups must resolve to exactly one record
   (NHS numbers, name+DOB, and record IDs are all corpus-unique — verified).
   Superlatives must have a unique winner; ties raise and force a respec.
3. **Unanswerables must be verified empty.** Prefer combos where both pairwise
   filters are non-empty but the triple is empty (non-trivially absent), or
   fabricated-but-valid keys (NHS number, CON- ref) asserted absent.
4. **No dosage/route/clinical-plausibility questions.** The synthetic data is not
   clinically validated (e.g. "Rivaroxaban 25mcg inhaled" exists).
5. **Never translate medication codes ↔ names.** The generator maps them
   inconsistently (`372687004` → Ibuprofen *and* Amoxicillin). Diagnosis
   code ↔ label IS 1:1 across all formats (verified) and is safe to use.
6. **Mind corpus uniformity when picking filter cells.** Because generation is
   uniform, mid-size windows are often empty: (diag,gender) cells ≈ 60–90,
   (allergen,town) ≈ 44, (etype,year) ≈ 195. Verified useful windows:
   (town,diag) 3–8 · (town,diag,gender) 2–4 · (allergen,town,severity) 4–12 ·
   (town,etype,year) 2–8 · (flag,test) 5–15 · (ward,etype) 5–15.
7. **SNOMED records are thin** — no allergies/procedures/med lists, no gender.
   Route allergy/procedure questions to FHIR/HL7; use SNOMED for codes, alerts,
   GP surgeries, labs-by-name.

## Hop accounting (multihop)

A hop = one resolution step: (1) resolve key or filter to record(s),
(2) read a nested/list field, (3) select an item inside it, (4) read that item's
attribute, (5) filter/count over a list. `metadata.hops` records the intended
depth. When adding templates, place each at its natural tier and keep the
gradient — the point is measurable difficulty growth for ReAct step counts.

## Answer conventions (also in tasks.json `answer_format`)

- names `First Last` · dates `YYYY-MM-DD` · numbers plain · booleans `true`/`false`
- name lists: comma-separated, alphabetical · record-ID lists: ascending
- contract values exactly as stored (`£3,074,270`)
- verified-empty: `no matching records`

## Workflow for each expansion batch

1. Add spec entries / widen counts in the generator (never hand-edit tasks.json).
2. `uv run python scripts/generate_lindisfarne_questions.py` — fix any assertion.
3. Skim the printed questions; reword templates that read awkwardly.
4. `--check` to confirm determinism, commit script + tasks.json together.
