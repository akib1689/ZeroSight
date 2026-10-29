"""Content-class disclosure metrics over the frozen paper-experiment cells.

Package layout (split out of the former monolithic
``scripts/derive_disclosure_metrics.py``):

  common.py     normalization, token counting, observation-text patterns,
                inventory-matching helpers
  contexts.py   per-dataset value inventories (lindisfarne, hitab, bird/spider)
                and observation classification
  analyze.py    single-pass trajectory analysis: every metric derives from one
                walk over each run's steps
  report.py     split-output writer: one reviewable JSON file per headline
                metric under ``results/paper-experiments/derived/``
  validate.py   known-example spot checks recorded into the artifact
  cli.py        argument parsing and orchestration

Content classes (what ``data_redacted_tokens`` used to lump together):

  values       record-store literals absent from the task question -- DISCLOSURE
  handles      record identifiers (REC-IDs, cell addresses) -- deliberate
               disclosure policy: the minimal least-sensitive handle set that
               keeps the referential chain functional
  cardinality  match/row/cell counts -- functional feedback, deliberate policy
  shape        structure scaffolding (types, keys, framing) -- cost, never
               disclosure

Provider-visibility model (agent.py stores model-emitted args, pre-
resolution; tools/eag_middleware.py resolves placeholders locally; artifacts
are never rendered):
  provider view = question (user-intended, out of scope) + model emissions
  (thoughts, emitted args incl. inert placeholders) + rendered observations
  (raw data in baseline; shape-only in EAG). Resolved values live only in
  local artifacts.

Headline metrics (one file each under derived/):
  dr.json        Disclosure Rate: share of data observations with >=1 value
  vvf.json       Verified Value-Freeness: EAG data observations containing
                 >=1 store value (expect 0), plus the model-emitted
                 arg-collision channel
  rd.json        Records Disclosed: distinct records with >=1 disclosed
                 identifying value (lindisfarne only -- the PHI story)
  maxleak.json   worst-case per-task value tokens
  common.json    definitions, run inventory, content-class ledger
                 (handles / cardinality / shape), validation

Verification coverage: lindisfarne = full corpus scan (values of every record
touched by a run); hitab = full per-task table cells, parsed structurally;
bird/spider = full per-database distinct-value inventory + question
exclusion. Short numeric literals (<3 chars) are skipped in generic matching;
hitab's structured parser captures them exactly.

Derived values (sums, averages produced by evaluator tools) are out of scope:
we measure disclosure of *store literals*, not computation products. The task
question and the final answer are inherent to the task in both modes.

Read-only over the frozen cells; writes the derived/ split next to them.
"""
