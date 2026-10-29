"""Generate the item-level McNemar significance table for the paper.

Reads results/paper-experiments/MANIFEST.json (single source of truth,
written by scripts/freeze_results.py), loads every frozen run's per-item
``correct`` flags, pairs baseline run i with EAG run i, and emits:

    paper/tables/results-mcnemar.tex  giant table with a nested 3x3 mini
                                      table (2x2 paired outcomes + margins,
                                      ruled) per dataset x model cell, plus
                                      the exact-binomial McNemar p value

Display scale: per-run means over the run pairs, so every number sits on
the familiar item-count scale (169/201/200/140) and the mini-table margins
reproduce the accuracy means of results-main.tex exactly (verified at
render time; hard-fails on divergence). The test itself runs on the pooled
integer counts (n_pairs x samples paired outcomes); pooling conventions
other than run-i<->run-i are NOT supported because they are anticonservative
under correlated run pairs.

All 13 frozen cells are rendered: the 12 main cells plus the hitab
glm-5.3-flash-toolset-v1 ablation (asterisk-marked, quantified in the
appendix). Output is deterministic.

Usage::

    uv run python scripts/mcnemar_tables.py             # regenerate table
    uv run python scripts/mcnemar_tables.py --check     # verify emitted file
    uv run python scripts/mcnemar_tables.py --report    # + markdown summary
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PAPER_EXPERIMENTS = REPO / "results" / "paper-experiments"
MANIFEST_PATH = PAPER_EXPERIMENTS / "MANIFEST.json"
TABLES_DIR = REPO / "results" / "tables"
MCNEMAR_TABLE_PATH = TABLES_DIR / "results-mcnemar.tex"

DATASET_ORDER = ["bird", "spider", "hitab", "lindisfarne"]
DATASET_NAMES = {
    "bird": "BIRD",
    "spider": "Spider",
    "hitab": "HiTab",
    "lindisfarne": "Lindisfarne",
}
MODEL_NAMES = {
    "glm-5.3-flash": "GLM-5.3-Flash",
    "glm-5.3-flash-toolset-v1": "GLM-5.3-Flash (v1)",
    "azure-gpt-5.6-sol": "GPT-5.6-Sol",
    "azure-deepseek-v4-flash": "DeepSeek-V4-Flash",
    "gemma-4-31b-qat-w4a16": "Gemma-4-31B-QAT",
}
ALPHA = 0.05

@dataclass
class McNemarCell:
    dataset: str
    model: str
    samples: int
    pairs: int
    partial: bool
    toolset_v1: bool
    n00: int
    n01: int
    n10: int
    n11: int
    p_value: float


def _toolset_v1(dataset: str, metrics: dict) -> bool:
    tools = metrics.get("exposure_by_tool") or {}
    return dataset == "hitab" and "compare_cells" not in tools


def _load_correct(paper_experiments: Path, rel: str) -> dict[str, bool]:
    data = json.loads((paper_experiments / rel).read_text())
    out: dict[str, bool] = {}
    for item in data["results"]:
        cid = item["id"]
        if "correct" not in item:
            raise SystemExit(f"FATAL: {rel} item {cid} has no 'correct' field")
        if cid in out:
            raise SystemExit(f"FATAL: {rel} has duplicate item id {cid}")
        out[cid] = bool(item["correct"])
    return out


def _exact_mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact binomial McNemar p on the discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return min(1.0, 2 * tail / 2**n)


def collect_cells(manifest: dict) -> list[McNemarCell]:
    cells: list[McNemarCell] = []
    for key in sorted(manifest["cells"]):
        section = manifest["cells"][key]
        dataset, model = section["dataset"], section["model"]
        baseline: dict[int, str] = {}
        eag: dict[int, str] = {}
        for entry in section["runs"]:
            target = baseline if entry["mode"] == "baseline" else eag
            target[entry["run"]] = entry["frozen_relative_path"]
        if sorted(baseline) != sorted(eag):
            raise SystemExit(
                f"FATAL: {key} baseline/EAG run numbers differ: "
                f"{sorted(baseline)} vs {sorted(eag)}"
            )
        run_paths = sorted(set(baseline.values()) | set(eag.values()))
        correctness = {p: _load_correct(PAPER_EXPERIMENTS, p) for p in run_paths}
        ids = {frozenset(c) for c in correctness.values()}
        if len(ids) != 1:
            raise SystemExit(f"FATAL: {key} item ids differ across frozen runs")
        shared = next(iter(ids))
        samples = len(shared)
        n00 = n01 = n10 = n11 = 0
        for run in sorted(baseline):
            b = correctness[baseline[run]]
            e = correctness[eag[run]]
            for cid in shared:
                bc, ec = b[cid], e[cid]
                if bc and ec:
                    n00 += 1
                elif bc:
                    n01 += 1
                elif ec:
                    n10 += 1
                else:
                    n11 += 1
        pairs = len(baseline)
        cells.append(
            McNemarCell(
                dataset=dataset,
                model=model,
                samples=samples,
                pairs=pairs,
                partial=section.get("status") == "partial",
                toolset_v1=_toolset_v1(dataset, section["runs"][0]["metrics"]),
                n00=n00,
                n01=n01,
                n10=n10,
                n11=n11,
                p_value=_exact_mcnemar_p(n01, n10),
            )
        )
    order = {name: i for i, name in enumerate(DATASET_ORDER)}
    cells.sort(key=lambda c: (order.get(c.dataset, len(order)), c.model))
    return cells


def _verify_margins(cells: list[McNemarCell], manifest: dict) -> None:
    """Mini-table margins must reproduce results-main accuracy means."""
    for cell in cells:
        key = f"{cell.dataset}/{cell.model}"
        runs = manifest["cells"][key]["runs"]
        base_margin = 100 * (cell.n00 + cell.n01) / cell.pairs / cell.samples
        eag_margin = 100 * (cell.n00 + cell.n10) / cell.pairs / cell.samples
        for mode, margin in (("baseline", base_margin), ("eag", eag_margin)):
            mode_accs = [
                entry["metrics"]["accuracy"]
                for entry in runs
                if entry["mode"] == mode
            ]
            mode_mean = 100 * sum(mode_accs) / len(mode_accs)
            if abs(margin - mode_mean) > 0.05:
                raise SystemExit(
                    f"FATAL: {key} {mode} margin {margin:.2f} diverges from "
                    "results-main accuracy; frozen runs changed?"
                )


def _model_cell(cell: McNemarCell) -> str:
    name = MODEL_NAMES.get(cell.model, cell.model)
    if cell.partial:
        name += r"$^\dagger$"
    if cell.toolset_v1:
        name += r"$^{\ast}$"
    return name


def _format_p(p: float) -> str:
    if p >= ALPHA:
        return f"{p:.3f}"
    exp = math.floor(math.log10(p))
    mant = p / 10**exp
    if round(mant, 1) >= 10.0:
        mant /= 10
        exp += 1
    return rf"$\mathbf{{{mant:.1f}{{\times}}10^{{{exp}}}}}$"


def _mini_table(cell: McNemarCell) -> str:
    s = cell.samples
    m00, m01, m10, m11 = (v / cell.pairs for v in (cell.n00, cell.n01, cell.n10, cell.n11))
    base_ok, base_err = m00 + m01, m10 + m11
    eag_ok, eag_err = m00 + m10, m01 + m11

    def pct(x: float) -> str:
        return f"{100 * x / s:.1f}\\%"

    def cnt(x: float) -> str:
        return f"{x:.1f}"

    return "\n".join(
        [
            r"\begin{tabular}[t]{@{}lrrr@{}}",
            r" & eag ok & eag err & $\Sigma$ \\",
            r"\hline",
            rf"base ok & {cnt(m00)} & {cnt(m01)} & {cnt(base_ok)} ({pct(base_ok)}) \\",
            rf"base err & {cnt(m10)} & {cnt(m11)} & {cnt(base_err)} ({pct(base_err)}) \\",
            r"\hline",
            rf"$\Sigma$ & {cnt(eag_ok)} ({pct(eag_ok)}) & {cnt(eag_err)} ({pct(eag_err)}) "
            rf"& {s} \\",
            r"\end{tabular}",
        ]
    )


def render_mcnemar_table(cells: list[McNemarCell]) -> str:
    pair_lo = min(c.pairs for c in cells)
    pair_hi = max(c.pairs for c in cells)
    n_pairs_txt = str(pair_lo) if pair_lo == pair_hi else f"{pair_lo}--{pair_hi}"
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Item-level McNemar analysis, baseline vs.\ EAG "
        rf"({len(cells)} cells; $\dagger$ marks a partial cell; $\ast$ marks "
        r"HiTab cells run on the pre-\emph{compare\_cells} toolset, "
        r"Appendix~\ref{app:toolset}). Each nested table pairs per-item "
        r"correctness across the same tasks in baseline (rows) and EAG "
        r"(columns), averaged over the frozen run pairs (run $i$ with run "
        r"$i$); off-diagonal cells are the per-run mean items only baseline "
        r"solves (upper right) and only EAG solves (lower left). Margins "
        r"reproduce the accuracy means of Table~\ref{tab:results-main} "
        r"(percent in parentheses). $p$: exact two-sided McNemar test "
        r"(binomial on the discordant pairs) on the pooled integer counts "
        rf"over all {n_pairs_txt} run pairs per cell; significant values "
        rf"($p<{ALPHA}$) bolded.}}",
        r"\label{tab:results-mcnemar}",
        r"\begin{tabular}{@{}llcr@{}}",
        r"\toprule",
        r"Dataset & Model & Paired outcomes (per run) & $p$ \\",
        r"\midrule",
    ]
    last_dataset = None
    for cell in cells:
        dataset_cell = DATASET_NAMES.get(cell.dataset, cell.dataset)
        if last_dataset is None:
            pass
        elif cell.dataset == last_dataset:
            dataset_cell = ""
            lines.append(r"\cmidrule(lr){2-4}")
        else:
            lines.append(r"\midrule")
        last_dataset = cell.dataset
        mini = _mini_table(cell)
        lines.append(
            f"{dataset_cell} & {_model_cell(cell)} & {mini} & "
            f"{_format_p(cell.p_value)} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ]
    return "\n".join(lines)


def render_report(cells: list[McNemarCell]) -> str:
    header = (
        "| dataset | model | pairs | n00 | n01 (b) | n10 (c) | n11 | pooled p |"
    )
    sep = "|---|---|---|---|---|---|---|---|"
    rows = []
    for cell in cells:
        rows.append(
            f"| {cell.dataset} | {cell.model} | {cell.pairs} "
            f"| {cell.n00} | {cell.n01} | {cell.n10} | {cell.n11} "
            f"| {cell.p_value:.2e} |"
        )
    return "\n".join([header, sep, *rows])


def write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content)
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--report", action="store_true", help="also print markdown summary"
    )
    parser.add_argument(
        "--check", action="store_true", help="verify emitted table matches generation"
    )
    args = parser.parse_args()

    manifest = json.loads(MANIFEST_PATH.read_text())
    cells = collect_cells(manifest)
    if not cells:
        print("No frozen cells found in", MANIFEST_PATH)
        return 1
    _verify_margins(cells, manifest)
    table = render_mcnemar_table(cells)

    if args.check:
        if MCNEMAR_TABLE_PATH.read_text() != table:
            print(f"CHECK FAILED: {MCNEMAR_TABLE_PATH} out of date")
            return 1
        print(f"OK: {len(cells)} cells rendered; table up to date.")
        return 0

    write_atomic(MCNEMAR_TABLE_PATH, table)
    print(f"Wrote {MCNEMAR_TABLE_PATH.relative_to(REPO)}")
    print(
        f"Cells rendered: {len(cells)} "
        f"({sum(1 for c in cells if c.p_value < ALPHA)} significant at p<{ALPHA})"
    )
    if args.report:
        print()
        print(render_report(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
