"""SQL tools for text-to-SQL benchmarks (BIRD, Spider).

Each tool implements the 3-function EAG interface:

  - ``execute(**params) → ToolResult``     — actual tool logic (mandatory)
  - ``validate(**params) → (bool, str)``   — safety/validation gate (optional)
  - ``generate_metadata(result) → str``    — privacy transform (optional)

In **baseline** mode, ``_run`` returns ``(execute().output, result)`` (raw result).
In **EAG** mode, ``_run`` returns ``(generate_metadata(execute()), result)``
(metadata-only content, full result as artifact). The artifact rides on the
``ToolMessage`` without ever reaching the model, enabling deferred
placeholder substitution in the final answer.

Metadata tools use identity transforms (already safe). Data tools override
``generate_metadata`` to strip raw values, returning only column names + row count.

Eight tools:
  Metadata: get_schema, get_column_meaning, get_all_column_meanings,
            get_external_knowledge_names, get_knowledge_definition
  Data:     get_sample_rows, execute_sql
  Eval:     submit_sql
"""

import csv
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

_MAX_ROWS = 500


# ── ToolResult ───────────────────────────────────────────────────────────────


@dataclass
class ToolResult:
    """Structured tool result — carries formatted output + extractable metadata.

    ``execute()`` returns this. Baseline mode shows ``output``.
    EAG mode runs ``generate_metadata(result)`` instead, but keeps the result
    around as a langchain ``artifact`` so the harness can locally resolve
    placeholders the model embeds in tool arguments or its final answer
    (deferred substitution). The raw data never enters the model's context.

    ``data`` is the generic parsed artifact (any JSON-serializable value):
    tabular tools store ``{"columns": [...], "rows": [...]}``, JSON-returning
    tools (e.g. τ-bench) store the parsed dict/list, scalar tools the value.
    The placeholder resolver navigates it with a unified path grammar.
    """

    output: str
    columns: list[str] = field(default_factory=list)
    row_count: int = 0
    error: str | None = None
    rows: list[tuple] = field(default_factory=list)
    data: Any = None


# ── EAGBaseTool ──────────────────────────────────────────────────────────────


class EAGBaseTool(BaseTool):
    """BaseTool with 3-function EAG interface.

    Subclasses implement ``execute()`` (mandatory) and optionally
    ``validate()`` and ``generate_metadata()``.

    The ``eag_mode`` flag toggles between baseline (raw) and EAG (metadata)
    behavior. When False, ``_run`` returns ``result.output`` (identity).
    When True, ``_run`` returns ``generate_metadata(result)``.

    In both modes ``_run`` returns a ``(content, artifact)`` tuple
    (``response_format = "content_and_artifact"``). The ``content`` is what the
    model sees — raw output in baseline, redacted metadata in EAG. The
    ``artifact`` is the full ``ToolResult`` (incl. raw ``rows``); it rides
    along on the ``ToolMessage`` but is never shown to the model, so the
    harness can resolve ``{{ step[N].col.<name>[R] }}`` placeholders locally.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    eag_mode: bool = False
    response_format: str = "content_and_artifact"

    def execute(self, **kwargs) -> ToolResult:
        """Actual tool logic. Must be implemented by every subclass."""
        raise NotImplementedError

    def validate(self, **kwargs) -> tuple[bool, str]:
        """Validation gate. Returns ``(ok, error_message)``.

        Default: pass everything. Override to add safety checks.
        """
        return True, ""

    def generate_metadata(self, result: ToolResult) -> str:
        """Transform result to metadata-only representation.

        Default: identity (return ``result.output``).
        Override for data tools to strip raw values.
        """
        return result.output

    def effective_exposure_type(self) -> str:
        """Exposure type after applying the EAG privacy transform.

        Data tools in EAG mode return schema-only observations (column
        names + row counts via ``generate_metadata``), so their effective
        exposure for accounting purposes is ``"data_redacted"`` rather
        than ``"data"``. The class-level ``exposure_type`` is preserved
        for would-have-leaked analysis.

        Metadata and evaluation tools are unaffected; they return their
        original ``exposure_type`` regardless of ``eag_mode``.
        """
        if self.eag_mode and self.exposure_type == "data":
            return "data_redacted"
        return self.exposure_type

    def _run(self, **kwargs) -> tuple[str, ToolResult | None]:
        """Return ``(content, artifact)``.

        ``content`` is model-visible (raw output in baseline, redacted metadata
        in EAG). ``artifact`` is the full ``ToolResult`` (local-only, used for
        deferred placeholder substitution) — ``None`` when the run errored at
        the validation gate before producing a result.
        """
        ok, msg = self.validate(**kwargs)
        if not ok:
            return f"Validation failed: {msg}", None
        result = self.execute(**kwargs)
        # Generic artifact: tabular results are auto-wrapped so every tool's
        # result is navigable by the unified placeholder path grammar.
        if result.data is None and result.columns:
            result.data = {
                "columns": list(result.columns),
                "rows": [list(r) for r in result.rows],
            }
        if self.eag_mode:
            return self.generate_metadata(result), result
        return result.output, result


# ── Helpers ──────────────────────────────────────────────────────────────────


def _connect_readonly(db_path: str) -> sqlite3.Connection:
    """Open a read-only SQLite connection."""
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.execute("PRAGMA query_only = 1")
    return conn


def _strip_sql_comments(sql: str) -> str:
    """Remove ``--`` line comments and ``/* */`` block comments."""
    sql = re.sub(r"--[^\n]*", "", sql)
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    return sql.strip()


def _validate_select(sql: str) -> str | None:
    """Return error message if SQL is not a SELECT/WITH query, else None."""
    cleaned = _strip_sql_comments(sql)
    if not cleaned:
        return "Empty SQL query."
    first_word = cleaned.split()[0].upper()
    if first_word not in ("SELECT", "WITH", "EXPLAIN"):
        return f"Only SELECT/WITH queries are allowed (got '{first_word}')."
    return None


def _validate_identifier(name: str) -> bool:
    """Check if a string is a safe SQL identifier."""
    if not name or len(name) > 128:
        return False
    return bool(re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name))


def _format_results(columns: list[str], rows: list[tuple]) -> str:
    """Format query results as a readable table."""
    if not columns:
        return "(Query returned no column headers)"
    header = " | ".join(columns)
    if not rows:
        return f"{header}\n\n(0 rows returned)"
    lines = [header]
    for row in rows:
        lines.append(" | ".join(str(v) for v in row))
    suffix = f"\n\n({len(rows)} rows returned)"
    if len(rows) >= _MAX_ROWS:
        suffix += " — results may be truncated"
    return "\n".join(lines) + suffix


def _format_metadata(result: ToolResult) -> str:
    """Level B metadata: column names + row count, no raw values."""
    if result.error:
        return f"[EAG] Query failed: {result.error}"
    cols = ", ".join(result.columns) if result.columns else "N/A"
    return (
        f"[EAG] Query executed successfully.\n"
        f"Columns: {cols}\n"
        f"Rows returned: {result.row_count}\n"
        f"(Raw data withheld — schema-only view)"
    )


def _load_column_descriptions(
    descriptions_dir: Path,
) -> dict[str, dict[str, dict]]:
    """Load BIRD's ``database_description/*.csv`` into nested dict."""
    result: dict[str, dict[str, dict]] = {}
    desc_path = Path(descriptions_dir)
    if not desc_path.is_dir():
        return result
    for csv_file in sorted(desc_path.glob("*.csv")):
        table_name = csv_file.stem
        result[table_name] = {}
        try:
            with open(csv_file, newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    col_name = (row.get("original_column_name") or "").strip()
                    if not col_name:
                        continue
                    result[table_name][col_name] = {
                        "column_name": (row.get("column_name") or "").strip(),
                        "description": (row.get("column_description") or "").strip(),
                        "data_format": (row.get("data_format") or "").strip(),
                        "value_description": (
                            row.get("value_description") or ""
                        ).strip(),
                    }
        except Exception:
            continue
    return result


# ── Input schemas ────────────────────────────────────────────────────────────


class _TableColumnInput(BaseModel):
    table_name: str = Field(..., description="Name of the table")
    column_name: str = Field(..., description="Name of the column")


class _KnowledgeInput(BaseModel):
    knowledge_name: str = Field(
        ...,
        description="Name of the knowledge topic (from get_external_knowledge_names)",
    )


class _SampleRowsInput(BaseModel):
    table_name: str = Field(..., description="Name of the table to sample")
    n_rows: int = Field(default=3, description="Number of sample rows (default: 3)")


class _SQLInput(BaseModel):
    sql_query: str = Field(
        ..., description="SQL SELECT query to execute against the database"
    )


# ── Metadata tools ───────────────────────────────────────────────────────────
# All inherit identity generate_metadata (already safe).


class SchemaTool(EAGBaseTool):
    """Get the complete database schema (DDL)."""

    name: str = "get_schema"
    description: str = (
        "Get the complete database schema: all CREATE TABLE statements with "
        "column names, data types, primary keys, and foreign keys."
    )
    exposure_type: str = "metadata"
    db_path: str = ""

    def execute(self, **kwargs) -> ToolResult:
        conn = _connect_readonly(self.db_path)
        try:
            cursor = conn.execute(
                "SELECT name, sql FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            )
            tables = cursor.fetchall()
            if not tables:
                return ToolResult(output="No tables found in the database.")
            parts = []
            for _name, ddl in tables:
                if ddl:
                    parts.append(ddl.strip() + ";")
                else:
                    parts.append(f"-- {_name} (DDL not available)")
            return ToolResult(output="\n\n".join(parts))
        finally:
            conn.close()


class ColumnMeaningTool(EAGBaseTool):
    """Get the description of a specific column."""

    name: str = "get_column_meaning"
    description: str = (
        "Get the human-written description, data format, and value description "
        "for a specific column in a table."
    )
    args_schema: type[BaseModel] = _TableColumnInput
    exposure_type: str = "metadata"
    descriptions: dict = Field(default_factory=dict)

    def execute(self, **kwargs) -> ToolResult:
        table_name = kwargs.get("table_name", "")
        column_name = kwargs.get("column_name", "")
        table_desc = self.descriptions.get(table_name, {})
        col_info = table_desc.get(column_name)
        if col_info is None:
            for k, v in table_desc.items():
                if k.lower() == column_name.lower():
                    col_info = v
                    break
        if col_info is None:
            return ToolResult(
                output=(
                    f"No description available for column '{column_name}' "
                    f"in table '{table_name}'."
                )
            )
        parts = [f"Column: {column_name} (table: {table_name})"]
        if col_info.get("column_name"):
            parts.append(f"  Friendly name: {col_info['column_name']}")
        if col_info.get("description"):
            parts.append(f"  Description: {col_info['description']}")
        if col_info.get("data_format"):
            parts.append(f"  Data format: {col_info['data_format']}")
        if col_info.get("value_description"):
            parts.append(f"  Value description: {col_info['value_description']}")
        return ToolResult(output="\n".join(parts))


class AllColumnMeaningsTool(EAGBaseTool):
    """Get descriptions for all columns in all tables."""

    name: str = "get_all_column_meanings"
    description: str = (
        "Get column descriptions for ALL tables and columns in the database. "
        "Use this to understand the semantics of every column."
    )
    exposure_type: str = "metadata"
    descriptions: dict = Field(default_factory=dict)

    def execute(self, **kwargs) -> ToolResult:
        if not self.descriptions:
            return ToolResult(
                output="No column descriptions available for this database."
            )
        parts = []
        for table_name, cols in sorted(self.descriptions.items()):
            table_lines = [f"\n[{table_name}]"]
            for col_name, info in cols.items():
                desc = info.get("description", "")
                fmt = info.get("data_format", "")
                vd = info.get("value_description", "")
                line = f"  {col_name}"
                if fmt:
                    line += f" ({fmt})"
                if desc:
                    line += f": {desc}"
                if vd:
                    line += f" [values: {vd}]"
                table_lines.append(line)
            if len(table_lines) > 1:
                parts.append("\n".join(table_lines))
        return ToolResult(output="\n".join(parts))


class KnowledgeNamesTool(EAGBaseTool):
    """List available external knowledge topics."""

    name: str = "get_external_knowledge_names"
    description: str = (
        "List all available external knowledge topics that provide domain "
        "hints and reasoning guidance for answering this question."
    )
    exposure_type: str = "metadata"
    knowledge_names: list = Field(default_factory=list)

    def execute(self, **kwargs) -> ToolResult:
        if not self.knowledge_names:
            return ToolResult(
                output="No external knowledge available for this question."
            )
        output = "Available knowledge topics:\n" + "\n".join(
            f"  - {name}" for name in self.knowledge_names
        )
        return ToolResult(output=output)


class KnowledgeDefinitionTool(EAGBaseTool):
    """Get the definition of a specific knowledge topic."""

    name: str = "get_knowledge_definition"
    description: str = (
        "Get the content of a specific external knowledge topic. "
        "Call get_external_knowledge_names first to see available topics, "
        "then retrieve the ones relevant to the question."
    )
    args_schema: type[BaseModel] = _KnowledgeInput
    exposure_type: str = "metadata"
    knowledge_definitions: dict = Field(default_factory=dict)

    def execute(self, **kwargs) -> ToolResult:
        knowledge_name = kwargs.get("knowledge_name", "")
        definition = self.knowledge_definitions.get(knowledge_name)
        if definition is None:
            for k, v in self.knowledge_definitions.items():
                if k.lower() == knowledge_name.lower():
                    definition = v
                    break
        if definition is None:
            available = (
                ", ".join(self.knowledge_definitions.keys())
                if self.knowledge_definitions
                else "none"
            )
            return ToolResult(
                output=(
                    f"Knowledge topic '{knowledge_name}' not found. "
                    f"Available: {available}"
                )
            )
        return ToolResult(output=f"[{knowledge_name}]\n{definition}")


# ── Data tools ───────────────────────────────────────────────────────────────
# Override validate() and generate_metadata() for privacy + safety.


class SampleRowsTool(EAGBaseTool):
    """Get sample rows from a table."""

    name: str = "get_sample_rows"
    description: str = (
        "Get sample rows from a table to understand actual data values and "
        "patterns. Returns the first N rows (default 3)."
    )
    args_schema: type[BaseModel] = _SampleRowsInput
    exposure_type: str = "data"
    db_path: str = ""

    def validate(self, **kwargs) -> tuple[bool, str]:
        table_name = kwargs.get("table_name", "")
        if not _validate_identifier(table_name):
            return False, f"Invalid table name: '{table_name}'"
        return True, ""

    def execute(self, **kwargs) -> ToolResult:
        table_name = kwargs.get("table_name", "")
        n_rows = max(1, min(kwargs.get("n_rows", 3), 50))
        conn = _connect_readonly(self.db_path)
        try:
            cursor = conn.execute(
                f'SELECT * FROM "{table_name}" LIMIT {int(n_rows)}'
            )
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            return ToolResult(
                output=_format_results(columns, rows),
                columns=columns,
                row_count=len(rows),
                rows=rows,
            )
        except sqlite3.Error as e:
            return ToolResult(
                output=f"Error sampling rows from '{table_name}': {e}",
                error=str(e),
            )
        finally:
            conn.close()

    def generate_metadata(self, result: ToolResult) -> str:
        return _format_metadata(result)


class ExecuteSQLTool(EAGBaseTool):
    """Execute a read-only SQL SELECT query."""

    name: str = "execute_sql"
    description: str = (
        "Execute a SQL SELECT query against the database and return the "
        "results. Use this to test and verify your queries before submitting."
    )
    args_schema: type[BaseModel] = _SQLInput
    exposure_type: str = "data"
    db_path: str = ""

    def validate(self, **kwargs) -> tuple[bool, str]:
        sql = kwargs.get("sql_query", "")
        err = _validate_select(sql)
        if err:
            return False, err
        return True, ""

    def execute(self, **kwargs) -> ToolResult:
        sql_query = kwargs.get("sql_query", "")
        conn = _connect_readonly(self.db_path)
        try:
            cursor = conn.execute(sql_query)
            columns = (
                [desc[0] for desc in cursor.description]
                if cursor.description
                else []
            )
            rows = cursor.fetchmany(_MAX_ROWS)
            return ToolResult(
                output=_format_results(columns, rows),
                columns=columns,
                row_count=len(rows),
                rows=rows,
            )
        except sqlite3.Error as e:
            return ToolResult(output=f"SQL Error: {e}", error=str(e))
        except Exception as e:
            return ToolResult(
                output=f"Error executing query: {e}", error=str(e)
            )
        finally:
            conn.close()

    def generate_metadata(self, result: ToolResult) -> str:
        return _format_metadata(result)


# ── Evaluation tool ──────────────────────────────────────────────────────────


class SubmitSQLTool(EAGBaseTool):
    """Submit the final SQL query for evaluation."""

    name: str = "submit_sql"
    description: str = (
        "Submit your final SQL query for evaluation. Call this when you are "
        "confident your SQL correctly answers the question. After calling "
        "submit_sql, respond with a brief text answer — do not call any more "
        "tools."
    )
    args_schema: type[BaseModel] = _SQLInput
    exposure_type: str = "evaluation"
    db_path: str = ""

    def validate(self, **kwargs) -> tuple[bool, str]:
        sql = kwargs.get("sql_query", "")
        err = _validate_select(sql)
        if err:
            return False, err
        return True, ""

    def execute(self, **kwargs) -> ToolResult:
        return ToolResult(
            output=(
                "SQL submitted successfully. The query will be evaluated "
                "against the gold answer."
            )
        )


# ── Factory functions ────────────────────────────────────────────────────────


def make_bird_tools(
    db_path: str,
    descriptions_dir: str | Path | None = None,
    evidence: str | None = None,
    eag_mode: bool = False,
) -> list[BaseTool]:
    """Create all 8 SQL tools for a BIRD task.

    Args:
        db_path: Path to the task's SQLite database.
        descriptions_dir: Path to ``database_description/`` directory.
        evidence: Per-question domain hint (BIRD ``evidence`` field).
        eag_mode: If True, data tools return metadata-only observations.
    """
    descriptions: dict[str, dict[str, dict]] = {}
    if descriptions_dir:
        descriptions = _load_column_descriptions(Path(descriptions_dir))

    knowledge_names: list[str] = []
    knowledge_defs: dict[str, str] = {}
    if evidence:
        knowledge_names = ["domain_hint"]
        knowledge_defs = {"domain_hint": evidence}

    return [
        SchemaTool(db_path=db_path, eag_mode=eag_mode),
        ColumnMeaningTool(descriptions=descriptions, eag_mode=eag_mode),
        AllColumnMeaningsTool(descriptions=descriptions, eag_mode=eag_mode),
        KnowledgeNamesTool(knowledge_names=knowledge_names, eag_mode=eag_mode),
        KnowledgeDefinitionTool(
            knowledge_definitions=knowledge_defs, eag_mode=eag_mode
        ),
        SampleRowsTool(db_path=db_path, eag_mode=eag_mode),
        ExecuteSQLTool(db_path=db_path, eag_mode=eag_mode),
        SubmitSQLTool(db_path=db_path, eag_mode=eag_mode),
    ]


def make_spider_tools(db_path: str, eag_mode: bool = False) -> list[BaseTool]:
    """Create 4 SQL tools for a Spider task (no descriptions or knowledge).

    Args:
        db_path: Path to the task's SQLite database.
        eag_mode: If True, data tools return metadata-only observations.
    """
    return [
        SchemaTool(db_path=db_path, eag_mode=eag_mode),
        SampleRowsTool(db_path=db_path, eag_mode=eag_mode),
        ExecuteSQLTool(db_path=db_path, eag_mode=eag_mode),
        SubmitSQLTool(db_path=db_path, eag_mode=eag_mode),
    ]
