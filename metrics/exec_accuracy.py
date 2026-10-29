"""Execution-equivalence scoring for text-to-SQL benchmarks (BIRD, Spider).

Runs both predicted and gold SQL against the same SQLite database and compares
result sets. This is the standard BIRD evaluation metric ("Ex-match").
"""

import sqlite3


def run_sql_readonly(
    db_path: str, sql: str, timeout: int = 30
) -> tuple[list[tuple], str | None]:
    """Execute SQL read-only against a SQLite database.

    Returns ``(rows, error_msg)`` where ``error_msg`` is None on success.
    """
    try:
        conn = sqlite3.connect(str(db_path), timeout=timeout)
        conn.execute("PRAGMA query_only = 1")
        cursor = conn.execute(sql)
        rows = cursor.fetchall()
        conn.close()
        return rows, None
    except Exception as e:
        return [], str(e)


def _normalize_value(val):
    """Normalize a cell value for comparison."""
    if isinstance(val, float):
        return round(val, 6)
    return val


def _cell_key(val):
    """Total-order sort key for any SQLite value.

    Orders None < numbers < strings < other (e.g. bytes), so a result column
    mixing NULLs with typed values sorts without cross-type comparison
    errors. The key fixes ordering only; equality is decided by
    ``_values_equal``.
    """
    if val is None:
        return (0, 0.0, "")
    if isinstance(val, (int, float)):
        return (1, float(val), "")
    if isinstance(val, str):
        return (2, 0.0, val)
    return (3, 0.0, repr(val))


def _values_equal(pred_val, gold_val, float_tol: float = 0.01) -> bool:
    """Compare two cell values with float tolerance and type coercion."""
    try:
        return abs(float(pred_val) - float(gold_val)) <= float_tol
    except (ValueError, TypeError):
        return str(pred_val).strip() == str(gold_val).strip()


def result_sets_equal(
    predicted: list[tuple],
    gold: list[tuple],
    float_tol: float = 0.01,
) -> bool:
    """Compare result sets as sorted lists of normalized tuples.

    Handles:
    - Order-insensitivity (rows sorted before comparison)
    - Float precision (tolerance-based comparison)
    - Type coercion (int vs float, string vs number)
    """
    if len(predicted) != len(gold):
        return False

    pred_sorted = sorted(
        (tuple(_normalize_value(v) for v in row) for row in predicted),
        key=lambda row: tuple(_cell_key(v) for v in row),
    )
    gold_sorted = sorted(
        (tuple(_normalize_value(v) for v in row) for row in gold),
        key=lambda row: tuple(_cell_key(v) for v in row),
    )

    for pred_row, gold_row in zip(pred_sorted, gold_sorted):
        if len(pred_row) != len(gold_row):
            return False
        for pv, gv in zip(pred_row, gold_row):
            if not _values_equal(pv, gv, float_tol):
                return False
    return True


def score_execution_accuracy(
    submitted_sql: str,
    gold_sql: str,
    db_path: str,
) -> dict:
    """Score by execution equivalence.

    Args:
        submitted_sql: The agent's final SQL (from ``submit_sql`` tool call).
        gold_sql: The gold SQL from the dataset.
        db_path: Path to the SQLite database to run both against.

    Returns:
        Dict with ``correct``, ``score``, ``metric``, and diagnostic fields.
    """
    pred_rows, pred_err = run_sql_readonly(db_path, submitted_sql)
    gold_rows, gold_err = run_sql_readonly(db_path, gold_sql)

    if pred_err:
        return {
            "correct": False,
            "score": 0.0,
            "metric": "execution_accuracy",
            "error": pred_err,
        }

    if gold_err:
        return {
            "correct": False,
            "score": 0.0,
            "metric": "execution_accuracy",
            "gold_error": gold_err,
        }

    correct = result_sets_equal(pred_rows, gold_rows)
    return {
        "correct": correct,
        "score": 1.0 if correct else 0.0,
        "metric": "execution_accuracy",
        "pred_row_count": len(pred_rows),
        "gold_row_count": len(gold_rows),
    }
