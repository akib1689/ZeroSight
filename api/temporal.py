"""Temporal helpers — half-open ranges and close-and-insert mutations.

History tables (salaries, titles, dept_emp, dept_manager) store validity as
half-open ranges [from_date, to_date); the current row carries
to_date = 9999-01-01. Mutations never update values in place or delete:
they close the current row at effective_date and insert a new one, keeping
ranges contiguous.
"""

from datetime import date

from fastapi import HTTPException

FOREVER_LITERAL = "9999-01-01"
FOREVER = date(9999, 1, 1)


def temporal_predicate(
    at_date: date | None,
    from_date: date | None,
    to_date: date | None,
    prefix: str = "",
) -> tuple[str, list]:
    """WHERE fragment selecting rows for the requested time window.

    No params → current rows only. at_date → rows effective at that date.
    from_date/to_date → rows overlapping the (half-open) range.
    """
    if at_date is not None and (from_date is not None or to_date is not None):
        raise HTTPException(
            status_code=400, detail="at_date cannot be combined with from_date/to_date"
        )
    f_col, t_col = f"{prefix}from_date", f"{prefix}to_date"
    if at_date is not None:
        return f"{f_col} <= %s AND {t_col} > %s", [at_date, at_date]
    if from_date is not None or to_date is not None:
        upper = to_date.isoformat() if to_date is not None else FOREVER_LITERAL
        lower = from_date.isoformat() if from_date is not None else "1000-01-01"
        return f"{f_col} < %s AND {t_col} > %s", [upper, lower]
    return f"{t_col} = '{FOREVER_LITERAL}'", []


def close_current(
    conn,
    table: str,
    match_sql: str,
    match_params: list,
    effective: date,
) -> int:
    """Close the current row(s) of a history table at effective date."""
    sql = (
        f"UPDATE {table} SET to_date = %s "
        f"WHERE to_date = '{FOREVER_LITERAL}' AND ({match_sql})"
    )
    with conn.cursor() as cur:
        cur.execute(sql, [effective, *match_params])
        return cur.rowcount
