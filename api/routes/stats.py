"""Aggregate statistics endpoints — computed server-side over current records."""

from typing import Literal

from fastapi import APIRouter, Depends, Query

from api.db import fetch_all, fetch_one, get_conn
from api.schemas import (
    DistributionOut,
    HeadcountRow,
    SalaryStatsRow,
    TopEarnerRow,
)
from api.temporal import FOREVER_LITERAL

router = APIRouter(prefix="/stats", tags=["stats"])


def _salary_source(dept_no: str | None, title: str | None) -> tuple[str, str, list]:
    from_sql = (
        f"FROM salaries s "
        f"JOIN titles tt ON tt.emp_no = s.emp_no AND tt.to_date = '{FOREVER_LITERAL}' "
        f"JOIN dept_emp de ON de.emp_no = s.emp_no AND de.to_date = '{FOREVER_LITERAL}' "
        f"JOIN departments d ON d.dept_no = de.dept_no"
    )
    conds = [f"s.to_date = '{FOREVER_LITERAL}'"]
    params: list = []
    if dept_no:
        conds.append("de.dept_no = %s")
        params.append(dept_no)
    if title:
        conds.append("tt.title = %s")
        params.append(title)
    return from_sql, " AND ".join(conds), params


@router.get("/salaries", response_model=list[SalaryStatsRow])
def salary_stats(
    conn=Depends(get_conn),
    dept_no: str | None = Query(None, max_length=4),
    title: str | None = Query(None, max_length=50),
    group_by: Literal["none", "dept", "title"] = "none",
):
    from_sql, where, params = _salary_source(dept_no, title)
    if group_by == "none":
        row = fetch_one(
            conn,
            f"SELECT ROUND(AVG(s.salary), 2) AS avg_salary, MIN(s.salary) AS min_salary, "
            f"MAX(s.salary) AS max_salary, COUNT(*) AS count "
            f"{from_sql} WHERE {where}",
            params,
        )
        return [row]
    if group_by == "dept":
        return fetch_all(
            conn,
            f"SELECT d.dept_no, d.dept_name, ROUND(AVG(s.salary), 2) AS avg_salary, "
            f"MIN(s.salary) AS min_salary, MAX(s.salary) AS max_salary, COUNT(*) AS count "
            f"{from_sql} WHERE {where} "
            f"GROUP BY d.dept_no, d.dept_name ORDER BY d.dept_no",
            params,
        )
    return fetch_all(
        conn,
        f"SELECT tt.title, ROUND(AVG(s.salary), 2) AS avg_salary, "
        f"MIN(s.salary) AS min_salary, MAX(s.salary) AS max_salary, COUNT(*) AS count "
        f"{from_sql} WHERE {where} GROUP BY tt.title ORDER BY tt.title",
        params,
    )


@router.get("/headcount", response_model=list[HeadcountRow])
def headcount(conn=Depends(get_conn)):
    return fetch_all(
        conn,
        f"SELECT d.dept_no, d.dept_name, COUNT(de.emp_no) AS headcount "
        f"FROM departments d LEFT JOIN dept_emp de ON de.dept_no = d.dept_no "
        f"AND de.to_date = '{FOREVER_LITERAL}' "
        f"GROUP BY d.dept_no, d.dept_name ORDER BY d.dept_no",
    )


@router.get("/top-earners", response_model=list[TopEarnerRow])
def top_earners(
    conn=Depends(get_conn),
    dept_no: str | None = Query(None, max_length=4),
    title: str | None = Query(None, max_length=50),
    n: int = Query(10, ge=1, le=50),
):
    from_sql, where, params = _salary_source(dept_no, title)
    return fetch_all(
        conn,
        f"SELECT s.emp_no, e.first_name, e.last_name, s.salary, tt.title, d.dept_name "
        f"{from_sql} JOIN employees e ON e.emp_no = s.emp_no "
        f"WHERE {where} ORDER BY s.salary DESC, s.emp_no ASC LIMIT %s",
        (*params, n),
    )


@router.get("/salary-distribution", response_model=DistributionOut)
def salary_distribution(
    conn=Depends(get_conn),
    dept_no: str | None = Query(None, max_length=4),
    title: str | None = Query(None, max_length=50),
    buckets: int = Query(10, ge=1, le=50),
):
    from_sql, where, params = _salary_source(dept_no, title)
    salaries = [
        r["salary"]
        for r in fetch_all(conn, f"SELECT s.salary {from_sql} WHERE {where}", params)
    ]
    if not salaries:
        return {"min_salary": None, "max_salary": None, "total": 0, "buckets": []}
    lo, hi = min(salaries), max(salaries)
    width = (hi - lo) / buckets
    counts = [0] * buckets
    for v in salaries:
        idx = min(int((v - lo) / width), buckets - 1) if width else 0
        counts[idx] += 1
    out_buckets = [
        {
            "lower": int(lo + i * width),
            "upper": int(lo + (i + 1) * width),
            "count": counts[i],
        }
        for i in range(buckets)
    ]
    return {
        "min_salary": lo,
        "max_salary": hi,
        "total": len(salaries),
        "buckets": out_buckets,
    }
