"""Department endpoints — listing, detail, members, managers."""

from datetime import date
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query

from api.db import execute, fetch_all, fetch_one, get_conn
from api.schemas import (
    DepartmentDetail,
    DepartmentListItem,
    DeptMemberItem,
    ManagerAssignRequest,
    ManagerRecord,
    Page,
)
from api.temporal import FOREVER, FOREVER_LITERAL, close_current, temporal_predicate

router = APIRouter(prefix="/departments", tags=["departments"])


def _dept_row(conn, dept_no: str) -> dict | None:
    return fetch_one(
        conn, "SELECT dept_no, dept_name FROM departments WHERE dept_no = %s", (dept_no,)
    )


@router.get("", response_model=list[DepartmentListItem])
def list_departments(
    include: Literal["headcount"] | None = None,
    conn=Depends(get_conn),
):
    if include == "headcount":
        return fetch_all(
            conn,
            f"SELECT d.dept_no, d.dept_name, COUNT(de.emp_no) AS headcount "
            f"FROM departments d LEFT JOIN dept_emp de ON de.dept_no = d.dept_no "
            f"AND de.to_date = '{FOREVER_LITERAL}' "
            f"GROUP BY d.dept_no, d.dept_name ORDER BY d.dept_no",
        )
    return fetch_all(conn, "SELECT dept_no, dept_name FROM departments ORDER BY dept_no")


@router.get("/{dept_no}", response_model=DepartmentDetail)
def department_detail(dept_no: str, conn=Depends(get_conn)):
    dept = _dept_row(conn, dept_no)
    if dept is None:
        raise HTTPException(status_code=404, detail=f"Department '{dept_no}' not found")
    manager = fetch_one(
        conn,
        f"SELECT e.emp_no, e.first_name, e.last_name, dm.from_date, dm.to_date "
        f"FROM dept_manager dm JOIN employees e ON e.emp_no = dm.emp_no "
        f"WHERE dm.dept_no = %s AND dm.to_date = '{FOREVER_LITERAL}'",
        (dept_no,),
    )
    headcount = fetch_one(
        conn,
        f"SELECT COUNT(*) AS n FROM dept_emp "
        f"WHERE dept_no = %s AND to_date = '{FOREVER_LITERAL}'",
        (dept_no,),
    )["n"]
    return {
        "dept_no": dept["dept_no"],
        "dept_name": dept["dept_name"],
        "manager": manager,
        "headcount": headcount,
    }


@router.get("/{dept_no}/employees", response_model=Page[DeptMemberItem])
def department_members(
    dept_no: str,
    conn=Depends(get_conn),
    at_date: date | None = None,
    from_date: date | None = None,
    to_date: date | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    if _dept_row(conn, dept_no) is None:
        raise HTTPException(status_code=404, detail=f"Department '{dept_no}' not found")
    pred, params = temporal_predicate(at_date, from_date, to_date, prefix="de.")
    where = f"de.dept_no = %s AND {pred}"
    from_tables = "dept_emp de JOIN employees e ON e.emp_no = de.emp_no"
    total = fetch_one(
        conn, f"SELECT COUNT(*) AS n FROM {from_tables} WHERE {where}", (dept_no, *params)
    )["n"]
    offset = (page - 1) * page_size
    rows = fetch_all(
        conn,
        f"SELECT e.emp_no, e.first_name, e.last_name, e.gender, e.hire_date, "
        f"de.from_date AS membership_from, de.to_date AS membership_to "
        f"FROM {from_tables} WHERE {where} "
        f"ORDER BY e.emp_no LIMIT %s OFFSET %s",
        (dept_no, *params, page_size, offset),
    )
    return {"items": rows, "total": total, "page": page, "page_size": page_size}


@router.get("/{dept_no}/managers", response_model=list[ManagerRecord])
def manager_history(
    dept_no: str,
    at_date: date | None = None,
    from_date: date | None = None,
    to_date: date | None = None,
    conn=Depends(get_conn),
):
    if _dept_row(conn, dept_no) is None:
        raise HTTPException(status_code=404, detail=f"Department '{dept_no}' not found")
    pred, params = temporal_predicate(at_date, from_date, to_date, prefix="dm.")
    return fetch_all(
        conn,
        f"SELECT e.emp_no, e.first_name, e.last_name, dm.from_date, dm.to_date "
        f"FROM dept_manager dm JOIN employees e ON e.emp_no = dm.emp_no "
        f"WHERE dm.dept_no = %s AND {pred} ORDER BY dm.from_date",
        (dept_no, *params),
    )


@router.put("/{dept_no}/manager", response_model=ManagerRecord)
def assign_manager(dept_no: str, body: ManagerAssignRequest, conn=Depends(get_conn)):
    if _dept_row(conn, dept_no) is None:
        raise HTTPException(status_code=404, detail=f"Department '{dept_no}' not found")
    emp = fetch_one(conn, "SELECT * FROM employees WHERE emp_no = %s", (body.emp_no,))
    if emp is None:
        raise HTTPException(status_code=404, detail=f"Employee {body.emp_no} not found")
    membership = fetch_one(
        conn,
        f"SELECT 1 FROM dept_emp WHERE emp_no = %s AND dept_no = %s "
        f"AND to_date = '{FOREVER_LITERAL}'",
        (body.emp_no, dept_no),
    )
    if membership is None:
        raise HTTPException(
            status_code=409,
            detail=f"Employee {body.emp_no} is not a current member of '{dept_no}'",
        )
    current = fetch_one(
        conn,
        f"SELECT from_date FROM dept_manager "
        f"WHERE dept_no = %s AND to_date = '{FOREVER_LITERAL}'",
        (dept_no,),
    )
    effective = body.effective_date or date.today()
    if current is not None and effective <= current["from_date"]:
        raise HTTPException(
            status_code=400,
            detail=f"effective_date must be after current record start "
            f"{current['from_date']}",
        )
    close_current(conn, "dept_manager", "dept_no = %s", (dept_no,), effective)
    execute(
        conn,
        "INSERT INTO dept_manager (dept_no, emp_no, from_date, to_date) "
        f"VALUES (%s, %s, %s, '{FOREVER_LITERAL}')",
        (dept_no, body.emp_no, effective),
    )
    conn.commit()
    return ManagerRecord(
        emp_no=body.emp_no,
        first_name=emp["first_name"],
        last_name=emp["last_name"],
        from_date=effective,
        to_date=FOREVER,
    )
