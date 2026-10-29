"""Employee endpoints — CRUD plus temporal sub-resources.

Mutations on temporal tables (salaries, titles, dept_emp, dept_manager)
follow the close-and-insert convention implemented in ``api/temporal.py``;
the ``employees`` table itself is plain CRUD (non-temporal).
"""

from datetime import date
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from api.db import execute, fetch_all, fetch_one, get_conn
from api.schemas import (
    CurrentDeptRef,
    EmployeeCreate,
    EmployeeListItem,
    EmployeeOut,
    EmployeeUpdate,
    MembershipRecord,
    Page,
    SalaryRecord,
    SalaryUpdate,
    TitleRecord,
    TitleUpdate,
    TransferRequest,
)
from api.temporal import FOREVER, FOREVER_LITERAL, close_current, temporal_predicate

router = APIRouter(prefix="/employees", tags=["employees"])

_SORT_COLUMNS = {
    "emp_no": "e.emp_no",
    "last_name": "e.last_name",
    "hire_date": "e.hire_date",
    "salary": "s.salary",
}


def _employee_row(conn, emp_no: int) -> dict | None:
    return fetch_one(conn, "SELECT * FROM employees WHERE emp_no = %s", (emp_no,))


def _current_row(conn, table: str, emp_no: int) -> dict | None:
    return fetch_one(
        conn,
        f"SELECT * FROM {table} WHERE emp_no = %s AND to_date = '{FOREVER_LITERAL}'",
        (emp_no,),
    )


def _known_titles(conn) -> set[str]:
    return {r["title"] for r in fetch_all(conn, "SELECT DISTINCT title FROM titles")}


def _employee_detail(conn, emp_no: int) -> dict:
    emp = _employee_row(conn, emp_no)
    if emp is None:
        raise HTTPException(status_code=404, detail=f"Employee {emp_no} not found")
    salary_row = fetch_one(
        conn,
        f"SELECT salary, from_date, to_date FROM salaries "
        f"WHERE emp_no = %s AND to_date = '{FOREVER_LITERAL}'",
        (emp_no,),
    )
    title_row = fetch_one(
        conn,
        f"SELECT title, from_date, to_date FROM titles "
        f"WHERE emp_no = %s AND to_date = '{FOREVER_LITERAL}'",
        (emp_no,),
    )
    dept_row = fetch_one(
        conn,
        f"SELECT d.dept_no, d.dept_name, de.from_date, de.to_date "
        f"FROM dept_emp de JOIN departments d ON d.dept_no = de.dept_no "
        f"WHERE de.emp_no = %s AND de.to_date = '{FOREVER_LITERAL}'",
        (emp_no,),
    )
    return {
        "emp_no": emp["emp_no"],
        "first_name": emp["first_name"],
        "last_name": emp["last_name"],
        "gender": emp["gender"],
        "birth_date": emp["birth_date"],
        "hire_date": emp["hire_date"],
        "active": dept_row is not None,
        "salary": salary_row["salary"] if salary_row else None,
        "title": (
            {
                "title": title_row["title"],
                "from_date": title_row["from_date"],
                "to_date": title_row["to_date"],
            }
            if title_row
            else None
        ),
        "department": (
            {
                "dept_no": dept_row["dept_no"],
                "dept_name": dept_row["dept_name"],
                "from_date": dept_row["from_date"],
                "to_date": dept_row["to_date"],
            }
            if dept_row
            else None
        ),
    }


@router.get("", response_model=Page[EmployeeListItem])
def list_employees(
    conn=Depends(get_conn),
    name: str | None = Query(None, max_length=50),
    first_name: str | None = Query(None, max_length=50),
    last_name: str | None = Query(None, max_length=50),
    emp_no: int | None = None,
    dept_no: str | None = Query(None, max_length=4),
    title: str | None = Query(None, max_length=50),
    gender: Literal["M", "F"] | None = None,
    hired_after: date | None = None,
    hired_before: date | None = None,
    min_salary: int | None = Query(None, gt=0),
    max_salary: int | None = Query(None, gt=0),
    active_at: date | None = None,
    sort: Literal["emp_no", "last_name", "hire_date", "salary"] = "emp_no",
    order: Literal["asc", "desc"] = "asc",
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    if active_at is not None:
        dept_join = (
            "LEFT JOIN dept_emp de ON de.emp_no = e.emp_no "
            f"AND de.from_date <= '{active_at}' AND de.to_date > '{active_at}'"
        )
    else:
        dept_join = (
            f"LEFT JOIN dept_emp de ON de.emp_no = e.emp_no "
            f"AND de.to_date = '{FOREVER_LITERAL}'"
        )
    joins = (
        f"LEFT JOIN salaries s ON s.emp_no = e.emp_no AND s.to_date = '{FOREVER_LITERAL}' "
        f"LEFT JOIN titles tt ON tt.emp_no = e.emp_no AND tt.to_date = '{FOREVER_LITERAL}' "
        f"{dept_join} "
        f"LEFT JOIN departments dp ON dp.dept_no = de.dept_no"
    )
    conds = ["1=1"]
    params: list = []
    if name:
        conds.append("(e.first_name LIKE %s OR e.last_name LIKE %s)")
        like = f"%{name}%"
        params += [like, like]
    if first_name:
        conds.append("e.first_name LIKE %s")
        params.append(f"%{first_name}%")
    if last_name:
        conds.append("e.last_name LIKE %s")
        params.append(f"%{last_name}%")
    if emp_no is not None:
        conds.append("e.emp_no = %s")
        params.append(emp_no)
    if dept_no:
        conds.append("de.dept_no = %s")
        params.append(dept_no)
    if title:
        conds.append("tt.title = %s")
        params.append(title)
    if gender:
        conds.append("e.gender = %s")
        params.append(gender)
    if hired_after:
        conds.append("e.hire_date >= %s")
        params.append(hired_after)
    if hired_before:
        conds.append("e.hire_date <= %s")
        params.append(hired_before)
    if min_salary is not None:
        conds.append("s.salary >= %s")
        params.append(min_salary)
    if max_salary is not None:
        conds.append("s.salary <= %s")
        params.append(max_salary)
    where = " AND ".join(conds)

    total = fetch_one(
        conn, f"SELECT COUNT(*) AS n FROM employees e {joins} WHERE {where}", params
    )["n"]

    sort_col = _SORT_COLUMNS[sort]
    null_guard = f"{sort_col} IS NULL," if sort == "salary" else ""
    direction = "ASC" if order == "asc" else "DESC"
    offset = (page - 1) * page_size
    rows = fetch_all(
        conn,
        f"SELECT e.emp_no, e.first_name, e.last_name, e.gender, e.hire_date, "
        f"s.salary AS salary, tt.title AS title, dp.dept_no AS dept_no, "
        f"dp.dept_name AS dept_name, de.emp_no IS NOT NULL AS active "
        f"FROM employees e {joins} WHERE {where} "
        f"ORDER BY {null_guard} {sort_col} {direction}, e.emp_no ASC "
        f"LIMIT %s OFFSET %s",
        [*params, page_size, offset],
    )
    items = [{**r, "active": bool(r["active"])} for r in rows]
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@router.post("", response_model=EmployeeOut, status_code=201)
def hire_employee(body: EmployeeCreate, conn=Depends(get_conn)):
    dept = fetch_one(
        conn, "SELECT dept_no FROM departments WHERE dept_no = %s", (body.dept_no,)
    )
    if dept is None:
        raise HTTPException(status_code=400, detail=f"Unknown dept_no '{body.dept_no}'")
    titles = _known_titles(conn)
    if body.title not in titles:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown title '{body.title}'. Valid titles: {sorted(titles)}",
        )
    if body.hire_date <= body.birth_date:
        raise HTTPException(status_code=400, detail="hire_date must be after birth_date")
    emp_no = (fetch_one(conn, "SELECT MAX(emp_no) AS m FROM employees")["m"] or 500000) + 1
    execute(
        conn,
        "INSERT INTO employees (emp_no, first_name, last_name, gender, birth_date, hire_date) "
        "VALUES (%s, %s, %s, %s, %s, %s)",
        (emp_no, body.first_name, body.last_name, body.gender, body.birth_date, body.hire_date),
    )
    execute(
        conn,
        "INSERT INTO salaries (emp_no, salary, from_date, to_date) "
        f"VALUES (%s, %s, %s, '{FOREVER_LITERAL}')",
        (emp_no, body.salary, body.hire_date),
    )
    execute(
        conn,
        "INSERT INTO titles (emp_no, title, from_date, to_date) "
        f"VALUES (%s, %s, %s, '{FOREVER_LITERAL}')",
        (emp_no, body.title, body.hire_date),
    )
    execute(
        conn,
        "INSERT INTO dept_emp (emp_no, dept_no, from_date, to_date) "
        f"VALUES (%s, %s, %s, '{FOREVER_LITERAL}')",
        (emp_no, body.dept_no, body.hire_date),
    )
    conn.commit()
    return _employee_detail(conn, emp_no)


@router.get("/{emp_no}", response_model=EmployeeOut)
def get_employee(emp_no: int, conn=Depends(get_conn)):
    return _employee_detail(conn, emp_no)


@router.patch("/{emp_no}", response_model=EmployeeOut)
def update_employee(emp_no: int, body: EmployeeUpdate, conn=Depends(get_conn)):
    if _employee_row(conn, emp_no) is None:
        raise HTTPException(status_code=404, detail=f"Employee {emp_no} not found")
    fields = {k: v for k, v in body.model_dump().items() if v is not None}
    if not fields:
        raise HTTPException(status_code=400, detail="No fields to update")
    sets = ", ".join(f"{k} = %s" for k in fields)
    execute(conn, f"UPDATE employees SET {sets} WHERE emp_no = %s", (*fields.values(), emp_no))
    conn.commit()
    return _employee_detail(conn, emp_no)


@router.delete("/{emp_no}", status_code=204)
def terminate_employee(emp_no: int, conn=Depends(get_conn)):
    if _employee_row(conn, emp_no) is None:
        raise HTTPException(status_code=404, detail=f"Employee {emp_no} not found")
    today = date.today()
    for table in ("salaries", "titles", "dept_emp", "dept_manager"):
        close_current(conn, table, "emp_no = %s", (emp_no,), today)
    conn.commit()
    return Response(status_code=204)


@router.get("/{emp_no}/salaries", response_model=list[SalaryRecord])
def salary_history(
    emp_no: int,
    at_date: date | None = None,
    from_date: date | None = None,
    to_date: date | None = None,
    conn=Depends(get_conn),
):
    if _employee_row(conn, emp_no) is None:
        raise HTTPException(status_code=404, detail=f"Employee {emp_no} not found")
    pred, params = temporal_predicate(at_date, from_date, to_date, prefix="s.")
    return fetch_all(
        conn,
        f"SELECT s.salary, s.from_date, s.to_date FROM salaries s "
        f"WHERE s.emp_no = %s AND {pred} ORDER BY s.from_date",
        (emp_no, *params),
    )


@router.put("/{emp_no}/salary", response_model=SalaryRecord)
def set_salary(emp_no: int, body: SalaryUpdate, conn=Depends(get_conn)):
    if _employee_row(conn, emp_no) is None:
        raise HTTPException(status_code=404, detail=f"Employee {emp_no} not found")
    current = _current_row(conn, "salaries", emp_no)
    if current is None:
        raise HTTPException(status_code=409, detail="Employee has no current salary record")
    effective = body.effective_date or date.today()
    if effective <= current["from_date"]:
        raise HTTPException(
            status_code=400,
            detail=f"effective_date must be after current record start "
            f"{current['from_date']}",
        )
    close_current(conn, "salaries", "emp_no = %s", (emp_no,), effective)
    execute(
        conn,
        "INSERT INTO salaries (emp_no, salary, from_date, to_date) "
        f"VALUES (%s, %s, %s, '{FOREVER_LITERAL}')",
        (emp_no, body.new_salary, effective),
    )
    conn.commit()
    return SalaryRecord(salary=body.new_salary, from_date=effective, to_date=FOREVER)


@router.get("/{emp_no}/titles", response_model=list[TitleRecord])
def title_history(
    emp_no: int,
    at_date: date | None = None,
    from_date: date | None = None,
    to_date: date | None = None,
    conn=Depends(get_conn),
):
    if _employee_row(conn, emp_no) is None:
        raise HTTPException(status_code=404, detail=f"Employee {emp_no} not found")
    pred, params = temporal_predicate(at_date, from_date, to_date, prefix="t.")
    return fetch_all(
        conn,
        f"SELECT t.title, t.from_date, t.to_date FROM titles t "
        f"WHERE t.emp_no = %s AND {pred} ORDER BY t.from_date",
        (emp_no, *params),
    )


@router.put("/{emp_no}/title", response_model=TitleRecord)
def set_title(emp_no: int, body: TitleUpdate, conn=Depends(get_conn)):
    if _employee_row(conn, emp_no) is None:
        raise HTTPException(status_code=404, detail=f"Employee {emp_no} not found")
    titles = _known_titles(conn)
    if body.new_title not in titles:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown title '{body.new_title}'. Valid titles: {sorted(titles)}",
        )
    current = _current_row(conn, "titles", emp_no)
    if current is None:
        raise HTTPException(status_code=409, detail="Employee has no current title record")
    effective = body.effective_date or date.today()
    if effective <= current["from_date"]:
        raise HTTPException(
            status_code=400,
            detail=f"effective_date must be after current record start "
            f"{current['from_date']}",
        )
    close_current(conn, "titles", "emp_no = %s", (emp_no,), effective)
    execute(
        conn,
        "INSERT INTO titles (emp_no, title, from_date, to_date) "
        f"VALUES (%s, %s, %s, '{FOREVER_LITERAL}')",
        (emp_no, body.new_title, effective),
    )
    conn.commit()
    return TitleRecord(title=body.new_title, from_date=effective, to_date=FOREVER)


@router.get("/{emp_no}/departments", response_model=list[MembershipRecord])
def department_history(
    emp_no: int,
    at_date: date | None = None,
    from_date: date | None = None,
    to_date: date | None = None,
    conn=Depends(get_conn),
):
    if _employee_row(conn, emp_no) is None:
        raise HTTPException(status_code=404, detail=f"Employee {emp_no} not found")
    pred, params = temporal_predicate(at_date, from_date, to_date, prefix="de.")
    return fetch_all(
        conn,
        f"SELECT d.dept_no, d.dept_name, de.from_date, de.to_date "
        f"FROM dept_emp de JOIN departments d ON d.dept_no = de.dept_no "
        f"WHERE de.emp_no = %s AND {pred} ORDER BY de.from_date",
        (emp_no, *params),
    )


@router.post("/{emp_no}/transfer", response_model=CurrentDeptRef)
def transfer_employee(emp_no: int, body: TransferRequest, conn=Depends(get_conn)):
    if _employee_row(conn, emp_no) is None:
        raise HTTPException(status_code=404, detail=f"Employee {emp_no} not found")
    dept = fetch_one(
        conn, "SELECT dept_no, dept_name FROM departments WHERE dept_no = %s", (body.dept_no,)
    )
    if dept is None:
        raise HTTPException(status_code=400, detail=f"Unknown dept_no '{body.dept_no}'")
    current = _current_row(conn, "dept_emp", emp_no)
    if current is None:
        raise HTTPException(
            status_code=409, detail="Employee has no current department membership"
        )
    if current["dept_no"] == body.dept_no:
        raise HTTPException(
            status_code=409, detail=f"Employee already belongs to '{body.dept_no}'"
        )
    effective = body.effective_date or date.today()
    if effective <= current["from_date"]:
        raise HTTPException(
            status_code=400,
            detail=f"effective_date must be after current record start "
            f"{current['from_date']}",
        )
    close_current(conn, "dept_emp", "emp_no = %s", (emp_no,), effective)
    execute(
        conn,
        "INSERT INTO dept_emp (emp_no, dept_no, from_date, to_date) "
        f"VALUES (%s, %s, %s, '{FOREVER_LITERAL}')",
        (emp_no, body.dept_no, effective),
    )
    conn.commit()
    return CurrentDeptRef(
        dept_no=dept["dept_no"],
        dept_name=dept["dept_name"],
        from_date=effective,
        to_date=FOREVER,
    )
