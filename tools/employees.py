"""Employees REST API tools — EAG-wrapped agent interface over the API layer.

Two consumers exist for the employees DB: this tool layer (agent-facing) and
the FastAPI layer in ``api/`` (frontend-facing). These tools are thin HTTP
clients over that API — they hold no DB connection and add no business logic,
so both consumers observe identical semantics.

Exposure thesis: the observation content is what leaks to the inference
provider. In **baseline** mode ``_run`` returns the raw JSON body (measured
via ``observation_tokens`` per step). In **EAG** mode ``generate_metadata``
replaces it with a shape-only view — keys, types, item counts — while the
parsed body rides the ``ToolMessage.artifact`` for local placeholder
resolution (``{{ search_employees.items[0].emp_no }}``). The referential
chain stays connected; raw values never enter the model context.

Classification (``exposure_type``):
  metadata   — get_reference (titles + department names: whitelisted
               reference data needed to compose arguments)
  data       — every read of employee/department records, stats, and every
               mutation response (redacted in EAG; ``data_redacted`` is the
               effective accounting type)
  evaluation — submit_answer

Mutations commit through the API (no rollback here). Benchmark isolation for
mutation tasks is a dataset-adapter concern (reset between tasks), not a
tool-layer one.
"""

import json
import os
from datetime import date
from typing import Any, Literal

import httpx
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from tools.evaluator import make_evaluator_tools
from tools.sql import EAGBaseTool, ToolResult

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
_TIMEOUT = 30.0


def _base_url() -> str:
    return os.environ.get("EMPLOYEES_API_URL", DEFAULT_BASE_URL)


def _qs(params: dict) -> dict:
    """Serialize query params — dates → ISO strings, drop empties."""
    out = {}
    for k, v in params.items():
        if v is None or v == "":
            continue
        out[k] = v.isoformat() if isinstance(v, date) else str(v)
    return out


# ── EAG redaction ────────────────────────────────────────────────────────────


def _scalar_type(value: Any) -> str:
    if value is None:
        return "<null>"
    if isinstance(value, bool):
        return "<bool>"
    if isinstance(value, int):
        return "<int>"
    if isinstance(value, float):
        return "<float>"
    return "<str>"


def _shape_of(value: Any, depth: int = 0) -> str:
    """Describe the structure of a parsed JSON body without its values."""
    if isinstance(value, dict):
        if depth >= 4:
            return "{...}"
        inner = ", ".join(f"{k}: {_shape_of(v, depth + 1)}" for k, v in value.items())
        return "{" + inner + "}"
    if isinstance(value, list):
        if not value:
            return "[] (empty)"
        return f"[{len(value)} items, each {_shape_of(value[0], depth + 1)}]"
    return _scalar_type(value)


def _path_prefix(data: Any) -> str:
    """Leading path fragment for placeholder hints."""
    if isinstance(data, list):
        return "[<i>]"
    if isinstance(data, dict):
        return ".<key>"
    return ""


# ── Input schemas ────────────────────────────────────────────────────────────


class _SearchInput(BaseModel):
    name: str | None = Field(None, description="Substring matched against first OR last name")
    first_name: str | None = Field(
        None, description="Substring matched against first name only — use with last_name "
        "to narrow a full name to a unique candidate"
    )
    last_name: str | None = Field(
        None, description="Substring matched against last name only"
    )
    emp_no: int | None = None
    dept_no: str | None = Field(None, description="Department number, e.g. d005")
    title: str | None = Field(None, description="Exact current title, e.g. Engineer")
    gender: Literal["M", "F"] | None = None
    hired_after: date | None = Field(
        None, description="hire_date >= this date (inclusive)")
    hired_before: date | None = Field(
        None, description="hire_date <= this date (inclusive)")
    min_salary: int | None = Field(
        None, description="Filter on current salary (inclusive lower bound)")
    max_salary: int | None = Field(
        None, description="Filter on current salary (inclusive upper bound)")
    active_at: date | None = Field(
        None, description="Point-in-time: only employees in a department on this date"
    )
    sort: Literal["emp_no", "last_name", "hire_date", "salary"] = "emp_no"
    order: Literal["asc", "desc"] = "asc"
    page: int = Field(1, ge=1)
    page_size: int = Field(10, ge=1, le=100)


class _EmpNoInput(BaseModel):
    emp_no: int = Field(..., description="Employee number")


class _HistoryInput(BaseModel):
    emp_no: int = Field(..., description="Employee number")
    at_date: date | None = Field(
        None, description="Rows effective on this date. Mutually exclusive with range"
    )
    from_date: date | None = Field(
        None, description="Range start (inclusive); with to_date selects overlapping rows"
    )
    to_date: date | None = Field(None, description="Range end (exclusive)")


class _SalaryInput(BaseModel):
    emp_no: int = Field(..., description="Employee number")
    new_salary: int = Field(..., gt=0, description="New annual salary")
    effective_date: date | None = Field(
        None, description="Defaults to today; must be after current record start"
    )


class _TitleInput(BaseModel):
    emp_no: int = Field(..., description="Employee number")
    new_title: str = Field(..., description="Exact title (see get_reference for valid values)")
    effective_date: date | None = None


class _TransferInput(BaseModel):
    emp_no: int = Field(..., description="Employee number")
    dept_no: str = Field(..., description="Destination department number")
    effective_date: date | None = None


class _HireInput(BaseModel):
    first_name: str = Field(..., max_length=14)
    last_name: str = Field(..., max_length=14)
    gender: Literal["M", "F"]
    birth_date: date
    hire_date: date
    dept_no: str
    salary: int = Field(..., gt=0)
    title: str = Field(..., description="Exact title (see get_reference)")


class _DeptNoInput(BaseModel):
    dept_no: str = Field(..., description="Department number, e.g. d005")


class _DeptMembersInput(BaseModel):
    dept_no: str
    at_date: date | None = Field(
        None, description="Membership on this date; default current"
    )
    from_date: date | None = None
    to_date: date | None = None
    page: int = Field(1, ge=1)
    page_size: int = Field(20, ge=1, le=100)


class _DeptHistoryInput(BaseModel):
    dept_no: str
    at_date: date | None = None
    from_date: date | None = None
    to_date: date | None = None


class _ManagerInput(BaseModel):
    dept_no: str
    emp_no: int = Field(..., description="Must currently belong to the department")
    effective_date: date | None = None


class _StatsInput(BaseModel):
    dept_no: str | None = None
    title: str | None = None
    group_by: Literal["none", "dept", "title"] = "none"


class _TopEarnersInput(BaseModel):
    dept_no: str | None = None
    title: str | None = None
    n: int = Field(10, ge=1, le=50)


class _DistributionInput(BaseModel):
    dept_no: str | None = None
    title: str | None = None
    buckets: int = Field(10, ge=1, le=50)


class _AnswerInput(BaseModel):
    # Accepts numbers too: EAG middleware resolves full-string placeholders
    # type-preservingly (e.g. {{ get_employee.salary }} → int), and a
    # pydantic str-only field would reject the call — the retry loop that
    # follows leaks the resolved value through ToolNode's error echo.
    answer: str | int | float = Field(
        ..., description="Final answer to submit for evaluation"
    )


# ── Base tool ────────────────────────────────────────────────────────────────


class EmployeesAPITool(EAGBaseTool):
    """Base: HTTP call → ToolResult; EAG content = shape-only redaction."""

    api_base: str = Field(default_factory=_base_url)

    def _request(
        self,
        method: str,
        path: str,
        params: dict | None = None,
        json_body: dict | None = None,
    ) -> ToolResult:
        try:
            with httpx.Client(base_url=self.api_base, timeout=_TIMEOUT) as client:
                resp = client.request(method, path, params=_qs(params or {}), json=json_body)
        except httpx.HTTPError as exc:
            return ToolResult(
                output=f"API request failed: {exc}", error=f"connection error: {exc}"
            )
        if resp.status_code == 204:
            return ToolResult(output='{"ok": true}', data={"ok": True})
        try:
            body = resp.json()
        except ValueError:
            body = {"detail": resp.text}
        if resp.is_success:
            n = len(body) if isinstance(body, list) else len(body.get("items", [])) \
                if isinstance(body, dict) else 0
            return ToolResult(
                output=json.dumps(body, ensure_ascii=False),
                data=body,
                row_count=n,
            )
        detail = body.get("detail", resp.text) if isinstance(body, dict) else resp.text
        return ToolResult(
            output=f"HTTP {resp.status_code}: {detail}",
            error=f"HTTP {resp.status_code}",
        )

    def generate_metadata(self, result: ToolResult) -> str:
        if result.error:
            # Error details echo model-chosen arguments / reference lists —
            # safe to surface, and required for retry recovery.
            return f"[EAG] {self.name} failed — {result.output}"
        data = result.data
        if data is None:
            return f"[EAG] {self.name} → OK (empty response)"
        prefix = _path_prefix(data)
        return (
            f"[EAG] {self.name} → OK\n"
            f"Structure: {_shape_of(data)}\n"
            f"(field order follows the API response; fields may be null)\n"
            f"Raw values withheld — reference them symbolically, "
            f"e.g. {{{{ {self.name}{prefix} }}}}"
        )


# ── Metadata tool ────────────────────────────────────────────────────────────


class GetReferenceTool(EmployeesAPITool):
    """Whitelisted reference data: valid titles and departments."""

    name: str = "get_reference"
    description: str = (
        "Get reference data: all valid job titles and all departments "
        "(dept_no + dept_name). Use for composing arguments that require a "
        "valid title or department number."
    )
    exposure_type: str = "metadata"

    def execute(self, **kwargs) -> ToolResult:
        titles = self._request("GET", "/api/titles")
        if titles.error:
            return titles
        depts = self._request("GET", "/api/departments")
        if depts.error:
            return depts
        data = {"titles": titles.data, "departments": depts.data}
        return ToolResult(
            output=json.dumps(data, ensure_ascii=False),
            data=data,
            row_count=len(depts.data or []),
        )


# ── Record tools ─────────────────────────────────────────────────────────────


class SearchEmployeesTool(EmployeesAPITool):
    name: str = "search_employees"
    description: str = (
        "Search employees by name substring (first or last) plus optional "
        "filters: department, title, gender, hire window, current salary "
        "range, point-in-time activity. Paginated; sorted per sort/order."
    )
    args_schema: type[BaseModel] = _SearchInput
    exposure_type: str = "data"

    def execute(self, **kwargs) -> ToolResult:
        return self._request("GET", "/api/employees", params=kwargs)


class GetEmployeeTool(EmployeesAPITool):
    name: str = "get_employee"
    description: str = (
        "Get one employee's profile: names, gender, birth/hire dates, "
        "active status, current salary, current title, current department."
    )
    args_schema: type[BaseModel] = _EmpNoInput
    exposure_type: str = "data"

    def execute(self, **kwargs) -> ToolResult:
        return self._request("GET", f"/api/employees/{kwargs['emp_no']}")


class SalaryHistoryTool(EmployeesAPITool):
    name: str = "get_salary_history"
    description: str = (
        "Get an employee's salary records. Default: current record only. "
        "Use at_date for a point-in-time value, or from_date/to_date for "
        "the full history overlapping that range."
    )
    args_schema: type[BaseModel] = _HistoryInput
    exposure_type: str = "data"

    def execute(self, **kwargs) -> ToolResult:
        emp_no = kwargs.pop("emp_no")
        return self._request("GET", f"/api/employees/{emp_no}/salaries", params=kwargs)


class TitleHistoryTool(EmployeesAPITool):
    name: str = "get_title_history"
    description: str = (
        "Get an employee's title records. Default: current record only. "
        "at_date for point-in-time, from_date/to_date for a range."
    )
    args_schema: type[BaseModel] = _HistoryInput
    exposure_type: str = "data"

    def execute(self, **kwargs) -> ToolResult:
        emp_no = kwargs.pop("emp_no")
        return self._request("GET", f"/api/employees/{emp_no}/titles", params=kwargs)


class DepartmentHistoryTool(EmployeesAPITool):
    name: str = "get_department_history"
    description: str = (
        "Get an employee's department membership records. Default: current "
        "only. at_date for point-in-time, from_date/to_date for a range."
    )
    args_schema: type[BaseModel] = _HistoryInput
    exposure_type: str = "data"

    def execute(self, **kwargs) -> ToolResult:
        emp_no = kwargs.pop("emp_no")
        return self._request("GET", f"/api/employees/{emp_no}/departments", params=kwargs)


class GetDepartmentTool(EmployeesAPITool):
    name: str = "get_department"
    description: str = (
        "Get department detail: name, current manager (name + tenure), "
        "current headcount."
    )
    args_schema: type[BaseModel] = _DeptNoInput
    exposure_type: str = "data"

    def execute(self, **kwargs) -> ToolResult:
        return self._request("GET", f"/api/departments/{kwargs['dept_no']}")


class DepartmentMembersTool(EmployeesAPITool):
    name: str = "get_department_employees"
    description: str = (
        "List members of a department, paginated. Default: current members; "
        "at_date for point-in-time membership (or from_date/to_date range)."
    )
    args_schema: type[BaseModel] = _DeptMembersInput
    exposure_type: str = "data"

    def execute(self, **kwargs) -> ToolResult:
        dept = kwargs.pop("dept_no")
        return self._request("GET", f"/api/departments/{dept}/employees", params=kwargs)


class DepartmentManagersTool(EmployeesAPITool):
    name: str = "get_department_managers"
    description: str = (
        "Get a department's manager records (name + tenure). Default: "
        "current manager; temporal params for history."
    )
    args_schema: type[BaseModel] = _DeptHistoryInput
    exposure_type: str = "data"

    def execute(self, **kwargs) -> ToolResult:
        dept = kwargs.pop("dept_no")
        return self._request("GET", f"/api/departments/{dept}/managers", params=kwargs)


# ── Stats tools ──────────────────────────────────────────────────────────────


class SalaryStatsTool(EmployeesAPITool):
    name: str = "get_salary_stats"
    description: str = (
        "Salary statistics (avg/min/max/count) over current salaries, "
        "optionally filtered by department or title, optionally grouped "
        "by department or title."
    )
    args_schema: type[BaseModel] = _StatsInput
    exposure_type: str = "data"

    def execute(self, **kwargs) -> ToolResult:
        return self._request("GET", "/api/stats/salaries", params=kwargs)


class HeadcountTool(EmployeesAPITool):
    name: str = "get_headcount"
    description: str = "Current headcount of every department."
    exposure_type: str = "data"

    def execute(self, **kwargs) -> ToolResult:
        return self._request("GET", "/api/stats/headcount")


class TopEarnersTool(EmployeesAPITool):
    name: str = "get_top_earners"
    description: str = (
        "Top-n earners by current salary, optionally filtered by "
        "department or title. Returns employee, salary, title, department."
    )
    args_schema: type[BaseModel] = _TopEarnersInput
    exposure_type: str = "data"

    def execute(self, **kwargs) -> ToolResult:
        return self._request("GET", "/api/stats/top-earners", params=kwargs)


class SalaryDistributionTool(EmployeesAPITool):
    name: str = "get_salary_distribution"
    description: str = (
        "Histogram of current salaries with min/max/total and equal-width "
        "buckets (bucket counts are inclusive of upper edge on the last "
        "bucket)."
    )
    args_schema: type[BaseModel] = _DistributionInput
    exposure_type: str = "data"

    def execute(self, **kwargs) -> ToolResult:
        return self._request("GET", "/api/stats/salary-distribution", params=kwargs)


# ── Mutation tools ───────────────────────────────────────────────────────────


class HireEmployeeTool(EmployeesAPITool):
    name: str = "hire_employee"
    description: str = (
        "Hire a new employee: creates the employee plus initial salary, "
        "title, and department membership (single transaction). Returns "
        "the new employee's full state including assigned emp_no."
    )
    args_schema: type[BaseModel] = _HireInput
    exposure_type: str = "data"

    def execute(self, **kwargs) -> ToolResult:
        return self._request("POST", "/api/employees", json_body=kwargs)


class SetSalaryTool(EmployeesAPITool):
    name: str = "set_salary"
    description: str = (
        "Set an employee's salary to a new value effective on a date "
        "(default today). Closes the current record and opens a new one — "
        "history is preserved."
    )
    args_schema: type[BaseModel] = _SalaryInput
    exposure_type: str = "data"

    def execute(self, **kwargs) -> ToolResult:
        emp_no = kwargs.pop("emp_no")
        return self._request("PUT", f"/api/employees/{emp_no}/salary", json_body=kwargs)


class SetTitleTool(EmployeesAPITool):
    name: str = "set_title"
    description: str = (
        "Promote/retitle an employee effective on a date (default today). "
        "Title must be a valid reference title. History is preserved."
    )
    args_schema: type[BaseModel] = _TitleInput
    exposure_type: str = "data"

    def execute(self, **kwargs) -> ToolResult:
        emp_no = kwargs.pop("emp_no")
        return self._request("PUT", f"/api/employees/{emp_no}/title", json_body=kwargs)


class TransferEmployeeTool(EmployeesAPITool):
    name: str = "transfer_employee"
    description: str = (
        "Transfer an employee to a different department effective on a "
        "date (default today). History is preserved."
    )
    args_schema: type[BaseModel] = _TransferInput
    exposure_type: str = "data"

    def execute(self, **kwargs) -> ToolResult:
        emp_no = kwargs.pop("emp_no")
        return self._request("POST", f"/api/employees/{emp_no}/transfer", json_body=kwargs)


class AssignManagerTool(EmployeesAPITool):
    name: str = "assign_department_manager"
    description: str = (
        "Make an employee the manager of a department effective on a date "
        "(default today). The employee must currently belong to that "
        "department. Previous manager's record is closed, not destroyed."
    )
    args_schema: type[BaseModel] = _ManagerInput
    exposure_type: str = "data"

    def execute(self, **kwargs) -> ToolResult:
        dept = kwargs.pop("dept_no")
        return self._request("PUT", f"/api/departments/{dept}/manager", json_body=kwargs)


class TerminateEmployeeTool(EmployeesAPITool):
    name: str = "terminate_employee"
    description: str = (
        "Terminate an employee: closes all current records (salary, "
        "title, department, manager). Row history is preserved; the "
        "employee becomes inactive."
    )
    args_schema: type[BaseModel] = _EmpNoInput
    exposure_type: str = "data"

    def execute(self, **kwargs) -> ToolResult:
        return self._request("DELETE", f"/api/employees/{kwargs['emp_no']}")


# ── Evaluation tool ──────────────────────────────────────────────────────────


class SubmitAnswerTool(EmployeesAPITool):
    name: str = "submit_answer"
    description: str = (
        "Submit the final answer. Call this when confident; afterwards "
        "respond with a brief text answer and call no more tools."
    )
    args_schema: type[BaseModel] = _AnswerInput
    exposure_type: str = "evaluation"

    def execute(self, **kwargs) -> ToolResult:
        return ToolResult(
            output="Answer submitted successfully. It will be scored "
            "against the gold answer.",
            data={"submitted_answer": str(kwargs.get("answer", ""))},
        )


# ── Factory ──────────────────────────────────────────────────────────────────


def make_employees_tools(eag_mode: bool = False, base_url: str | None = None) -> list[BaseTool]:
    """All 20 employees-API tools plus the 5 local evaluator tools.

    Args:
        eag_mode: True → data tools emit shape-only observations with the
            raw body kept as a local artifact for placeholder resolution.
        base_url: Override API origin (default env EMPLOYEES_API_URL or
            http://127.0.0.1:8000).
    """
    common = {"eag_mode": eag_mode}
    if base_url:
        common["api_base"] = base_url
    classes = [
        GetReferenceTool,
        SearchEmployeesTool,
        GetEmployeeTool,
        SalaryHistoryTool,
        TitleHistoryTool,
        DepartmentHistoryTool,
        GetDepartmentTool,
        DepartmentMembersTool,
        DepartmentManagersTool,
        SalaryStatsTool,
        HeadcountTool,
        TopEarnersTool,
        SalaryDistributionTool,
        HireEmployeeTool,
        SetSalaryTool,
        SetTitleTool,
        TransferEmployeeTool,
        AssignManagerTool,
        TerminateEmployeeTool,
        SubmitAnswerTool,
    ]
    tools: list[BaseTool] = [cls(**common) for cls in classes]
    # Local compute tier (compare/compute/concat/count_items/filter_items) —
    # same surface in both modes so the ablation varies only observations.
    tools += make_evaluator_tools(eag_mode=eag_mode)
    return tools
