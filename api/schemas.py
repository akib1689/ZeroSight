"""Pydantic models — request bodies and response shapes."""

from datetime import date
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, Field


class EmployeeCreate(BaseModel):
    first_name: str = Field(min_length=1, max_length=14)
    last_name: str = Field(min_length=1, max_length=14)
    gender: Literal["M", "F"]
    birth_date: date
    hire_date: date
    dept_no: str = Field(min_length=1, max_length=4)
    salary: int = Field(gt=0, le=10_000_000)
    title: str = Field(min_length=1, max_length=50)


class EmployeeUpdate(BaseModel):
    first_name: str | None = Field(None, min_length=1, max_length=14)
    last_name: str | None = Field(None, min_length=1, max_length=14)
    gender: Literal["M", "F"] | None = None
    birth_date: date | None = None


class SalaryUpdate(BaseModel):
    new_salary: int = Field(gt=0, le=10_000_000)
    effective_date: date | None = None


class TitleUpdate(BaseModel):
    new_title: str = Field(min_length=1, max_length=50)
    effective_date: date | None = None


class TransferRequest(BaseModel):
    dept_no: str = Field(min_length=1, max_length=4)
    effective_date: date | None = None


class ManagerAssignRequest(BaseModel):
    emp_no: int
    effective_date: date | None = None


class SalaryRecord(BaseModel):
    salary: int
    from_date: date
    to_date: date


class TitleRecord(BaseModel):
    title: str
    from_date: date
    to_date: date


class MembershipRecord(BaseModel):
    dept_no: str
    dept_name: str
    from_date: date
    to_date: date


class ManagerRecord(BaseModel):
    emp_no: int
    first_name: str
    last_name: str
    from_date: date
    to_date: date


class CurrentTitleRef(BaseModel):
    title: str
    from_date: date
    to_date: date


class CurrentDeptRef(BaseModel):
    dept_no: str
    dept_name: str
    from_date: date
    to_date: date


class EmployeeOut(BaseModel):
    emp_no: int
    first_name: str
    last_name: str
    gender: str
    birth_date: date
    hire_date: date
    active: bool
    salary: int | None = None
    title: CurrentTitleRef | None = None
    department: CurrentDeptRef | None = None


class EmployeeListItem(BaseModel):
    emp_no: int
    first_name: str
    last_name: str
    gender: str
    hire_date: date
    active: bool
    salary: int | None = None
    title: str | None = None
    dept_no: str | None = None
    dept_name: str | None = None


class DeptMemberItem(BaseModel):
    emp_no: int
    first_name: str
    last_name: str
    gender: str
    hire_date: date
    membership_from: date
    membership_to: date


class DepartmentListItem(BaseModel):
    dept_no: str
    dept_name: str
    headcount: int | None = None


class DepartmentDetail(BaseModel):
    dept_no: str
    dept_name: str
    manager: ManagerRecord | None = None
    headcount: int


class SalaryStatsRow(BaseModel):
    avg_salary: float | None = None
    min_salary: int | None = None
    max_salary: int | None = None
    count: int
    dept_no: str | None = None
    dept_name: str | None = None
    title: str | None = None


class HeadcountRow(BaseModel):
    dept_no: str
    dept_name: str
    headcount: int


class TopEarnerRow(BaseModel):
    emp_no: int
    first_name: str
    last_name: str
    salary: int
    title: str
    dept_name: str


class DistributionBucket(BaseModel):
    lower: int
    upper: int
    count: int


class DistributionOut(BaseModel):
    min_salary: int | None = None
    max_salary: int | None = None
    total: int
    buckets: list[DistributionBucket]


T = TypeVar("T")


class Page(BaseModel, Generic[T]):
    items: list[T]
    total: int
    page: int
    page_size: int
