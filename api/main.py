"""Employees API — FastAPI app over the employees MySQL database."""

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.db import fetch_one, get_conn
from api.routes import departments, employees, reference, stats

app = FastAPI(
    title="Employees API",
    version="0.1.0",
    description="REST API over the MySQL 'employees' sample database (datacharmer/test_db).",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(employees.router, prefix="/api")
app.include_router(departments.router, prefix="/api")
app.include_router(reference.router, prefix="/api")
app.include_router(stats.router, prefix="/api")


@app.get("/api/health", tags=["ops"])
def health(conn=Depends(get_conn)):
    fetch_one(conn, "SELECT 1 AS ok")
    return {"status": "ok"}
