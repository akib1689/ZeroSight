"""Runtime configuration — employees DB connection settings from env."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    db_host: str
    db_port: int
    db_user: str
    db_password: str
    db_name: str


def get_settings() -> Settings:
    return Settings(
        db_host=os.environ.get("EMPLOYEES_DB_HOST", "127.0.0.1"),
        db_port=int(os.environ.get("EMPLOYEES_DB_PORT", "3306")),
        db_user=os.environ.get("EMPLOYEES_DB_USER", "root"),
        db_password=os.environ.get("EMPLOYEES_DB_PASSWORD", "root"),
        db_name=os.environ.get("EMPLOYEES_DB_DATABASE", "employees"),
    )
