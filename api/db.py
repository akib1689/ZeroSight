"""Database access — per-request pymysql connections and fetch helpers."""

from collections.abc import Generator

import pymysql
import pymysql.connections
from fastapi import Depends

from api.config import Settings, get_settings


def get_conn(
    settings: Settings = Depends(get_settings),
) -> Generator[pymysql.connections.Connection, None, None]:
    conn = pymysql.connect(
        host=settings.db_host,
        port=settings.db_port,
        user=settings.db_user,
        password=settings.db_password,
        database=settings.db_name,
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=False,
        charset="utf8mb4",
    )
    try:
        yield conn
    finally:
        try:
            conn.rollback()
        except Exception:
            pass
        conn.close()


def fetch_all(conn: pymysql.connections.Connection, sql: str, params=None) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(sql, params or ())
        return list(cur.fetchall())


def fetch_one(conn: pymysql.connections.Connection, sql: str, params=None) -> dict | None:
    rows = fetch_all(conn, sql, params)
    return rows[0] if rows else None


def execute(conn: pymysql.connections.Connection, sql: str, params=None) -> int:
    with conn.cursor() as cur:
        cur.execute(sql, params or ())
        return cur.rowcount
