"""Reference data endpoints — valid values for dropdowns and validation."""

from fastapi import APIRouter, Depends

from api.db import fetch_all, get_conn

router = APIRouter(prefix="/titles", tags=["reference"])


@router.get("", response_model=list[str])
def list_titles(conn=Depends(get_conn)):
    rows = fetch_all(conn, "SELECT DISTINCT title FROM titles ORDER BY title")
    return [r["title"] for r in rows]
