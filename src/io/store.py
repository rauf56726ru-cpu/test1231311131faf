"""Data storage helpers for parquet and SQLite."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from sqlalchemy import create_engine

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_parquet(df: pd.DataFrame, path: Path | str) -> None:
    path = Path(path)
    ensure_parent(path)
    LOGGER.info("Writing parquet: %s", path.as_posix())
    df.to_parquet(path)


def append_parquet(df: pd.DataFrame, path: Path | str) -> None:
    path = Path(path)
    ensure_parent(path)
    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, df]).drop_duplicates()
    else:
        combined = df
    combined.to_parquet(path)


def append_sqlite(df: pd.DataFrame, table: str, db_url: str) -> None:
    engine = create_engine(db_url)
    with engine.begin() as conn:
        LOGGER.info("Appending %s rows to table=%s", len(df), table)
        df.to_sql(table, conn, if_exists="append", index=False)


def read_sqlite(query: str, db_url: str) -> pd.DataFrame:
    engine = create_engine(db_url)
    with engine.connect() as conn:
        return pd.read_sql(query, conn)
