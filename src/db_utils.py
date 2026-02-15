"""
Database utilities for Ovarian Cancer Risk Prediction project.

Provides:
- SQLite connection handling
- SQL query execution
- Model dataset loading
- Table/view validation
"""

import sqlite3
import pandas as pd
from typing import Optional
from contextlib import contextmanager

from .config import DATABASE_PATH


# ======================================================
# Connection Management
# ======================================================

def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """
    Create and return a SQLite connection.

    Parameters
    ----------
    db_path : str, optional
        Path to database file. Defaults to configured DATABASE_PATH.

    Returns
    -------
    sqlite3.Connection
    """
    path = db_path if db_path else DATABASE_PATH
    return sqlite3.connect(path)


@contextmanager
def db_connection(db_path: Optional[str] = None):
    """
    Context manager for safe database connection handling.
    """
    conn = get_connection(db_path)
    try:
        yield conn
    finally:
        conn.close()


# ======================================================
# Query Execution
# ======================================================

def execute_query(query: str, db_path: Optional[str] = None) -> pd.DataFrame:
    """
    Execute SQL query and return results as DataFrame.
    """
    with db_connection(db_path) as conn:
        return pd.read_sql(query, conn)


def execute_script(script_path: str, db_path: Optional[str] = None) -> None:
    """
    Execute full SQL script file.
    """
    with open(script_path, "r") as file:
        sql_script = file.read()

    with db_connection(db_path) as conn:
        conn.executescript(sql_script)
        conn.commit()


# ======================================================
# Table / View Checks
# ======================================================

def table_exists(table_name: str, db_path: Optional[str] = None) -> bool:
    """
    Check whether a table or view exists in the database.
    """
    query = f"""
    SELECT name FROM sqlite_master
    WHERE type IN ('table', 'view')
    AND name = '{table_name}';
    """

    result = execute_query(query, db_path)
    return not result.empty


# ======================================================
# Model Dataset Loader
# ======================================================

def load_model_dataset(db_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load the ML-ready dataset from SQL (model_dataset table).

    Returns
    -------
    pd.DataFrame
    """
    if not table_exists("model_dataset", db_path):
        raise ValueError(
            "Table 'model_dataset' does not exist. "
            "Ensure SQL scripts have been executed in order."
        )

    query = "SELECT * FROM model_dataset;"
    return execute_query(query, db_path)


# ======================================================
# Generic Table Loader
# ======================================================

def load_table(table_name: str, db_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load any table or view from the database.

    Parameters
    ----------
    table_name : str
        Name of table or view.

    Returns
    -------
    pd.DataFrame
    """
    if not table_exists(table_name, db_path):
        raise ValueError(f"Table or view '{table_name}' does not exist.")

    query = f"SELECT * FROM {table_name};"
    return execute_query(query, db_path)


# ======================================================
# Save DataFrame to Database
# ======================================================

def save_dataframe(
    df: pd.DataFrame,
    table_name: str,
    if_exists: str = "replace",
    db_path: Optional[str] = None
) -> None:
    """
    Save pandas DataFrame to database.

    Parameters
    ----------
    df : pd.DataFrame
    table_name : str
    if_exists : str
        'replace', 'append', or 'fail'
    """
    with db_connection(db_path) as conn:
        df.to_sql(table_name, conn, if_exists=if_exists, index=False)
