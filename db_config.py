"""Tychain — DB connection factory (Turso libSQL + local SQLite fallback).

Usage from app.py:

    from db_config import get_connection
    conn = get_connection()         # Row-factory enabled, FK on, WAL when local

The connection object is fully sqlite3.Connection-compatible:
- conn.execute / executemany / executescript
- cursor() with row.keys() / row['col'] access
- conn.commit() / conn.rollback() / conn.close()
- works inside `with conn:` blocks

Selection logic
---------------
1. If TURSO_DATABASE_URL is set in the environment, connect to that remote
   libSQL database via the `libsql-experimental` driver, authenticated with
   TURSO_AUTH_TOKEN. This is the production path used on Hugging Face Spaces.
2. Otherwise fall back to local SQLite at TYCHAIN_DB_PATH (or ./tychain.db).
   This keeps `python app.py` working for local development with no network.

Required environment variables (production)
-------------------------------------------
- TURSO_DATABASE_URL   e.g. libsql://tychain-1-<your-org>.turso.io
- TURSO_AUTH_TOKEN     Turso JWT (rotate via `turso db tokens create <db>`)

Optional
--------
- TYCHAIN_DB_PATH      Override local SQLite file location (default ./tychain.db)
- TURSO_SYNC_URL       If set, use embedded-replica mode (writes go to remote,
                       reads served from a local replica file at TYCHAIN_DB_PATH)

Never commit credentials. See .env.example.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Any

# Defer the libsql import so dev environments without the wheel still work.
_libsql = None
_libsql_import_error: str | None = None


def _try_import_libsql():
    """Import libsql-experimental once and cache the module/error."""
    global _libsql, _libsql_import_error
    if _libsql is not None or _libsql_import_error is not None:
        return _libsql
    try:
        import libsql_experimental as libsql  # type: ignore
        _libsql = libsql
    except ImportError as e:
        _libsql_import_error = str(e)
    return _libsql


# ── Public API ─────────────────────────────────────────────────────────────────

def using_turso() -> bool:
    """True when Turso env vars are present (production path)."""
    return bool(os.environ.get("TURSO_DATABASE_URL"))


def dict_row(cursor):
    """Fetch ONE row from `cursor` and return it as a plain dict (or None).

    Works against both sqlite3.Row objects (which support row['col']) and
    plain tuples returned by libsql_experimental — the latter use the
    cursor's `description` to recover column names.
    """
    row = cursor.fetchone()
    if row is None:
        return None
    if hasattr(row, "keys"):
        return {k: row[k] for k in row.keys()}
    cols = [d[0] for d in cursor.description] if cursor.description else []
    return dict(zip(cols, row))


def dict_rows(cursor):
    """Fetch ALL rows from `cursor` and return as a list of plain dicts.

    Same compatibility logic as `dict_row`. Always returns a list (possibly empty).
    """
    rows = cursor.fetchall()
    if not rows:
        return []
    if hasattr(rows[0], "keys"):
        return [{k: r[k] for k in r.keys()} for r in rows]
    cols = [d[0] for d in cursor.description] if cursor.description else []
    return [dict(zip(cols, r)) for r in rows]


def get_connection() -> Any:
    """Return a DB connection — Turso libSQL when configured, else local SQLite.

    The returned object is sqlite3-API compatible. Caller is responsible for
    `commit()` / `close()` (or use a `with` block).
    """
    if using_turso():
        return _connect_turso()
    return _connect_sqlite()

def get_users_connection() -> Any:
    """Return a DB connection for the users *section*.

    The user-facing requirement is "one Turso DB with two sections":
    - users + login_attempts on one side (auth)
    - market_data + signals + forum_* on the other side (app data)

    By default we therefore route the users section to the *same* Turso DB
    as `get_connection()`. Set TURSO_USERS_DATABASE_URL only if you want
    auth in a physically separate database (legacy split-DB deployments).
    """
    if os.environ.get("TURSO_USERS_DATABASE_URL"):
        return _connect_turso_users()
    if using_turso():
        # Single-DB mode: users live in the same Turso database as app data.
        return _connect_turso()
    return _connect_sqlite_users()


# ── Local SQLite ───────────────────────────────────────────────────────────────

def _local_db_path() -> str:
    """Resolve the local sqlite path (used in dev or as embedded-replica file)."""
    return os.environ.get(
        "TYCHAIN_DB_PATH",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "tychain.db"),
    )

def _local_users_db_path() -> str:
    return os.environ.get(
        "TYCHAIN_USERS_DB_PATH",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "tychain_users.db"),
    )


def _connect_sqlite() -> sqlite3.Connection:
    conn = sqlite3.connect(_local_db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def _connect_sqlite_users() -> sqlite3.Connection:
    conn = sqlite3.connect(_local_users_db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# ── Turso (libSQL) ─────────────────────────────────────────────────────────────

def _connect_turso():
    """Open a connection to Turso. Two modes:

    * Remote-only (default):     libsql.connect(database=URL, auth_token=TOKEN)
    * Embedded replica (opt-in): libsql.connect(database=LOCAL_FILE,
                                                sync_url=URL, auth_token=TOKEN)

    Embedded replica is preferred for read-heavy workloads (lower latency)
    but requires writable local storage. Toggle with TURSO_SYNC_URL.
    """
    libsql = _try_import_libsql()
    if libsql is None:
        raise RuntimeError(
            "TURSO_DATABASE_URL is set but the `libsql-experimental` package "
            "is not installed. Add it to requirements.txt or pip install it. "
            f"(Import error: {_libsql_import_error})"
        )

    url = os.environ["TURSO_DATABASE_URL"]
    token = os.environ.get("TURSO_AUTH_TOKEN", "")
    if not token:
        raise RuntimeError(
            "TURSO_DATABASE_URL is set but TURSO_AUTH_TOKEN is missing. "
            "Generate one with: turso db tokens create <database>"
        )

    sync_url = os.environ.get("TURSO_SYNC_URL")
    if sync_url:
        # Embedded replica: local file kept in sync with remote
        conn = libsql.connect(
            database=_local_db_path(),
            sync_url=sync_url,
            auth_token=token,
        )
        try:
            conn.sync()  # initial pull
        except Exception as e:
            # Non-fatal — the replica may already be up-to-date or remote unreachable
            print(f"[db_config] libsql sync warning: {e}")
    else:
        # Remote-only
        conn = libsql.connect(database=url, auth_token=token)

    # libsql_experimental returns sqlite3-style connections; honour our conventions.
    try:
        conn.row_factory = _row_factory  # type: ignore[attr-defined]
    except Exception:
        pass  # some versions expose Row natively
    try:
        conn.execute("PRAGMA foreign_keys = ON")
    except Exception:
        pass
    return conn

def _connect_turso_users():
    libsql = _try_import_libsql()
    if libsql is None:
        raise RuntimeError(
            "TURSO_DATABASE_URL or TURSO_USERS_DATABASE_URL is set but `libsql-experimental` "
            "is not installed."
        )

    url = os.environ.get("TURSO_USERS_DATABASE_URL", "libsql://tychain-users-ibrahimaydinn.turso.io")
    token = os.environ.get("TURSO_USERS_AUTH_TOKEN", os.environ.get("TURSO_AUTH_TOKEN", ""))
    
    if not token:
        raise RuntimeError("Missing TURSO_USERS_AUTH_TOKEN or TURSO_AUTH_TOKEN for users database.")

    conn = libsql.connect(database=url, auth_token=token)
    try:
        conn.row_factory = _row_factory
    except Exception:
        pass
    try:
        conn.execute("PRAGMA foreign_keys = ON")
    except Exception:
        pass
    return conn


def _row_factory(cursor, row):
    """sqlite3.Row-style mapping: row['col'] and row.keys()."""
    cols = [c[0] for c in cursor.description]
    return _Row(cols, row)


class _Row(tuple):
    """Tuple subclass that supports row['col'] and row.keys()."""
    __slots__ = ()
    _cols: tuple

    def __new__(cls, cols, values):
        inst = super().__new__(cls, values)
        inst._cols = tuple(cols)  # type: ignore[attr-defined]
        return inst

    def keys(self):
        return list(self._cols)

    def __getitem__(self, key):
        if isinstance(key, str):
            try:
                return tuple.__getitem__(self, self._cols.index(key))
            except ValueError as e:
                raise KeyError(key) from e
        return tuple.__getitem__(self, key)


# ── Diagnostics (run `python db_config.py` to verify) ──────────────────────────

if __name__ == "__main__":
    print(f"using_turso = {using_turso()}")
    if using_turso():
        print(f"TURSO_DATABASE_URL = {os.environ.get('TURSO_DATABASE_URL')}")
        print(f"TURSO_AUTH_TOKEN   = {'<set>' if os.environ.get('TURSO_AUTH_TOKEN') else '<MISSING>'}")
        print(f"TURSO_SYNC_URL     = {os.environ.get('TURSO_SYNC_URL', '(unset — remote-only)')}")
    else:
        print(f"Local SQLite path  = {_local_db_path()}")

    try:
        conn = get_connection()
        rows = list(conn.execute("SELECT 1 AS ok").fetchall())
        print(f"Test query result  = {rows}")
        conn.close()
    except Exception as e:
        print(f"Connection failed: {type(e).__name__}: {e}")
