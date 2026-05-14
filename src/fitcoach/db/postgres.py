"""Postgres SessionStore: workout sessions and their per-rep records.

The schema has two tables:
- `sessions`: one row per workout session (exercise, optional user, timestamps).
- `reps`: many rows per session, one per completed rep, FK to sessions.
"""
from __future__ import annotations

from typing import Any

import psycopg


_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          SERIAL PRIMARY KEY,
    started_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at    TIMESTAMPTZ,
    exercise    TEXT NOT NULL,
    user_id     TEXT
);

CREATE TABLE IF NOT EXISTS reps (
    id           SERIAL PRIMARY KEY,
    session_id   INTEGER NOT NULL
                 REFERENCES sessions(id) ON DELETE CASCADE,
    rep_index    INTEGER NOT NULL,
    min_angle    REAL NOT NULL,
    form_score   REAL NOT NULL,
    form_passed  BOOLEAN NOT NULL,
    recorded_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (session_id, rep_index)
);
"""


def ensure_schema(conn: psycopg.Connection) -> None:
    """Create the sessions/reps tables if they don't already exist."""
    with conn.cursor() as cur:
        cur.execute(_SCHEMA)


class SessionStore:
    """Thin SQL wrapper for workout sessions and their reps.

    Doesn't manage the connection itself — caller owns transaction boundaries.
    """

    def __init__(self, conn: psycopg.Connection) -> None:
        self._conn = conn

    def create_session(self, *, exercise: str, user_id: str | None = None) -> int:
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO sessions (exercise, user_id) VALUES (%s, %s) RETURNING id",
                (exercise, user_id),
            )
            row = cur.fetchone()
            assert row is not None
            return int(row[0])

    def end_session(self, session_id: int) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE sessions SET ended_at = NOW() WHERE id = %s",
                (session_id,),
            )

    def get_session(self, session_id: int) -> dict[str, Any]:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT id, started_at, ended_at, exercise, user_id "
                "FROM sessions WHERE id = %s",
                (session_id,),
            )
            row = cur.fetchone()
            if row is None:
                raise LookupError(f"session {session_id} not found")
            return {
                "id": row[0], "started_at": row[1], "ended_at": row[2],
                "exercise": row[3], "user_id": row[4],
            }

    def add_rep(
        self,
        session_id: int,
        *,
        rep_index: int,
        min_angle: float,
        form_score: float,
        form_passed: bool,
    ) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO reps "
                "  (session_id, rep_index, min_angle, form_score, form_passed) "
                "VALUES (%s, %s, %s, %s, %s)",
                (session_id, rep_index, min_angle, form_score, form_passed),
            )

    def list_reps(self, session_id: int) -> list[dict[str, Any]]:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT rep_index, min_angle, form_score, form_passed, recorded_at "
                "FROM reps WHERE session_id = %s ORDER BY rep_index",
                (session_id,),
            )
            return [
                {
                    "rep_index": r[0], "min_angle": r[1],
                    "form_score": r[2], "form_passed": r[3],
                    "recorded_at": r[4],
                }
                for r in cur.fetchall()
            ]
