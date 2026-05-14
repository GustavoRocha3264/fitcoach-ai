"""Integration tests for the Postgres SessionStore.

Requires the Docker Postgres container from docker-compose.yml to be running.
Each test runs inside a transaction that's rolled back, so they don't pollute
the database.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

psycopg = pytest.importorskip("psycopg")

from fitcoach.db.postgres import SessionStore, ensure_schema  # noqa: E402


def _dsn() -> str:
    return (
        f"host={os.environ.get('POSTGRES_HOST', 'localhost')} "
        f"port={os.environ.get('POSTGRES_PORT', '5432')} "
        f"user={os.environ.get('POSTGRES_USER', 'fitcoach')} "
        f"password={os.environ.get('POSTGRES_PASSWORD', 'fitcoach')} "
        f"dbname={os.environ.get('POSTGRES_DB', 'fitcoach')}"
    )


@pytest.fixture
def conn():
    """Yield a connection inside a transaction that always gets rolled back."""
    try:
        c = psycopg.connect(_dsn(), connect_timeout=2)
    except psycopg.OperationalError as exc:
        pytest.skip(f"Postgres not reachable: {exc}")
    c.autocommit = False
    try:
        ensure_schema(c)
        c.commit()
        yield c
    finally:
        c.rollback()
        c.close()


def test_create_session_returns_id(conn) -> None:
    store = SessionStore(conn)
    sid = store.create_session(exercise="squat", user_id="alice")
    assert isinstance(sid, int) and sid > 0


def test_add_rep_persists_fields(conn) -> None:
    store = SessionStore(conn)
    sid = store.create_session(exercise="pushup")
    store.add_rep(sid, rep_index=1, min_angle=85.0, form_score=0.83, form_passed=True)
    store.add_rep(sid, rep_index=2, min_angle=120.0, form_score=0.0, form_passed=False)
    reps = store.list_reps(sid)
    assert [r["rep_index"] for r in reps] == [1, 2]
    assert reps[0]["min_angle"] == pytest.approx(85.0)
    assert reps[0]["form_passed"] is True
    assert reps[1]["form_passed"] is False


def test_end_session_sets_ended_at(conn) -> None:
    store = SessionStore(conn)
    sid = store.create_session(exercise="curl")
    assert store.get_session(sid)["ended_at"] is None
    store.end_session(sid)
    assert store.get_session(sid)["ended_at"] is not None


def test_reps_cascade_delete_with_session(conn) -> None:
    store = SessionStore(conn)
    sid = store.create_session(exercise="squat")
    store.add_rep(sid, rep_index=1, min_angle=80.0, form_score=1.0, form_passed=True)
    with conn.cursor() as cur:
        cur.execute("DELETE FROM sessions WHERE id = %s", (sid,))
    assert store.list_reps(sid) == []


def test_add_rep_unknown_session_raises(conn) -> None:
    store = SessionStore(conn)
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        store.add_rep(999_999, rep_index=1, min_angle=80.0, form_score=1.0, form_passed=True)


def test_rep_index_is_unique_per_session(conn) -> None:
    store = SessionStore(conn)
    sid = store.create_session(exercise="squat")
    store.add_rep(sid, rep_index=1, min_angle=80.0, form_score=1.0, form_passed=True)
    with pytest.raises(psycopg.errors.UniqueViolation):
        store.add_rep(sid, rep_index=1, min_angle=70.0, form_score=1.0, form_passed=True)
