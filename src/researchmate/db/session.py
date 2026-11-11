"""Engine, sessions and schema creation.

A *session* is SQLAlchemy's unit of work: changes are collected in it and
written together on ``commit()``, or discarded on ``rollback()``.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from researchmate.config import Settings
from researchmate.db.models import Base


def create_db_engine(settings: Settings) -> Engine:
    """Create the engine for ``settings.resolved_database_url``.

    SQLite-specific setup (only applied when the URL is SQLite):

    * ``check_same_thread=False``: FastAPI runs sync routes in a thread pool,
      so a connection may be used by a thread other than the one that
      created it. Safe because each request gets its own session.
    * ``PRAGMA foreign_keys=ON`` on every connection: SQLite ignores foreign
      keys (and ``ON DELETE CASCADE``) unless this is enabled per connection.
    """
    url = settings.resolved_database_url
    is_sqlite = url.startswith("sqlite")
    connect_args: dict[str, Any] = {"check_same_thread": False} if is_sqlite else {}
    engine = create_engine(url, connect_args=connect_args)

    if is_sqlite:
        @event.listens_for(engine, "connect")
        def _enable_foreign_keys(dbapi_connection: Any, _record: Any) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


def init_db(engine: Engine) -> None:
    """Create missing tables. Idempotent: existing tables are left untouched.

    This is enough while the schema is new. Once real data must survive
    schema changes, a migration tool (Alembic) replaces it.
    """
    Base.metadata.create_all(engine)


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    # expire_on_commit=False: objects stay readable after commit (e.g. to
    # build an API response) without another round-trip to the database.
    return sessionmaker(bind=engine, expire_on_commit=False)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    """Commit if the block succeeds, roll back if it raises, always close."""
    session = factory()
    try:
        yield session
        session.commit()
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()
