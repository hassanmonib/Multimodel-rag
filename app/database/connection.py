"""
SQLAlchemy engine, session factory, and database initialisation.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings
from app.models.db_models import Base

logger = logging.getLogger(__name__)

_engine: Engine | None = None
_SessionFactory: sessionmaker | None = None


def _get_engine() -> Engine:
    """Create (or return cached) SQLAlchemy engine."""
    global _engine
    if _engine is None:
        settings = get_settings()
        connect_args: dict = {}

        # SQLite: enable WAL mode and foreign key enforcement
        if settings.database_url.startswith("sqlite"):
            connect_args["check_same_thread"] = False

            @event.listens_for(Engine, "connect")
            def _sqlite_pragmas(dbapi_conn, _conn_record):  # type: ignore[type-arg]
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

        _engine = create_engine(
            settings.database_url,
            connect_args=connect_args,
            pool_pre_ping=True,
            echo=False,
        )
        logger.info("Database engine created: %s", settings.database_url)
    return _engine


def _get_session_factory() -> sessionmaker:
    """Return a cached session factory."""
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(
            bind=_get_engine(),
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )
    return _SessionFactory


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """
    Context manager that yields a database session and handles
    commit / rollback / close automatically.

    Usage::

        with get_session() as session:
            session.add(some_object)
    """
    factory = _get_session_factory()
    session: Session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> None:
    """
    Create all tables defined in ORM models (idempotent).
    Call once at application startup.
    """
    engine = _get_engine()
    Base.metadata.create_all(engine)

    # Verify connectivity
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    logger.info("Database initialised and all tables created.")
