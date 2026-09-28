"""Database connection and session configuration layer."""

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# We will use SQLite for local deployment, but this completely abstracts
# the dialect allowing us to swap to PostgreSQL cleanly via ENV vars.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./voltguard.db")

engine = create_engine(
    DATABASE_URL,
    # check_same_thread=False is needed only for SQLite in FastAPI.
    # It allows multiple threads to access a single connection pool safely.
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# This is the Base class that all our future ORM models will inherit from
Base = declarative_base()


def get_db():
    """FastAPI Dependency for executing individual secure DB transactions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
