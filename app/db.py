"""
db.py — SQLAlchemy models and database initialisation.

Tables:
  songs             — one row per audio file in the library
  song_fingerprints — spectrogram peak hashes produced by fingerprint_engine
  melody_contours   — pYIN pitch-direction arrays produced by melody_engine
"""

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    JSON,
    ForeignKey,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from app.config import DATABASE_URL

# ---------------------------------------------------------------------------
# Engine + session factory
# ---------------------------------------------------------------------------
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Song(Base):
    """One row per audio file that has been ingested."""
    __tablename__ = "songs"

    id        = Column(Integer, primary_key=True, index=True)
    title     = Column(String, nullable=False)
    artist    = Column(String, nullable=True)
    file_path = Column(String, nullable=True)          # local path (may be None if R2-only)
    r2_key    = Column(String, nullable=True, unique=True)  # R2 object key
    r2_url    = Column(String, nullable=True)          # public streaming URL

    fingerprints = relationship("SongFingerprint", back_populates="song",
                                cascade="all, delete-orphan")
    melody       = relationship("MelodyContour",   back_populates="song",
                                uselist=False,
                                cascade="all, delete-orphan")


class SongFingerprint(Base):
    """
    Each row is ONE hash pair:  (freq_anchor, freq_point, delta_time) → song.
    Many thousands of rows per song.
    """
    __tablename__ = "song_fingerprints"

    id       = Column(Integer, primary_key=True, index=True)
    song_id  = Column(Integer, ForeignKey("songs.id"), nullable=False, index=True)
    hash_val = Column(String,  nullable=False, index=True)   # hex string
    offset   = Column(Float,   nullable=False)                # seconds from start

    __table_args__ = (
        UniqueConstraint("song_id", "hash_val", "offset", name="uq_fp_entry"),
    )

    song = relationship("Song", back_populates="fingerprints")


class MelodyContour(Base):
    """One row per song: the pitch-direction array [-1, 0, 1, ...]."""
    __tablename__ = "melody_contours"

    id      = Column(Integer, primary_key=True, index=True)
    song_id = Column(Integer, ForeignKey("songs.id"), nullable=False, unique=True)
    contour = Column(JSON, nullable=False)   # list[int]

    song = relationship("Song", back_populates="melody")


# ---------------------------------------------------------------------------
# Helper: create all tables (called once at startup / ingest)
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create all tables if they do not already exist."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency — yields a DB session, closes it when done."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
