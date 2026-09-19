"""
ingest.py — Process your R2 audio library into the fingerprint + melody DB.

Modes:
  R2 mode (default): reads audio files directly from your Cloudflare R2 bucket.
  Local mode (--local): reads from song_library/ folder and also uploads to R2.

Usage:
    # From song-id-app/ with venv active:
    python -m app.ingest            # pull from R2 bucket
    python -m app.ingest --local    # process local files in song_library/

The script is FULLY IDEMPOTENT — re-running skips already-ingested songs.
R2 files are NEVER deleted, overwritten, or modified.
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import SONG_LIBRARY_DIR
from app.db import init_db, SessionLocal, Song
from app import fingerprint_engine, melody_engine
from app import r2_storage

LOCAL_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}


def _title_from_key(key: str) -> str:
    """Best-effort title from an R2 key or local filename."""
    basename = os.path.splitext(os.path.basename(key))[0]
    return basename.replace("_", " ").replace("-", " ").strip()


def _ingest_one(song_id: int, audio_path: str, db) -> None:
    """Run fingerprint + melody engines on a local audio file path."""
    # Fingerprint
    try:
        n = fingerprint_engine.store_fingerprints(song_id, audio_path, db)
        print(f"     [fingerprint]  {n} hashes stored")
    except Exception as exc:
        print(f"     [fingerprint]  ERROR: {exc}")

    # Melody contour
    try:
        ok = melody_engine.store_contour(song_id, audio_path, db)
        print("     [melody]       contour stored" if ok
              else "     [melody]       contour too short — skipped")
    except Exception as exc:
        print(f"     [melody]       ERROR: {exc}")


def ingest_from_r2(prefix: str = "", limit: int = 0) -> None:
    """
    Download each audio file from R2 to a temp location, process it,
    then delete the temp copy. R2 bucket is NEVER modified.

    Args:
        prefix: Only process keys that start with this string (e.g. 'zones/zone-001/').
        limit:  Stop after this many NEW songs are ingested (0 = no limit).
    """
    print(f"[ingest] Mode: R2 bucket → '{r2_storage.R2_BUCKET_NAME}'")
    if limit:
        print(f"[ingest] Limit: {limit} new songs")
    print("[ingest] Initialising database tables …")
    init_db()

    print("[ingest] Listing audio files in R2 …")
    keys = r2_storage.list_audio_keys(prefix=prefix)

    if not keys:
        print("[ingest] No audio files found in bucket. Check R2_BUCKET_NAME in config.py.")
        return

    print(f"[ingest] Found {len(keys)} audio file(s).\n")
    db = SessionLocal()
    ingested = 0
    try:
        for key in sorted(keys):
            if limit and ingested >= limit:
                print(f"[ingest] Reached limit of {limit} songs — stopping.")
                break

            print(f"  → {key}")
            r2_url = r2_storage.get_public_url(key)

            # Skip if already ingested (idempotent via r2_key)
            existing = db.query(Song).filter(Song.r2_key == key).first()
            if existing:
                print("     [songs]       already ingested — skipping")
                print()
                continue

            # Register song
            title = _title_from_key(key)
            song  = Song(title=title, file_path=key, r2_key=key, r2_url=r2_url)
            db.add(song)
            db.commit()
            db.refresh(song)
            print(f"     [songs]       registered as '{song.title}' (id={song.id})")

            # Download temp copy → process → delete
            tmp_path = None
            try:
                print("     [r2]          downloading temp copy …")
                tmp_path = r2_storage.download_temp(key)
                _ingest_one(song.id, tmp_path, db)
                ingested += 1
            except Exception as exc:
                print(f"     [r2]          ERROR downloading: {exc}")
            finally:
                if tmp_path:
                    r2_storage.cleanup_temp(tmp_path)
                    print("     [r2]          temp copy deleted")
            print()
    finally:
        db.close()

    print(f"[ingest] Done. {ingested} new songs indexed.")


def ingest_from_local(library_dir: str = SONG_LIBRARY_DIR) -> None:
    """
    Process audio files from the local song_library/ folder.
    Stores the local file path in the DB (r2_key/r2_url will be None).
    """
    print(f"[ingest] Mode: local → '{os.path.abspath(library_dir)}'")
    print("[ingest] Initialising database tables …")
    init_db()

    audio_files = []
    for root, _, files in os.walk(library_dir):
        for fname in files:
            if os.path.splitext(fname)[1].lower() in LOCAL_EXTENSIONS:
                audio_files.append(os.path.join(root, fname))

    if not audio_files:
        print("[ingest] No audio files found. Drop mp3/wav files into song_library/")
        return

    print(f"[ingest] Found {len(audio_files)} audio file(s).\n")
    db = SessionLocal()
    try:
        for audio_path in sorted(audio_files):
            print(f"  → {os.path.basename(audio_path)}")
            existing = db.query(Song).filter(Song.file_path == audio_path).first()
            if existing:
                print("     [songs]       already registered — skipping")
                print()
                continue

            title = _title_from_key(audio_path)
            song  = Song(title=title, file_path=audio_path)
            db.add(song)
            db.commit()
            db.refresh(song)
            print(f"     [songs]       registered as '{song.title}' (id={song.id})")
            _ingest_one(song.id, audio_path, db)
            print()
    finally:
        db.close()

    print("[ingest] Done.")


def _parse_limit(val):
    try:
        return int(val)
    except (ValueError, TypeError):
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Song ID — ingest audio library")
    parser.add_argument("--local", action="store_true",
                        help="Process local song_library/ instead of R2 bucket")
    parser.add_argument("--prefix", default="",
                        help="R2 key prefix/folder to filter (e.g. 'zones/zone-001/')")
    parser.add_argument("--limit", type=_parse_limit, default=0,
                        help="Stop after N new songs are ingested (0 = no limit)")
    args = parser.parse_args()

    if args.local:
        ingest_from_local()
    else:
        ingest_from_r2(prefix=args.prefix, limit=args.limit)
