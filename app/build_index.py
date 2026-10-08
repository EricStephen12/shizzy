"""
build_index.py — Populate Redis hash index from existing Postgres/SQLite data.

Run this once after setting up Redis to backfill all existing songs:
    python -m app.build_index

It is safe to re-run — already-indexed songs are skipped.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db import init_db, SessionLocal, Song, SongFingerprint
from app.redis_client import ping_redis
from app import hash_index


def build_index() -> None:
    if not ping_redis():
        print("[build_index] ERROR: Redis is not reachable. Start Redis first.")
        print("[build_index] Windows: download Redis from https://github.com/tporadowski/redis/releases")
        print("[build_index]          or use: docker run -d -p 6379:6379 redis:alpine")
        return

    init_db()
    db = SessionLocal()

    try:
        songs = db.query(Song).all()
        print(f"[build_index] Found {len(songs)} songs to index\n")

        total_hashes = 0
        for song in songs:
            if hash_index.song_exists_in_index(song.id):
                print(f"  → Song {song.id} already indexed — skipping")
                continue

            # Load all hashes for this song from DB
            rows = (
                db.query(SongFingerprint.hash_val, SongFingerprint.offset)
                  .filter(SongFingerprint.song_id == song.id)
                  .all()
            )

            if not rows:
                print(f"  → Song {song.id} has no fingerprints — skipping")
                continue

            hashes = [(row.hash_val, row.offset) for row in rows]
            stored = hash_index.store_hashes(song.id, hashes)
            hash_index.mark_song_indexed(song.id)
            total_hashes += stored
            print(f"  → Song {song.id} '{song.title[:40]}': {stored} hashes indexed")

        stats = hash_index.get_index_stats()
        print(f"\n[build_index] Done. {total_hashes} hashes added.")
        print(f"[build_index] Redis index stats: {stats}")

    finally:
        db.close()


if __name__ == "__main__":
    build_index()
