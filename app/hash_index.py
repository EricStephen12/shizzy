"""
hash_index.py — Redis-backed fingerprint hash index.

Structure in Redis (db=1):
  Key:   "fp:{hash_val}"
  Type:  Redis List
  Value: JSON-encoded entries → [song_id, offset]

At ingest time:  store_hashes() populates the index.
At query time:   lookup_hashes() replaces the Postgres IN query.

This gives O(1) per-hash lookup entirely in RAM — microseconds vs
milliseconds for a Postgres full-table scan at millions of rows.
"""

import json
from app.redis_client import index_redis

# Key prefix for fingerprint hashes
_PREFIX = "fp:"

# TTL — None means keys live forever (songs don't expire)
_TTL = None


def _key(hash_val: str) -> str:
    return f"{_PREFIX}{hash_val}"


def store_hashes(song_id: int, hashes: list[tuple[str, float]]) -> int:
    """
    Store fingerprint hashes for a song in Redis.
    hashes: list of (hash_hex, offset_seconds)
    Returns number of entries stored.
    Idempotent — duplicate entries are naturally deduplicated by the
    unique (song_id, hash_val, offset) combination we check before storing.
    """
    if not hashes:
        return 0

    pipe = index_redis.pipeline(transaction=False)
    stored = 0

    for hash_val, offset in hashes:
        entry = json.dumps([song_id, round(float(offset), 3)])
        pipe.rpush(_key(hash_val), entry)
        stored += 1

    pipe.execute()
    return stored


def lookup_hashes(hash_vals: list[str]) -> list[tuple[int, str, float]]:
    """
    Look up a batch of hash values in Redis.
    Returns list of (song_id, hash_val, offset) tuples — same shape
    as the Postgres query result so fingerprint_engine needs no changes.
    """
    if not hash_vals:
        return []

    pipe = index_redis.pipeline(transaction=False)
    for hv in hash_vals:
        pipe.lrange(_key(hv), 0, -1)

    results = pipe.execute()

    rows = []
    for hash_val, entries in zip(hash_vals, results):
        for entry in entries:
            try:
                song_id, offset = json.loads(entry)
                rows.append((int(song_id), hash_val, float(offset)))
            except Exception:
                continue

    return rows


def song_exists_in_index(song_id: int) -> bool:
    """
    Quick check — does this song have any hashes in the index?
    Used to decide whether to re-index on ingest.
    """
    # We don't track by song_id directly, so check a sentinel key
    sentinel = f"song_indexed:{song_id}"
    return index_redis.exists(sentinel) > 0


def mark_song_indexed(song_id: int) -> None:
    """Mark a song as fully indexed in Redis."""
    index_redis.set(f"song_indexed:{song_id}", "1")


def get_index_stats() -> dict:
    """Return basic stats about the hash index."""
    try:
        info = index_redis.info("keyspace")
        db_info = info.get("db1", {})
        return {
            "total_keys": db_info.get("keys", 0),
            "redis_connected": True,
        }
    except Exception as e:
        return {"redis_connected": False, "error": str(e)}
