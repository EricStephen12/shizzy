"""
redis_client.py — Shared Redis connection for queue, hash index, and cache.

Three Redis databases:
  db=0  — Celery job queue (default)
  db=1  — Fingerprint hash index  (hash_val → [(song_id, offset), ...])
  db=2  — Result cache + usage counters
"""

import redis
from app.config import REDIS_URL


def _make_url(db: int) -> str:
    """Replace the db number in the Redis URL."""
    base = REDIS_URL.rstrip("/")
    # Strip existing db suffix if present
    parts = base.rsplit("/", 1)
    if len(parts) == 2 and parts[1].isdigit():
        base = parts[0]
    return f"{base}/{db}"


# Queue — used by Celery
queue_redis = redis.from_url(_make_url(0), decode_responses=True)

# Fingerprint hash index — fast in-memory hash lookup
index_redis = redis.from_url(_make_url(1), decode_responses=True)

# Result cache + usage counters
cache_redis = redis.from_url(_make_url(2), decode_responses=True)


def ping_redis() -> bool:
    """Return True if Redis is reachable."""
    try:
        return queue_redis.ping()
    except Exception:
        return False
