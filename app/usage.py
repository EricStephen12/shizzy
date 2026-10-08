"""
usage.py — Free tier usage tracking.

Tracks identification count per user in Redis.
Key:  usage:{user_id}
TTL:  30 days (resets monthly)

Free tier: FREE_TIER_LIMIT identifications before paywall.
"""

from app.redis_client import cache_redis, ping_redis
from app.config import FREE_TIER_LIMIT

_MONTHLY_TTL = 30 * 24 * 60 * 60  # 30 days in seconds


def get_usage(user_id: str) -> int:
    """Return how many identifications this user has made this month."""
    if not ping_redis():
        return 0  # Redis down — fail open, allow request
    try:
        val = cache_redis.get(f"usage:{user_id}")
        return int(val) if val else 0
    except Exception:
        return 0


def increment_usage(user_id: str) -> int:
    """Increment usage counter. Returns new count."""
    if not ping_redis():
        return 0
    try:
        key = f"usage:{user_id}"
        pipe = cache_redis.pipeline()
        pipe.incr(key)
        pipe.expire(key, _MONTHLY_TTL)
        results = pipe.execute()
        return int(results[0])
    except Exception:
        return 0


def is_within_free_tier(user_id: str) -> bool:
    """Return True if user still has free identifications remaining."""
    return get_usage(user_id) < FREE_TIER_LIMIT


def get_remaining_free(user_id: str) -> int:
    """Return how many free identifications remain this month."""
    used = get_usage(user_id)
    return max(0, FREE_TIER_LIMIT - used)
