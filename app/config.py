import os as _os

def _get_env(key: str, default: str) -> str:
    val = _os.environ.get(key, "").strip()
    return val if val else default

# ---------------------------------------------------------------------------
# Database — SQLite for local dev, PostgreSQL for production
# Set DATABASE_URL env var to a postgres:// URL to use PostgreSQL instead.
# ---------------------------------------------------------------------------
_pg_url = _os.environ.get("DATABASE_URL", "").strip()

if _pg_url:
    # Railway provides postgresql:// — psycopg2 needs postgresql+psycopg2://
    if _pg_url.startswith("postgresql://") and "+psycopg2" not in _pg_url:
        _pg_url = _pg_url.replace("postgresql://", "postgresql+psycopg2://", 1)
    elif _pg_url.startswith("postgres://"):
        _pg_url = _pg_url.replace("postgres://", "postgresql+psycopg2://", 1)
    DATABASE_URL = _pg_url
else:
    _db_path = _os.path.join(
        _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
        "song_id.db"
    )
    DATABASE_URL = f"sqlite:///{_db_path}"

# ---------------------------------------------------------------------------
# Song library directory (relative to project root or absolute)
# ---------------------------------------------------------------------------
SONG_LIBRARY_DIR = "song_library"

# ---------------------------------------------------------------------------
# Cloudflare R2 (audio file storage) — READ-ONLY token
# ---------------------------------------------------------------------------
R2_ACCOUNT_ID    = _get_env("R2_ACCOUNT_ID",    "b2e5411830e116cf4ce6e91e90843db0")
R2_ACCESS_KEY_ID = _get_env("R2_ACCESS_KEY_ID", "57155c88bb8b1905a634a66094da019e")
R2_SECRET_KEY    = _get_env("R2_SECRET_KEY",    "e72f63ce524463796e1afaf6583d42a92d1635cea69807f5ff9380a1ec64269f")
R2_BUCKET_NAME   = _get_env("R2_BUCKET_NAME",   "rehearsalhub-media")
R2_PUBLIC_URL    = _get_env("R2_PUBLIC_URL",    "https://pub-cb7697578fcc48d3b3aeb70a47eb2f65.r2.dev")
# Endpoint is always:  https://<account_id>.r2.cloudflarestorage.com
R2_ENDPOINT_URL  = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"

# ---------------------------------------------------------------------------
# Redis — queue + fingerprint hash index + result cache
# ---------------------------------------------------------------------------
REDIS_URL = _get_env("REDIS_URL", "redis://localhost:6379/0")

# Free tier limit before paywall
FREE_TIER_LIMIT = 3
# Number of spectrogram peaks kept per second (higher = more precise, slower ingest)
FP_PEAKS_PER_SEC = 10
# Frequency band for fingerprinting (Hz)
FP_FREQ_MIN = 300
FP_FREQ_MAX = 3000
# Minimum Jaccard-style match ratio to consider a fingerprint hit valid
FP_CONFIDENCE_THRESHOLD = 0.03   # 3% of hashes must align

# ---------------------------------------------------------------------------
# Melody / pYIN settings
# ---------------------------------------------------------------------------
PYIN_FMIN = "C2"   # librosa note string → ~65 Hz
PYIN_FMAX = "C7"   # librosa note string → ~2093 Hz
# Maximum DTW distance (normalised per frame) before melody match is discarded
MELODY_DTW_MAX_DISTANCE = 150
