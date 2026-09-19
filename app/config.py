import os as _os

# ---------------------------------------------------------------------------
# PostgreSQL connection  (env vars override hardcoded values — used by CI)
# ---------------------------------------------------------------------------
DB_HOST     = _os.environ.get("DB_HOST",     "tokaido.proxy.rlwy.net")
DB_PORT     = int(_os.environ.get("DB_PORT", "51701"))
DB_NAME     = _os.environ.get("DB_NAME",     "railway")
DB_USER     = _os.environ.get("DB_USER",     "postgres")
DB_PASSWORD = _os.environ.get("DB_PASSWORD", "jpJmNBeVMWJieVJjUIMCcxKvnWBzlduS")

DATABASE_URL = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# ---------------------------------------------------------------------------
# Song library directory (relative to project root or absolute)
# ---------------------------------------------------------------------------
SONG_LIBRARY_DIR = "song_library"

# ---------------------------------------------------------------------------
# Cloudflare R2 (audio file storage) — READ-ONLY token
# ---------------------------------------------------------------------------
R2_ACCOUNT_ID    = _os.environ.get("R2_ACCOUNT_ID",    "b2e5411830e116cf4ce6e91e90843db0")
R2_ACCESS_KEY_ID = _os.environ.get("R2_ACCESS_KEY_ID", "57155c88bb8b1905a634a66094da019e")
R2_SECRET_KEY    = _os.environ.get("R2_SECRET_KEY",    "e72f63ce524463796e1afaf6583d42a92d1635cea69807f5ff9380a1ec64269f")
R2_BUCKET_NAME   = _os.environ.get("R2_BUCKET_NAME",   "rehearsalhub-media")
R2_PUBLIC_URL    = _os.environ.get("R2_PUBLIC_URL",    "https://pub-cb7697578fcc48d3b3aeb70a47eb2f65.r2.dev")
# Endpoint is always:  https://<account_id>.r2.cloudflarestorage.com
R2_ENDPOINT_URL  = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"

# ---------------------------------------------------------------------------
# Custom fingerprinter settings
# ---------------------------------------------------------------------------
# Number of spectrogram peaks kept per second (higher = more precise, slower ingest)
FP_PEAKS_PER_SEC = 10
# Frequency band for fingerprinting (Hz)
FP_FREQ_MIN = 300
FP_FREQ_MAX = 3000
# Minimum Jaccard-style match ratio to consider a fingerprint hit valid
FP_CONFIDENCE_THRESHOLD = 0.05   # 5 % of hashes must align

# ---------------------------------------------------------------------------
# Melody / pYIN settings
# ---------------------------------------------------------------------------
PYIN_FMIN = "C2"   # librosa note string → ~65 Hz
PYIN_FMAX = "C7"   # librosa note string → ~2093 Hz
# Maximum DTW distance (normalised per frame) before melody match is discarded
MELODY_DTW_MAX_DISTANCE = 500
