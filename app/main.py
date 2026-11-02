"""
main.py — FastAPI application for the Song ID service.

Endpoints:
    POST /identify
        Accepts audio file, queues job, returns job_id immediately.
        If Redis/Celery unavailable — falls back to synchronous processing.

    GET  /result/{job_id}
        Poll for identification result. Returns status: pending | done | failed.

    GET  /health
        Service health check including Redis status.

    GET  /songs
        List all songs in the library.
"""

import os
import shutil
import tempfile
import uuid
import json
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Depends, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import hashlib
import secrets
from pydantic import BaseModel

from app.db import init_db, get_db, Song, User
from app import fingerprint_engine, melody_engine
from app.redis_client import ping_redis, cache_redis
from app.usage import get_usage, increment_usage, get_remaining_free, is_within_free_tier
from app.config import FREE_TIER_LIMIT, FP_CONFIDENCE_THRESHOLD

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Song Sort API",
    description="Loveworld Song Sort — identify songs by audio fingerprint or melody.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.dashboard import router as dashboard_router
app.include_router(dashboard_router)


from app.db import init_db, get_db, Song, SessionLocal
import numpy as np


@app.on_event("startup")
def on_startup():
    init_db()
    # Pre-load hot in-memory search index for sub-50ms search latency
    db = SessionLocal()
    try:
        fp_count = fingerprint_engine.load_index_from_db(db)
        mel_count = melody_engine.load_contours_from_db(db)
        print(f"[startup] Hot index preloaded: {fp_count} fingerprints, {mel_count} melodies")
        # Warm up DSP engine to avoid cold JIT compilation lag on first user request
        try:
            dummy = np.zeros(fingerprint_engine.N_FFT, dtype=np.float32)
            fingerprint_engine._spectrogram(dummy)
            print("[startup] DSP engine warmed up.")
        except Exception:
            pass

        # Step 7: Pre-load HumV3 Self-Built Two-Branch Embedding Model into RAM
        try:
            from app.hum_v3.engine import HumV3Engine
            h3_engine = HumV3Engine.get_instance()
            h3_engine.load_sync()
        except Exception as h3_err:
            print(f"[startup] HumV3 pre-load notice: {h3_err}")
    finally:
        db.close()

    redis_status = "connected" if ping_redis() else "disabled (using ultra-fast native in-memory engine)"
    print(f"[startup] Search engine: Native In-Memory (Redis: {redis_status})")
    print(f"[startup] Free tier limit: {FREE_TIER_LIMIT} identifications/month")



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_upload(upload: UploadFile) -> str:
    """Save uploaded file to temp path. Caller must delete it."""
    suffix = os.path.splitext(upload.filename or ".wav")[1] or ".wav"
    tmp_path = os.path.join(tempfile.gettempdir(), f"songid_{uuid.uuid4().hex}{suffix}")
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return tmp_path


def _get_user_id(request: Request) -> str:
    """
    Extract user identifier from request.
    Uses X-User-ID header if present, falls back to IP address.
    In production replace with JWT token verification.
    """
    return request.headers.get("X-User-ID") or request.client.host or "anonymous"


def _get_stream_url(song: Optional[Song], request: Optional[Request] = None) -> Optional[str]:
    if not song:
        return None
    if song.r2_url:
        return song.r2_url
    if request:
        forwarded_host = request.headers.get("x-forwarded-host")
        if forwarded_host:
            proto = request.headers.get("x-forwarded-proto", "https")
            return f"{proto}://{forwarded_host}/audio/{song.id}"
        base = str(request.base_url).rstrip("/")
        return f"{base}/audio/{song.id}"
    return f"/audio/{song.id}"


def _run_sync(tmp_path: str, mode: str, db: Session, request: Optional[Request] = None) -> dict:
    """
    Synchronous identification — used when Redis/Celery is unavailable.
    Same logic as the worker task.
    """
    fp_result = None
    mel_results = []
    fp_hint_id = None

    if mode in ("auto", "fingerprint"):
        try:
            fp_result = fingerprint_engine.match_fingerprints(tmp_path, db)
            print(f"[sync] fingerprint result: {fp_result}")
            if fp_result and fp_result.get("hint_song_id"):
                fp_hint_id = fp_result["hint_song_id"]
        except Exception as exc:
            print(f"[sync] fingerprint error: {exc}")

    fp_min_confidence = FP_CONFIDENCE_THRESHOLD
    if fp_result and fp_result["confidence"] >= fp_min_confidence:
        song = db.query(Song).filter_by(id=fp_result["song_id"]).first()
        stream_url = fp_result.get("stream_url") or _get_stream_url(song, request)
        return {
            "status":     "done",
            "song":       fp_result["title"],
            "artist":     fp_result.get("artist"),
            "confidence": fp_result["confidence"],
            "method":     "fingerprint",
            "stream_url": stream_url,
        }

    if mode != "fingerprint":
        # Step 7: HumV3 Two-Branch Embedding Model Search
        try:
            from app.hum_v3.engine import HumV3Engine
            h3_engine = HumV3Engine.get_instance()
            if not h3_engine.is_loaded():
                h3_engine.load_sync()
            if h3_engine.is_loaded():
                h3_res = h3_engine.index.search_audio(tmp_path, top_k=5)
                top = h3_res.get("top_song")
                if top and (not h3_res.get("is_low_confidence", False) or top.get("confidence", 0) >= 0.50):
                    top_song = db.query(Song).filter_by(id=top["song_id"]).first()
                    alternatives = []
                    for alt in h3_res.get("matches", [])[1:]:
                        alt_song = db.query(Song).filter_by(id=alt["song_id"]).first()
                        alternatives.append({
                            "song_id": alt["song_id"],
                            "title": alt["title"],
                            "artist": alt.get("artist") or (alt_song.artist if alt_song else "Loveworld Song"),
                            "confidence": alt["confidence"],
                            "stream_url": _get_stream_url(alt_song, request) if alt_song else None,
                        })
                    return {
                        "status":       "done",
                        "song":         top["title"],
                        "artist":       top.get("artist") or (top_song.artist if top_song else "Loveworld Song"),
                        "confidence":   top["confidence"],
                        "method":       "hum_v3",
                        "stream_url":   _get_stream_url(top_song, request),
                        "alternatives": alternatives,
                    }
        except Exception as h3_exc:
            print(f"[sync] hum_v3 search notice: {h3_exc}")

        # Step 7 Requirement: Keep DTW as fallback for comparison
        try:
            mel_results = melody_engine.match_melody(tmp_path, db, top_n=5, hint_song_id=fp_hint_id)
            print(f"[sync] melody DTW results: {mel_results}")
        except Exception as exc:
            print(f"[sync] melody error: {exc}")

        if mel_results:
            top = mel_results[0]
            top_song = db.query(Song).filter_by(id=top["song_id"]).first()
            alternatives = []
            for alt in mel_results[1:]:
                alt_song = db.query(Song).filter_by(id=alt["song_id"]).first()
                alternatives.append({
                    "song_id": alt["song_id"],
                    "title": alt["title"],
                    "artist": alt.get("artist") or (alt_song.artist if alt_song else "Loveworld Song"),
                    "confidence": alt["confidence"],
                    "stream_url": _get_stream_url(alt_song, request) if alt_song else None,
                })
            return {
                "status":       "done",
                "song":         top["title"],
                "artist":       top.get("artist") or (top_song.artist if top_song else "Loveworld Song"),
                "confidence":   top["confidence"],
                "method":       "melody",
                "stream_url":   _get_stream_url(top_song, request),
                "alternatives": alternatives,
            }

    return {"status": "done", "song": None, "artist": None, "confidence": 0.0, "method": "none"}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    redis_ok = ping_redis()
    return {
        "status": "ok",
        "redis":  "connected" if redis_ok else "unavailable",
        "mode":   "async" if redis_ok else "sync",
    }


@app.post("/identify")
async def identify(
    request: Request,
    audio: UploadFile = File(..., description="Audio clip (wav, mp3, m4a)"),
    mode:  str        = Form("auto", description="auto | fingerprint | melody"),
    db:    Session    = Depends(get_db),
):
    """
    Identify a song from an audio clip or hum.

    Returns immediately with a job_id if Redis is available (async mode).
    Falls back to synchronous processing if Redis is down.

    Async response:
        { "job_id": "...", "status": "pending", "poll_url": "/result/..." }

    Sync response (Redis unavailable):
        { "status": "done", "song": "...", "confidence": 0.0, "method": "..." }
    """
    if mode not in ("auto", "fingerprint", "melody"):
        raise HTTPException(status_code=422, detail="mode must be auto, fingerprint, or melody")

    user_id = _get_user_id(request)

    # --- Free tier check ---
    redis_ok = ping_redis()
    # Force sync mode — async worker has cross-container file sharing issues
    # Will switch back to async when R2 temp file upload is implemented
    use_async = False
    # Free tier disabled during testing
    # if redis_ok:
    #     remaining = get_remaining_free(user_id)
    #     if remaining <= 0: raise HTTPException(402, ...)

    tmp_path = _save_upload(audio)
    file_size = os.path.getsize(tmp_path)
    print(f"[identify] user={user_id} file={audio.filename} size={file_size}b mode={mode}")

    # --- Async path (Redis available) ---
    if redis_ok and use_async:
        try:
            import base64
            from app.worker import identify_task

            # Read audio file and encode as base64 so worker can receive it
            # cross-container (worker has different /tmp/ filesystem)
            with open(tmp_path, "rb") as f:
                audio_bytes = f.read()
            ext = os.path.splitext(tmp_path)[1] or ".wav"
            audio_b64 = "b64:" + ext + ":" + base64.b64encode(audio_bytes).decode("ascii")

            task = identify_task.delay(audio_b64, mode, user_id)

            # Increment usage counter
            increment_usage(user_id)
            remaining_after = get_remaining_free(user_id)

            # Cleanup local temp file — worker has the data in base64
            try:
                os.remove(tmp_path)
            except OSError:
                pass

            return {
                "job_id":    task.id,
                "status":    "pending",
                "poll_url":  f"/result/{task.id}",
                "free_remaining": remaining_after,
            }
        except Exception as exc:
            print(f"[identify] queue error, falling back to sync: {exc}")
            # Fall through to sync

    # --- Sync fallback (Redis unavailable) ---
    try:
        if redis_ok:
            increment_usage(user_id)
        result = _run_sync(tmp_path, mode, db, request=request)
        result["free_remaining"] = get_remaining_free(user_id)
        return result
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


@app.post("/search/hum")
async def search_hum(
    audio: UploadFile = File(..., description="Audio clip (wav, mp3, m4a)"),
    top_k: int = Form(5, description="Number of top songs to return"),
    include_dtw: bool = Form(False, description="Run DTW comparison fallback if confidence is low"),
    db: Session = Depends(get_db),
):
    """
    Step 7 Endpoint: Humming Search v3 — Two-Branch Self-Built Embedding Model.
    Returns ranked songs only with calibrated confidence scores and low-confidence warnings.
    """
    from app.hum_v3.engine import HumV3Engine

    tmp_path = _save_upload(audio)
    try:
        engine = HumV3Engine.get_instance()
        res = await engine.search_audio_queued(
            audio_path=tmp_path,
            db_session=db if include_dtw else None,
            top_k=top_k,
            allow_dtw_fallback=include_dtw,
        )
        return res
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


@app.get("/result/{job_id}")
def get_result(job_id: str):
    """
    Poll for an identification result.

    Returns:
        { "status": "pending" }  — still processing
        { "status": "done", "song": "...", ... }  — complete
        { "status": "failed", "error": "..." }  — error
    """
    # Check Redis cache first (fastest path)
    if ping_redis():
        try:
            cached = cache_redis.get(f"result:{job_id}")
            if cached:
                return json.loads(cached)
        except Exception:
            pass

        # Check Celery task state
        try:
            from app.worker import identify_task
            task = identify_task.AsyncResult(job_id)

            if task.state == "PENDING":
                return {"status": "pending", "job_id": job_id}
            elif task.state == "STARTED":
                return {"status": "pending", "job_id": job_id}
            elif task.state == "SUCCESS":
                result = task.result or {}
                result["status"] = "done"
                return result
            elif task.state == "FAILURE":
                return {"status": "failed", "error": str(task.info), "job_id": job_id}
            else:
                return {"status": "pending", "job_id": job_id}
        except Exception as exc:
            return {"status": "failed", "error": str(exc), "job_id": job_id}

    return {"status": "failed", "error": "Redis unavailable", "job_id": job_id}


@app.get("/usage")
def get_user_usage(request: Request):
    """Return current usage stats for the requesting user."""
    user_id = _get_user_id(request)
    used = get_usage(user_id)
    remaining = get_remaining_free(user_id)
    return {
        "user_id":        user_id,
        "used":           used,
        "free_limit":     FREE_TIER_LIMIT,
        "remaining_free": remaining,
        "is_free_tier":   remaining > 0,
    }


@app.post("/admin/ingest-url")
async def admin_ingest_url(
    title:      str = Form(...),
    artist:     str = Form(""),
    source_url: str = Form(...),
    db:         Session = Depends(get_db),
):
    """
    Admin endpoint — queue a song for ingestion from a URL.
    Returns immediately — Celery worker handles download + fingerprint in background.
    """
    from app.worker import ingest_url_task

    # Register song in DB immediately with the URL as stream_url
    existing = db.query(Song).filter(Song.r2_url == source_url).first()
    if existing:
        return {"status": "already_exists", "song_id": existing.id, "title": existing.title}

    song = Song(
        title=title,
        artist=artist or None,
        file_path=source_url,
        r2_key=None,
        r2_url=source_url,
    )
    db.add(song)
    db.commit()
    db.refresh(song)

    # Queue fingerprinting in background
    if ping_redis():
        ingest_url_task.delay(song.id, source_url)
        return {"status": "queued", "song_id": song.id, "title": song.title, "message": "Fingerprinting in background"}
    else:
        # Sync fallback
        import urllib.request
        from app import fingerprint_engine, melody_engine
        tmp_path = os.path.join(tempfile.gettempdir(), f"songid_{uuid.uuid4().hex}.mp3")
        try:
            urllib.request.urlretrieve(source_url, tmp_path)
            fp_count = fingerprint_engine.store_fingerprints(song.id, tmp_path, db)
            melody_engine.store_contour(song.id, tmp_path, db)
            return {"status": "ingested", "song_id": song.id, "title": song.title, "hashes": fp_count}
        finally:
            try: os.remove(tmp_path)
            except: pass


@app.get("/audio/{song_id}")
def stream_audio(song_id: int, db: Session = Depends(get_db)):
    """Stream audio file for a given song ID."""
    song = db.query(Song).filter(Song.id == song_id).first()
    if not song:
        raise HTTPException(status_code=404, detail="Song not found")

    if song.r2_url:
        return RedirectResponse(url=song.r2_url)

    # Check local library paths
    possible_paths = []
    if song.file_path:
        possible_paths.extend([
            song.file_path,
            os.path.join(os.path.dirname(__file__), "..", song.file_path),
            os.path.join(os.path.dirname(__file__), "..", "song_library", os.path.basename(song.file_path)),
            os.path.join("song_library", os.path.basename(song.file_path)),
        ])

    for p in possible_paths:
        if os.path.isfile(p):
            return FileResponse(p, media_type="audio/mpeg", filename=f"{song.title}.mp3")

    raise HTTPException(status_code=404, detail="Audio file not found on server")


@app.get("/songs")
def list_songs(request: Request, db: Session = Depends(get_db)):
    """Return all songs in the library with guaranteed stream_url."""
    songs = db.query(Song).all()
    forwarded_host = request.headers.get("x-forwarded-host")
    if forwarded_host:
        proto = request.headers.get("x-forwarded-proto", "https")
        base_url = f"{proto}://{forwarded_host}"
    else:
        base_url = str(request.base_url).rstrip("/")
    return [
        {
            "id":         s.id,
            "title":      s.title,
            "artist":     s.artist,
            "r2_key":     s.r2_key,
            "r2_url":     s.r2_url,
            "stream_url": s.r2_url or f"{base_url}/audio/{s.id}",
        }
        for s in songs
    ]


@app.post("/admin/upload")
async def admin_upload(
    audio:  UploadFile = File(..., description="Audio file to upload and ingest"),
    title:  str        = Form(..., description="Song title"),
    artist: str        = Form("", description="Artist name (optional)"),
    source_url: str    = Form("", description="Original source URL (optional, used as stream_url)"),
    db:     Session    = Depends(get_db),
):
    """
    Admin endpoint — upload a song with a real title.
    Runs fingerprint + melody ingest immediately.
    Stores source_url as the stream_url if provided.
    """
    from app import fingerprint_engine, melody_engine

    tmp_path = _save_upload(audio)

    try:
        # Try to upload to R2 if credentials are available
        r2_key = None
        r2_url = source_url or None

        try:
            from app.r2_storage import _get_client, R2_BUCKET_NAME, get_public_url
            from app.config import R2_ACCESS_KEY_ID
            if R2_ACCESS_KEY_ID:
                ext = os.path.splitext(audio.filename or ".mp3")[1] or ".mp3"
                r2_key = f"uploads/{uuid.uuid4().hex}{ext}"
                client = _get_client()
                client.upload_file(tmp_path, R2_BUCKET_NAME, r2_key)
                r2_url = get_public_url(r2_key)
        except Exception as r2_err:
            print(f"[admin/upload] R2 upload skipped: {r2_err}")
            # Continue without R2 — use source_url as stream_url

        # Save song metadata
        song = Song(
            title=title,
            artist=artist or None,
            file_path=r2_key or audio.filename,
            r2_key=r2_key,
            r2_url=r2_url,
        )
        db.add(song)
        db.commit()
        db.refresh(song)

        # Run fingerprint + melody ingest
        fp_count = fingerprint_engine.store_fingerprints(song.id, tmp_path, db)
        melody_ok = melody_engine.store_contour(song.id, tmp_path, db)

        return {
            "status":    "ingested",
            "song_id":   song.id,
            "title":     song.title,
            "artist":    song.artist,
            "r2_key":    r2_key,
            "stream_url": r2_url,
            "hashes":    fp_count,
            "melody":    melody_ok,
        }

    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Authentication Endpoints
# ---------------------------------------------------------------------------

class SignUpRequest(BaseModel):
    name: str
    email: str
    password: str


class SignInRequest(BaseModel):
    email: str
    password: str


class ResetPasswordRequest(BaseModel):
    email: str
    new_password: str


def _hash_password(password: str) -> str:
    """PBKDF2 HMAC SHA-256 password hash."""
    salt = b"shizzy_songsort_salt_2026"
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100000).hex()


@app.post("/auth/signup")
def auth_signup(req: SignUpRequest, db: Session = Depends(get_db)):
    email_clean = req.email.strip().lower()
    name_clean = req.name.strip()
    if not email_clean or "@" not in email_clean:
        raise HTTPException(status_code=400, detail="Please enter a valid email address.")
    if len(req.password) < 4:
        raise HTTPException(status_code=400, detail="Password must be at least 4 characters.")

    existing = db.query(User).filter_by(email=email_clean).first()
    if existing:
        raise HTTPException(status_code=400, detail="An account with this email already exists.")

    token = secrets.token_hex(24)
    user = User(
        name=name_clean or "Music Lover",
        email=email_clean,
        password_hash=_hash_password(req.password),
        token=token,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return {
        "status": "ok",
        "token": user.token,
        "user": {
            "id": user.id,
            "name": user.name,
            "email": user.email,
        },
    }


@app.post("/auth/signin")
def auth_signin(req: SignInRequest, db: Session = Depends(get_db)):
    email_clean = req.email.strip().lower()
    user = db.query(User).filter_by(email=email_clean).first()
    if not user:
        raise HTTPException(status_code=400, detail="Invalid email or password.")

    if user.password_hash != _hash_password(req.password):
        raise HTTPException(status_code=400, detail="Invalid email or password.")

    if not user.token:
        user.token = secrets.token_hex(24)
        db.commit()
        db.refresh(user)

    return {
        "status": "ok",
        "token": user.token,
        "user": {
            "id": user.id,
            "name": user.name,
            "email": user.email,
        },
    }


@app.post("/auth/reset-password")
def auth_reset_password(req: ResetPasswordRequest, db: Session = Depends(get_db)):
    """
    Reset password directly — user provides email + new password.
    No email verification for now. Will be upgraded with Resend later.
    """
    email_clean = req.email.strip().lower()
    if not email_clean or "@" not in email_clean:
        raise HTTPException(status_code=400, detail="Please enter a valid email address.")
    if not req.new_password or len(req.new_password) < 4:
        raise HTTPException(status_code=400, detail="Password must be at least 4 characters.")

    user = db.query(User).filter_by(email=email_clean).first()
    if not user:
        raise HTTPException(status_code=400, detail="No account found with that email address.")

    user.password_hash = _hash_password(req.new_password)
    # Invalidate existing sessions so they must sign in again with new password
    user.token = secrets.token_hex(24)
    db.commit()

    return {"status": "ok", "message": "Password updated successfully."}


@app.get("/auth/me")
def auth_me(request: Request, db: Session = Depends(get_db)):
    auth_header = request.headers.get("Authorization", "")
    token = auth_header.replace("Bearer ", "").strip()
    if not token:
        raise HTTPException(status_code=401, detail="Unauthorized")

    user = db.query(User).filter_by(token=token).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")

    return {
        "status": "ok",
        "user": {
            "id": user.id,
            "name": user.name,
            "email": user.email,
        },
    }

