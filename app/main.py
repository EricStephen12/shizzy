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
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.db import init_db, get_db, Song
from app import fingerprint_engine, melody_engine
from app.redis_client import ping_redis, cache_redis
from app.usage import get_usage, increment_usage, get_remaining_free, is_within_free_tier
from app.config import FREE_TIER_LIMIT

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Song ID API",
    description="Private Shazam-style song identification — fingerprint + melody matching.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()
    redis_status = "connected" if ping_redis() else "unavailable (using Postgres fallback)"
    print(f"[startup] Redis: {redis_status}")
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


def _run_sync(tmp_path: str, mode: str, db: Session) -> dict:
    """
    Synchronous identification — used when Redis/Celery is unavailable.
    Same logic as the worker task.
    """
    fp_result = None
    mel_results = []

    if mode in ("auto", "fingerprint"):
        try:
            fp_result = fingerprint_engine.match_fingerprints(tmp_path, db)
            print(f"[sync] fingerprint result: {fp_result}")
        except Exception as exc:
            print(f"[sync] fingerprint error: {exc}")

    fp_min_confidence = 0.03 if mode == "fingerprint" else 0.05
    if fp_result and fp_result["confidence"] >= fp_min_confidence:
        song = db.query(Song).filter_by(id=fp_result["song_id"]).first()
        return {
            "status":     "done",
            "song":       fp_result["title"],
            "artist":     fp_result.get("artist"),
            "confidence": fp_result["confidence"],
            "method":     "fingerprint",
            "stream_url": song.r2_url if song else None,
        }

    if mode != "fingerprint":
        try:
            mel_results = melody_engine.match_melody(tmp_path, db, top_n=3)
            print(f"[sync] melody results: {mel_results}")
        except Exception as exc:
            print(f"[sync] melody error: {exc}")

        if mel_results:
            top = mel_results[0]
            top_song = db.query(Song).filter_by(id=top["song_id"]).first()
            return {
                "status":       "done",
                "song":         top["title"],
                "artist":       top.get("artist"),
                "confidence":   top["confidence"],
                "method":       "melody",
                "stream_url":   top_song.r2_url if top_song else None,
                "alternatives": mel_results[1:],
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
    if redis_ok:
        remaining = get_remaining_free(user_id)
        if remaining <= 0:
            raise HTTPException(
                status_code=402,
                detail={
                    "error":   "free_tier_exceeded",
                    "message": f"You have used your {FREE_TIER_LIMIT} free identifications this month.",
                    "upgrade": "Subscribe for $0.99/month for unlimited identifications.",
                }
            )

    tmp_path = _save_upload(audio)
    file_size = os.path.getsize(tmp_path)
    print(f"[identify] user={user_id} file={audio.filename} size={file_size}b mode={mode}")

    # --- Async path (Redis available) ---
    if redis_ok:
        try:
            from app.worker import identify_task
            task = identify_task.delay(tmp_path, mode, user_id)

            # Increment usage counter
            increment_usage(user_id)
            remaining_after = get_remaining_free(user_id)

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
        result = _run_sync(tmp_path, mode, db)
        result["free_remaining"] = get_remaining_free(user_id)
        return result
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


@app.get("/songs")
def list_songs(db: Session = Depends(get_db)):
    """Return all songs in the library."""
    songs = db.query(Song).all()
    return [
        {
            "id":     s.id,
            "title":  s.title,
            "artist": s.artist,
            "r2_key": s.r2_key,
            "r2_url": s.r2_url,
        }
        for s in songs
    ]


@app.post("/admin/upload")
async def admin_upload(
    audio:  UploadFile = File(..., description="Audio file to upload and ingest"),
    title:  str        = Form(..., description="Song title"),
    artist: str        = Form("", description="Artist name (optional)"),
    db:     Session    = Depends(get_db),
):
    """
    Admin endpoint — upload a song with a real title.
    Saves to R2, stores metadata in DB, runs fingerprint + melody ingest.
    """
    from app import fingerprint_engine, melody_engine
    from app.r2_storage import _get_client, R2_BUCKET_NAME, get_public_url

    # Save upload to temp
    tmp_path = _save_upload(audio)

    try:
        # Upload to R2
        ext = os.path.splitext(audio.filename or ".mp3")[1] or ".mp3"
        r2_key = f"uploads/{uuid.uuid4().hex}{ext}"

        client = _get_client()
        client.upload_file(tmp_path, R2_BUCKET_NAME, r2_key)
        r2_url = get_public_url(r2_key)

        # Save song metadata
        song = Song(
            title=title,
            artist=artist or None,
            file_path=r2_key,
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
            "r2_url":    r2_url,
            "hashes":    fp_count,
            "melody":    melody_ok,
        }

    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
