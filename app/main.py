"""
main.py — FastAPI application for the Song ID service.

Endpoints:
    POST /identify        — submit audio, returns job_id instantly
    GET  /result/{job_id} — poll for identification result
    GET  /health          — service health + Redis/DB status
    GET  /songs           — list all songs in library (admin/debug)

Flow:
    1. App sends audio → POST /identify → gets job_id immediately (<100ms)
    2. Celery worker processes audio async (fingerprint + melody)
    3. App polls GET /result/{job_id} every second until done
    4. Result returned with song, confidence, method

If Redis/Celery is unavailable → falls back to synchronous processing.
"""

import os
import json
import shutil
import tempfile
import uuid
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.db import init_db, get_db, Song
from app import fingerprint_engine, melody_engine
from app.redis_client import ping_redis, cache_redis
from app.usage import get_remaining_free, increment_usage, is_within_free_tier
from app.config import FREE_TIER_LIMIT

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Song ID API",
    description="Shazam-style song identification — fingerprint + melody matching.",
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_upload_to_temp(upload: UploadFile) -> str:
    suffix = os.path.splitext(upload.filename or ".wav")[1] or ".wav"
    tmp_path = os.path.join(tempfile.gettempdir(), f"songid_{uuid.uuid4().hex}{suffix}")
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return tmp_path


def _identify_sync(tmp_path: str, mode: str, db: Session) -> dict:
    """
    Synchronous fallback identification — used when Redis/Celery unavailable.
    Same logic as the Celery worker.
    """
    fp_result = None
    if mode in ("auto", "fingerprint"):
        try:
            fp_result = fingerprint_engine.match_fingerprints(tmp_path, db)
            print(f"[fingerprint] result: {fp_result}")
        except Exception as exc:
            print(f"[fingerprint] error: {exc}")

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

    if mode == "fingerprint":
        return {"status": "done", "song": None, "artist": None, "confidence": 0.0, "method": "none"}

    try:
        mel_results = melody_engine.match_melody(tmp_path, db, top_n=3)
        print(f"[melody] results: {mel_results}")
    except Exception as exc:
        mel_results = []
        print(f"[melody] error: {exc}")

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
    redis_up = ping_redis()
    return {
        "status":  "ok",
        "redis":   "connected" if redis_up else "unavailable (sync mode)",
        "mode":    "async" if redis_up else "sync",
    }


@app.post("/identify")
async def identify(
    audio:   UploadFile = File(...),
    mode:    str        = Form("auto"),
    user_id: str        = Form("anonymous"),
    db:      Session    = Depends(get_db),
):
    """
    Submit audio for identification.

    - Returns instantly with a job_id if Redis is available (async mode)
    - Falls back to synchronous processing if Redis is down
    - Free tier: 3 identifications/month, then requires subscription
    """
    if mode not in ("auto", "fingerprint", "melody"):
        raise HTTPException(status_code=422, detail="mode must be 'auto', 'fingerprint', or 'melody'")

    # --- Free tier check ---
    redis_up = ping_redis()
    if redis_up and user_id != "anonymous":
        if not is_within_free_tier(user_id):
            remaining = get_remaining_free(user_id)
            raise HTTPException(
                status_code=402,
                detail={
                    "error":      "free_tier_exceeded",
                    "message":    f"You have used all {FREE_TIER_LIMIT} free identifications this month.",
                    "remaining":  remaining,
                    "upgrade_url": "/subscribe",
                }
            )

    # Save uploaded file
    tmp_path = _save_upload_to_temp(audio)
    file_size = os.path.getsize(tmp_path)
    print(f"[identify] user={user_id} file={audio.filename} size={file_size}b mode={mode}")

    # Track usage
    if redis_up and user_id != "anonymous":
        increment_usage(user_id)

    # --- Async mode (Redis + Celery available) ---
    if redis_up:
        try:
            from app.worker import identify_task
            task = identify_task.delay(tmp_path, mode, user_id)
            return {
                "status":  "queued",
                "job_id":  task.id,
                "message": "Audio queued for processing. Poll /result/{job_id} for result.",
            }
        except Exception as exc:
            print(f"[identify] Celery unavailable, falling back to sync: {exc}")

    # --- Sync fallback (no Redis/Celery) ---
    try:
        result = _identify_sync(tmp_path, mode, db)
        return result
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


@app.get("/result/{job_id}")
async def get_result(job_id: str):
    """
    Poll for identification result.

    Returns:
        {"status": "queued"}   — still processing
        {"status": "done", "song": ..., "confidence": ..., "method": ...}
        {"status": "failed", "error": ...}
    """
    # Check Redis cache first (fastest)
    if ping_redis():
        cached = cache_redis.get(f"result:{job_id}")
        if cached:
            return json.loads(cached)

    # Check Celery backend
    try:
        from app.worker import celery_app
        task = celery_app.AsyncResult(job_id)

        if task.state == "PENDING":
            return {"status": "queued", "job_id": job_id}
        elif task.state == "STARTED":
            return {"status": "processing", "job_id": job_id}
        elif task.state == "SUCCESS":
            return task.result
        elif task.state == "FAILURE":
            return {"status": "failed", "error": str(task.info), "job_id": job_id}
        else:
            return {"status": task.state.lower(), "job_id": job_id}
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")


@app.get("/usage/{user_id}")
def get_usage_info(user_id: str):
    """Return usage info for a user — used by the app to show remaining free uses."""
    from app.usage import get_usage
    used = get_usage(user_id)
    remaining = max(0, FREE_TIER_LIMIT - used)
    return {
        "user_id":        user_id,
        "used":           used,
        "remaining":      remaining,
        "free_limit":     FREE_TIER_LIMIT,
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
