"""
worker.py — Celery worker for async audio identification.

Each job:
  1. Receives audio file path + job metadata
  2. Runs fingerprint engine (Redis index → Postgres fallback)
  3. Runs melody engine if fingerprint fails
  4. Stores result in Redis cache (expires in 5 minutes)
  5. App polls GET /result/{job_id} to retrieve it

Start worker:
    celery -A app.worker worker --loglevel=info --concurrency=4
"""

import os
import json
import uuid
import time

from celery import Celery
from app.config import REDIS_URL, FREE_TIER_LIMIT

# ---------------------------------------------------------------------------
# Celery app
# ---------------------------------------------------------------------------
celery_app = Celery(
    "songid",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=300,          # results expire after 5 minutes
    task_track_started=True,
    worker_prefetch_multiplier=1, # one task at a time per worker — fair processing
)


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, name="songid.ingest_url")
def ingest_url_task(self, song_id: int, source_url: str):
    """Download audio from URL and run fingerprint + melody ingest."""
    import urllib.request
    from app.db import SessionLocal
    from app import fingerprint_engine, melody_engine

    db = SessionLocal()
    tmp_path = None
    try:
        tmp_path = os.path.join(tempfile.gettempdir(), f"songid_ingest_{song_id}.mp3")
        print(f"[ingest_url] Downloading song {song_id} from {source_url}")
        urllib.request.urlretrieve(source_url, tmp_path)
        size = os.path.getsize(tmp_path)
        print(f"[ingest_url] Downloaded {size//1024}KB")

        fp_count = fingerprint_engine.store_fingerprints(song_id, tmp_path, db)
        melody_ok = melody_engine.store_contour(song_id, tmp_path, db)
        print(f"[ingest_url] Song {song_id} done — {fp_count} hashes, melody={melody_ok}")
        return {"song_id": song_id, "hashes": fp_count, "melody": melody_ok}
    except Exception as exc:
        print(f"[ingest_url] ERROR song {song_id}: {exc}")
        raise
    finally:
        db.close()
        if tmp_path:
            try: os.remove(tmp_path)
            except: pass


@celery_app.task(bind=True, name="songid.identify")
def identify_task(self, audio_path: str, mode: str = "auto", user_id: str = "anonymous"):
    """
    Main identification task. Runs fingerprint + melody matching.
    Result stored in Redis cache under key: result:{task_id}
    """
    from app.db import SessionLocal, Song
    from app import fingerprint_engine, melody_engine
    from app.redis_client import cache_redis

    job_id = self.request.id
    start_time = time.time()

    try:
        db = SessionLocal()

        fp_result = None
        mel_results = []

        # --- Fingerprint ---
        if mode in ("auto", "fingerprint"):
            try:
                fp_result = fingerprint_engine.match_fingerprints(audio_path, db)
            except Exception as exc:
                print(f"[worker] fingerprint error: {exc}")

        fp_min_confidence = 0.03 if mode == "fingerprint" else 0.05
        if fp_result and fp_result["confidence"] >= fp_min_confidence:
            song = db.query(Song).filter_by(id=fp_result["song_id"]).first()
            result = {
                "status":     "done",
                "song":       fp_result["title"],
                "artist":     fp_result.get("artist"),
                "confidence": fp_result["confidence"],
                "method":     "fingerprint",
                "stream_url": song.r2_url if song else None,
            }
        elif mode != "fingerprint":
            # --- Melody fallback ---
            try:
                mel_results = melody_engine.match_melody(audio_path, db, top_n=3)
            except Exception as exc:
                print(f"[worker] melody error: {exc}")

            if mel_results:
                top = mel_results[0]
                top_song = db.query(Song).filter_by(id=top["song_id"]).first()
                result = {
                    "status":       "done",
                    "song":         top["title"],
                    "artist":       top.get("artist"),
                    "confidence":   top["confidence"],
                    "method":       "melody",
                    "stream_url":   top_song.r2_url if top_song else None,
                    "alternatives": mel_results[1:],
                }
            else:
                result = {
                    "status":     "done",
                    "song":       None,
                    "artist":     None,
                    "confidence": 0.0,
                    "method":     "none",
                }
        else:
            result = {
                "status":     "done",
                "song":       None,
                "artist":     None,
                "confidence": 0.0,
                "method":     "none",
            }

        db.close()

        # Cleanup temp file
        try:
            os.remove(audio_path)
        except OSError:
            pass

        elapsed = round(time.time() - start_time, 3)
        result["processing_time_s"] = elapsed
        print(f"[worker] job {job_id} done in {elapsed}s — {result['method']} → {result['song']}")

        # Cache result in Redis for polling
        try:
            cache_redis.setex(f"result:{job_id}", 300, json.dumps(result))
        except Exception:
            pass  # Cache failure is non-fatal

        return result

    except Exception as exc:
        print(f"[worker] job {job_id} ERROR: {exc}")
        try:
            os.remove(audio_path)
        except OSError:
            pass
        raise
