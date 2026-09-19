"""
main.py — FastAPI application for the Song ID service.

Single endpoint:
    POST /identify
      Content-Type: multipart/form-data
      Body:  audio (file)
      Query: mode (optional) = "auto" | "fingerprint" | "melody"
             Defaults to "auto" (fingerprint first, melody fallback)

Response:
    {
        "song":       "Song Title",
        "artist":     "Artist Name" | null,
        "confidence": 0.0–1.0,
        "method":     "fingerprint" | "melody" | "none"
    }
"""

import os
import shutil
import tempfile
import uuid
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.db import init_db, get_db
from app import fingerprint_engine, melody_engine

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Song ID API",
    description="Private Shazam-style song identification — fingerprint + melody matching.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # tighten this when you go to production
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    """Create DB tables on first launch."""
    init_db()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_upload_to_temp(upload: UploadFile) -> str:
    """Save an uploaded file to a temp path; caller must delete it."""
    suffix = os.path.splitext(upload.filename or ".wav")[1] or ".wav"
    tmp_path = os.path.join(tempfile.gettempdir(), f"songid_{uuid.uuid4().hex}{suffix}")
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return tmp_path


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/identify")
async def identify(
    audio: UploadFile = File(..., description="Audio clip to identify (mp3, wav, etc.)"),
    mode:  str        = Form("auto", description="'auto' | 'fingerprint' | 'melody'"),
    db:    Session    = Depends(get_db),
):
    """
    Identify a song from an audio clip or hum recording.

    - **audio**: the recorded clip (multipart file upload)
    - **mode**:  `auto` (default) tries exact fingerprint first, falls back to melody.
                 `fingerprint` forces exact matching only.
                 `melody` forces hum/melody matching only.
    """
    if mode not in ("auto", "fingerprint", "melody"):
        raise HTTPException(status_code=422, detail="mode must be 'auto', 'fingerprint', or 'melody'")

    tmp_path = _save_upload_to_temp(audio)
    try:
        # --- Fingerprint match ---
        if mode in ("auto", "fingerprint"):
            try:
                fp_result = fingerprint_engine.match_fingerprints(tmp_path, db)
            except Exception as exc:
                fp_result = None
                print(f"[fingerprint] error: {exc}")

            if fp_result:
                song = db.query(__import__('app.db', fromlist=['Song']).Song).filter_by(id=fp_result['song_id']).first()
                return {
                    "song":       fp_result["title"],
                    "artist":     fp_result.get("artist"),
                    "confidence": fp_result["confidence"],
                    "method":     "fingerprint",
                    "stream_url": song.r2_url if song else None,
                }

        if mode == "fingerprint":
            # Caller forced fingerprint-only — no match
            return {"song": None, "artist": None, "confidence": 0.0, "method": "none"}

        # --- Melody / hum match ---
        try:
            mel_results = melody_engine.match_melody(tmp_path, db, top_n=3)
        except Exception as exc:
            mel_results = []
            print(f"[melody] error: {exc}")

        if mel_results:
            top = mel_results[0]
            top_song = db.query(__import__('app.db', fromlist=['Song']).Song).filter_by(id=top['song_id']).first()
            return {
                "song":         top["title"],
                "artist":       top.get("artist"),
                "confidence":   top["confidence"],
                "method":       "melody",
                "stream_url":   top_song.r2_url if top_song else None,
                "alternatives": mel_results[1:],
            }

        # --- No match ---
        return {"song": None, "artist": None, "confidence": 0.0, "method": "none"}

    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


@app.get("/songs")
def list_songs(db: Session = Depends(get_db)):
    """Return all songs in the library (useful for debugging)."""
    from app.db import Song
    songs = db.query(Song).all()
    return [
        {
            "id":       s.id,
            "title":    s.title,
            "artist":   s.artist,
            "r2_key":   s.r2_key,
            "r2_url":   s.r2_url,
        }
        for s in songs
    ]
