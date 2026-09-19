"""
melody_engine.py — Hum/melody matching via pYIN pitch extraction + fastdtw.

Algorithm:
  1. Load audio (hum or reference song)
  2. Run librosa.pyin() to get a frame-by-frame fundamental frequency (F0)
  3. Discard unvoiced frames (NaN)
  4. Convert F0 sequence to a *relative direction contour*:
       +1 = pitch rose, -1 = pitch fell, 0 = stayed roughly the same
  5. Store the contour in Postgres (once, at ingest time)

Matching:
  1. Extract contour from the incoming hum clip
  2. Run fastdtw against every stored contour
  3. Return top-3 matches sorted by DTW distance (lower = better)
  4. Normalise distance → 0–1 confidence (inverse, clipped)
"""

from typing import Optional

import numpy as np
import librosa
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sqlalchemy.orm import Session

from app.config import PYIN_FMIN, PYIN_FMAX, MELODY_DTW_MAX_DISTANCE
from app.db import Song, MelodyContour

# ---------------------------------------------------------------------------
# Contour extraction
# ---------------------------------------------------------------------------

def extract_contour(audio_path: str) -> list[int]:
    """
    Load audio and return a pitch-direction contour: list of -1, 0, or 1.

    Unvoiced frames (where pYIN returns NaN) are dropped so the contour
    only captures the melody, not silence.
    """
    y, sr = librosa.load(audio_path, sr=None, mono=True)

    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz(PYIN_FMIN),
        fmax=librosa.note_to_hz(PYIN_FMAX),
        sr=sr,
    )

    # Keep only voiced frames with a valid F0
    voiced_f0 = [float(v) for v, flag in zip(f0, voiced_flag)
                 if flag and not np.isnan(v)]

    if len(voiced_f0) < 2:
        return []

    contour = []
    for i in range(1, len(voiced_f0)):
        diff = voiced_f0[i] - voiced_f0[i - 1]
        if diff > 2.0:       # threshold: >2 Hz = clearly went up
            contour.append(1)
        elif diff < -2.0:    # threshold: >2 Hz = clearly went down
            contour.append(-1)
        else:
            contour.append(0)

    return contour


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def store_contour(song_id: int, audio_path: str, db: Session) -> bool:
    """
    Extract and store a melody contour for a song.
    Returns True if stored, False if the contour was too short (< 10 frames).
    Idempotent — overwrites any existing contour for the same song.
    """
    contour = extract_contour(audio_path)
    if len(contour) < 10:
        return False

    existing = db.query(MelodyContour).filter(MelodyContour.song_id == song_id).first()
    if existing:
        existing.contour = contour
    else:
        db.add(MelodyContour(song_id=song_id, contour=contour))

    db.commit()
    return True


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def _dtw_distance(a: list[int], b: list[int]) -> float:
    """Wrapper around fastdtw using Euclidean distance on 1-D int arrays."""
    arr_a = np.array(a, dtype=np.float32).reshape(-1, 1)
    arr_b = np.array(b, dtype=np.float32).reshape(-1, 1)
    distance, _ = fastdtw(arr_a, arr_b, dist=euclidean)
    return float(distance)


def _normalise_confidence(distance: float, contour_len: int) -> float:
    """
    Convert a raw DTW distance to a 0–1 confidence score.
    We normalise by contour length so short vs long hums are comparable.
    """
    if contour_len == 0:
        return 0.0
    per_frame = distance / max(contour_len, 1)
    # Map 0 → 1.0 confidence, MELODY_DTW_MAX_DISTANCE/len → 0.0 confidence
    max_per_frame = MELODY_DTW_MAX_DISTANCE / max(contour_len, 1)
    score = 1.0 - min(per_frame / max_per_frame, 1.0)
    return round(score, 4)


def match_melody(audio_path: str, db: Session, top_n: int = 3) -> list[dict]:
    """
    Match a hum/melody clip against all stored contours.

    Returns a list (up to top_n) of:
        { "song_id": int, "title": str, "artist": str, "confidence": float }
    sorted best-first.  Returns [] if the hum is too short or no matches exist.
    """
    hum_contour = extract_contour(audio_path)
    if len(hum_contour) < 5:
        return []

    all_contours = (
        db.query(MelodyContour)
          .join(Song, Song.id == MelodyContour.song_id)
          .all()
    )
    if not all_contours:
        return []

    scores = []
    for row in all_contours:
        lib_contour = row.contour
        if not lib_contour:
            continue
        dist = _dtw_distance(hum_contour, lib_contour)
        confidence = _normalise_confidence(dist, len(hum_contour))
        scores.append({
            "song_id":    row.song_id,
            "title":      row.song.title if row.song else "Unknown",
            "artist":     row.song.artist if row.song else None,
            "confidence": confidence,
            "_distance":  dist,          # internal, stripped before response
        })

    scores.sort(key=lambda x: x["_distance"])
    for s in scores:
        s.pop("_distance")

    return scores[:top_n]
