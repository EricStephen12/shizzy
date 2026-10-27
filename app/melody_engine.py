"""
melody_engine.py — High-Accuracy Hum & Singing Recognition Engine.

Algorithm:
  1. Load audio clip (hum or reference song) → mono at 22,050 Hz.
  2. Extract fundamental frequency (F0) with librosa.pyin().
  3. Filter unvoiced / silence frames.
  4. Convert F0 in Hz to continuous MIDI semitones:  69 + 12 * log2(F0 / 440).
  5. Apply median filtering to smooth vocal vibrato / pitch tremor.
  6. Center the contour around its median pitch (Key Invariant: whether sung
     in a high or low pitch, the relative melody intervals are identical).
  7. Store the normalized semitone contour in Postgres.

Matching:
  1. Extract semitone contour from the hummed audio clip.
  2. Slide across reference contours using FastDTW (Dynamic Time Warping).
  3. Returns matches ranked by lowest semitone deviation.
"""

from typing import Optional, List
import numpy as np
import librosa
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.signal import medfilt
from sqlalchemy.orm import Session

from app.config import PYIN_FMIN, PYIN_FMAX
from app.db import Song, MelodyContour

# Sampling resolution: ~10.7 frames per second (smooth, fast, accurate)
HOP_LENGTH = 2048
SR = 22050

# Maximum average semitone error for a plausible melody hit
MAX_ACCEPTABLE_SEMITONE_ERROR = 2.2


def extract_contour(audio_path: str) -> List[float]:
    """
    Extract a key-invariant, semitone pitch contour from an audio clip.
    Returns a list of relative semitones (e.g. [-2.1, 0.0, +2.0, +3.5...])
    or [] if the audio has insufficient vocal/melodic content.
    """
    try:
        y, sr = librosa.load(audio_path, sr=SR, mono=True)
    except Exception as exc:
        print(f"[melody] Failed to load audio {audio_path}: {exc}")
        return []

    if len(y) < SR * 0.8:  # Less than 0.8s
        return []

    try:
        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz(PYIN_FMIN),
            fmax=librosa.note_to_hz(PYIN_FMAX),
            sr=sr,
            hop_length=HOP_LENGTH,
        )
    except Exception as exc:
        print(f"[melody] pYIN error: {exc}")
        return []

    # Extract voiced frequencies
    voiced_f0 = [
        float(v) for v, flag in zip(f0, voiced_flag)
        if flag and not np.isnan(v) and v > 50.0
    ]

    # Need at least ~1.5s of voiced melody to match accurately
    if len(voiced_f0) < 12:
        return []

    # Convert Hz to MIDI semitones: 69 + 12 * log2(f / 440)
    midi = 69.0 + 12.0 * np.log2(np.array(voiced_f0) / 440.0)

    # Smooth out human vocal vibrato (pitch wobble) with a 5-frame median filter
    if len(midi) >= 5:
        midi = medfilt(midi, kernel_size=5)

    # Key Invariance: subtract median pitch so key of C and key of G match identically
    median_pitch = float(np.median(midi))
    normalized_contour = midi - median_pitch

    # Round to 2 decimal places to save DB storage while preserving micro-tonal nuance
    return [round(float(p), 2) for p in normalized_contour]


def store_contour(song_id: int, audio_path: str, db: Session) -> bool:
    """
    Extract and save melody contour for a song.
    Overwrites any existing contour for the song.
    """
    contour = extract_contour(audio_path)
    if len(contour) < 15:
        return False

    existing = db.query(MelodyContour).filter(MelodyContour.song_id == song_id).first()
    if existing:
        existing.contour = contour
    else:
        db.add(MelodyContour(song_id=song_id, contour=contour))

    db.commit()
    return True


def _dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """FastDTW distance between two 1D pitch arrays."""
    dist, _ = fastdtw(a.reshape(-1, 1), b.reshape(-1, 1), dist=euclidean)
    return float(dist)


def _best_window_distance(hum: np.ndarray, lib: np.ndarray) -> float:
    """
    Slide a window equal to the hum length across the song contour.
    Returns the minimum average semitone error per frame.
    """
    hum_len = len(hum)
    lib_len = len(lib)

    if lib_len <= hum_len:
        raw_dist = _dtw_distance(hum, lib)
        return raw_dist / max(hum_len, 1)

    # Step through reference song in 3-frame increments (~0.3s steps)
    step = 3
    best_avg_error = float("inf")

    for start in range(0, lib_len - hum_len + 1, step):
        window = lib[start:start + hum_len]
        # Re-center window to handle local key variations within the song
        window_norm = window - np.median(window)
        raw_dist = _dtw_distance(hum, window_norm)
        avg_error = raw_dist / hum_len

        if avg_error < best_avg_error:
            best_avg_error = avg_error
            # Early break if an almost perfect match is found (< 0.3 semitones error)
            if best_avg_error < 0.3:
                break

    return best_avg_error


def match_melody(audio_path: str, db: Session, top_n: int = 3) -> List[dict]:
    """
    Match a hummed/sung clip against all stored song contours.
    Returns ranked list of candidate songs with confidence scores.
    """
    hum_list = extract_contour(audio_path)
    if len(hum_list) < 12:
        print("[melody] Hum too short or quiet to extract melody")
        return []

    hum_arr = np.array(hum_list, dtype=np.float32)
    # Ensure zero-median
    hum_arr = hum_arr - np.median(hum_arr)

    all_contours = (
        db.query(MelodyContour)
          .join(Song, Song.id == MelodyContour.song_id)
          .all()
    )
    if not all_contours:
        return []

    candidates = []
    for row in all_contours:
        lib_data = row.contour
        if not lib_data or len(lib_data) < 15:
            continue

        lib_arr = np.array(lib_data, dtype=np.float32)
        avg_semitone_error = _best_window_distance(hum_arr, lib_arr)

        if avg_semitone_error <= MAX_ACCEPTABLE_SEMITONE_ERROR:
            # Map error: 0.0 error -> 1.0 (100%), 2.2 error -> 0.0 (0%)
            confidence = max(0.0, 1.0 - (avg_semitone_error / MAX_ACCEPTABLE_SEMITONE_ERROR))
            confidence = round(float(confidence), 3)

            candidates.append({
                "song_id": row.song_id,
                "title": row.song.title if row.song else "Unknown",
                "artist": row.song.artist if row.song else "Loveworld Singers",
                "confidence": confidence,
                "avg_error": round(avg_semitone_error, 2),
            })

    # Sort by highest confidence (lowest error)
    candidates.sort(key=lambda x: x["confidence"], reverse=True)

    # Clean up internal fields
    results = []
    for c in candidates[:top_n]:
        results.append({
            "song_id": c["song_id"],
            "title": c["title"],
            "artist": c["artist"],
            "confidence": c["confidence"],
        })

    return results
