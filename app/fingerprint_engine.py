"""
fingerprint_engine.py — Custom spectrogram peak-hashing fingerprinter.

No external fingerprinting library required — built entirely on scipy/numpy
which are already in the stack.

Algorithm (classic "Shazam-style"):
  1. Load audio → mono, 22 kHz
  2. STFT → magnitude spectrogram
  3. Find local peaks in time-frequency space (constellation map)
  4. Pair each peak with its nearest neighbours in a fan-out window
  5. Hash each pair: sha1(freq1, freq2, Δtime) → hex
  6. Store (hash, offset) per song in Postgres

Matching:
  1. Extract hashes from the clip
  2. Look up each hash in the DB
  3. For each candidate song, build a histogram of (clip_offset − db_offset) deltas
  4. The largest coherent cluster → confidence score
"""

import hashlib
from collections import defaultdict
from typing import Optional

import numpy as np
from scipy.ndimage import maximum_filter
import librosa
from sqlalchemy.orm import Session

from app.config import (
    FP_PEAKS_PER_SEC,
    FP_FREQ_MIN,
    FP_FREQ_MAX,
    FP_CONFIDENCE_THRESHOLD,
)
from app.db import Song, SongFingerprint

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SR          = 22_050       # sample rate for loading
HOP_LENGTH  = 512          # STFT hop
N_FFT       = 4096         # STFT window — bigger window → finer frequency resolution
PEAK_NEIGHBOURHOOD = 10    # pixels in each axis for local-max filter
FAN_VALUE   = 5            # how many neighbour peaks to pair with each anchor
MIN_HASH_DT = 0            # min Δtime (frames) between paired peaks
MAX_HASH_DT = 200          # max Δtime (frames) between paired peaks


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_audio(audio_path: str) -> tuple[np.ndarray, int]:
    """Load to mono float32 at fixed sample rate."""
    y, sr = librosa.load(audio_path, sr=SR, mono=True)
    return y, int(sr)


def _spectrogram(y: np.ndarray) -> np.ndarray:
    """Return magnitude spectrogram clipped to our frequency band."""
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    # Convert frequency axis limits to bin indices
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    lo = int(np.searchsorted(freqs, FP_FREQ_MIN))
    hi = int(np.searchsorted(freqs, FP_FREQ_MAX)) + 1
    return S[lo:hi, :]


def _find_peaks(S: np.ndarray) -> list[tuple[int, int]]:
    """
    Return (freq_bin, time_frame) for prominent local peaks.
    Enforces FP_PEAKS_PER_SEC so we keep only the strongest peaks per second,
    preventing explosive database growth and out-of-memory crashes.
    """
    struct = np.ones((PEAK_NEIGHBOURHOOD, PEAK_NEIGHBOURHOOD))
    local_max = (maximum_filter(S, footprint=struct) == S)
    # Suppress lower 50% noise floor
    threshold = np.percentile(S, 50)
    detected = local_max & (S > threshold)
    freq_idxs, time_idxs = np.where(detected)
    if len(freq_idxs) == 0:
        return []

    # Select top FP_PEAKS_PER_SEC loudest peaks per second
    amps = S[freq_idxs, time_idxs]
    frames_per_sec = max(int(SR / HOP_LENGTH), 1)
    peaks_by_sec = defaultdict(list)
    for f, t, amp in zip(freq_idxs, time_idxs, amps):
        sec = int(t // frames_per_sec)
        peaks_by_sec[sec].append((amp, f, t))

    selected_peaks = []
    for sec in sorted(peaks_by_sec.keys()):
        sec_peaks = peaks_by_sec[sec]
        sec_peaks.sort(key=lambda x: x[0], reverse=True)
        for amp, f, t in sec_peaks[:FP_PEAKS_PER_SEC]:
            selected_peaks.append((int(f), int(t)))

    return selected_peaks


def _make_hashes(peaks: list[tuple[int, int]]) -> list[tuple[str, float]]:
    """
    For every anchor peak, pair it with up to FAN_VALUE later peaks.
    Return list of (hash_hex, anchor_offset_in_seconds).
    """
    # Sort by time so fan-out always goes forward
    peaks = sorted(peaks, key=lambda p: p[1])
    hashes = []
    for i, (f1, t1) in enumerate(peaks):
        for j in range(1, FAN_VALUE + 1):
            idx = i + j
            if idx >= len(peaks):
                break
            f2, t2 = peaks[idx]
            dt = t2 - t1
            if dt < MIN_HASH_DT or dt > MAX_HASH_DT:
                continue
            raw = f"{f1}|{f2}|{dt}"
            h = hashlib.sha1(raw.encode()).hexdigest()
            offset_sec = t1 * HOP_LENGTH / SR
            hashes.append((h, offset_sec))
    return hashes


def _extract_hashes(audio_path: str) -> list[tuple[str, float]]:
    """Full pipeline: audio → hashes."""
    y, _ = _load_audio(audio_path)
    S = _spectrogram(y)
    peaks = _find_peaks(S)
    return _make_hashes(peaks)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def store_fingerprints(song_id: int, audio_path: str, db: Session) -> int:
    """
    Fingerprint an audio file and persist its hashes to the DB.
    Returns the number of hashes stored.
    Idempotent — existing hashes for the same song_id are skipped via UNIQUE constraint.
    """
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    hashes = _extract_hashes(audio_path)
    if not hashes:
        return 0

    # Deduplicate in-memory to prevent duplicate keys in same batch
    seen = set()
    unique_rows = []
    for h, offset in hashes:
        key = (song_id, h, offset)
        if key not in seen:
            seen.add(key)
            unique_rows.append({"song_id": song_id, "hash_val": h, "offset": offset})

    # Chunk into safe batches of 250 rows to keep queries lightweight
    chunk_size = 250
    for i in range(0, len(unique_rows), chunk_size):
        chunk = unique_rows[i:i + chunk_size]
        stmt = (
            pg_insert(SongFingerprint)
            .values(chunk)
            .on_conflict_do_nothing(constraint="uq_fp_entry")
        )
        db.execute(stmt)
        db.commit()

    return len(unique_rows)


def match_fingerprints(audio_path: str, db: Session) -> Optional[dict]:
    """
    Match a clip against the stored fingerprint library.

    Returns:
        { "song_id": int, "title": str, "artist": str, "confidence": float }
        or None if no match exceeds FP_CONFIDENCE_THRESHOLD.
    """
    hashes = _extract_hashes(audio_path)
    if not hashes:
        return None

    hash_set = {h for h, _ in hashes}
    clip_offsets = {h: off for h, off in hashes}

    # Fetch specific columns instead of heavy ORM objects
    db_rows = (
        db.query(SongFingerprint.song_id, SongFingerprint.hash_val, SongFingerprint.offset)
          .filter(SongFingerprint.hash_val.in_(hash_set))
          .all()
    )

    if not db_rows:
        return None

    # Build delta histograms: for each song, count how many hashes align
    # at the same (db_offset − clip_offset) = consistent time alignment
    deltas = defaultdict(lambda: defaultdict(int))
    for s_id, h_val, s_offset in db_rows:
        clip_off = clip_offsets.get(str(h_val), 0.0)
        delta = round(float(s_offset) - clip_off, 2)   # rounded to 10 ms bins
        deltas[int(s_id)][delta] += 1

    # Best song = highest peak count in its delta histogram
    best_song_id = None
    best_count   = 0
    for song_id, delta_hist in deltas.items():
        peak = max(delta_hist.values(), default=0)
        if peak > best_count:
            best_count   = peak
            best_song_id = song_id

    if best_song_id is None:
        return None

    # Normalise against how many query hashes we had
    confidence = min(best_count / len(hashes), 1.0)
    if confidence < FP_CONFIDENCE_THRESHOLD:
        return None

    song = db.query(Song).filter(Song.id == best_song_id).first()
    return {
        "song_id":    best_song_id,
        "title":      song.title if song else "Unknown",
        "artist":     song.artist if song else None,
        "confidence": round(confidence, 4),
    }
