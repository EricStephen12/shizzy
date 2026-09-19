# Song ID App

A private "Shazam for your own library" — two matching engines behind one FastAPI endpoint.

| Engine | Technique | Best for |
|---|---|---|
| Fingerprint | Spectrogram peak-hashing (Shazam-style) | Exact audio clips |
| Melody | pYIN pitch extraction + fastdtw | Humming / singing |

---

## Prerequisites

| Tool | Notes |
|---|---|
| Python 3.10+ | `python --version` |
| PostgreSQL | Create `song_id_db` manually |
| ffmpeg | Required by librosa for mp3 decoding |

### ffmpeg on Windows
1. Download the **essentials** build from <https://www.gyan.dev/ffmpeg/builds/>
2. Extract and copy `ffmpeg.exe`, `ffprobe.exe` to `C:\ffmpeg\bin\`
3. Add `C:\ffmpeg\bin` to your **System PATH** (System Properties → Advanced → Environment Variables)
4. Verify: `ffmpeg -version`

---

## Setup

```powershell
# 1. Create & activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Edit credentials
#    Open app/config.py and set DB_USER, DB_PASSWORD (and DB_HOST/DB_NAME if needed)

# 4. Create the Postgres database
#    psql -U postgres -c "CREATE DATABASE song_id_db;"
```

---

## Ingest your library

Drop audio files (mp3, wav, flac, ogg, m4a) into `song_library/`, then:

```powershell
cd song-id-app
python -m app.ingest
```

Output shows each file being registered, fingerprinted, and melody-indexed.  
Re-running is safe — already-ingested files are skipped.

---

## Run the API server

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Interactive docs: <http://localhost:8000/docs>

---

## Test with curl

### Identify a song clip (exact match)
```bash
curl -X POST http://localhost:8000/identify \
  -F "audio=@clip.wav"
```

### Identify a hum (melody match)
```bash
curl -X POST http://localhost:8000/identify \
  -F "audio=@hum.wav" \
  -F "mode=melody"
```

### List ingested songs
```bash
curl http://localhost:8000/songs
```

### Expected response
```json
{
  "song": "Bohemian Rhapsody",
  "artist": null,
  "confidence": 0.87,
  "method": "fingerprint"
}
```

---

## Project structure

```
song-id-app/
├── app/
│   ├── config.py            ← credentials & tuning knobs
│   ├── db.py                ← SQLAlchemy models
│   ├── fingerprint_engine.py← spectrogram peak-hashing
│   ├── melody_engine.py     ← pYIN + fastdtw
│   ├── ingest.py            ← one-time library processor
│   └── main.py              ← FastAPI app
├── song_library/            ← drop your mp3/wav files here
└── requirements.txt
```

---

## Tuning

Edit `app/config.py`:

| Setting | Default | Effect |
|---|---|---|
| `FP_CONFIDENCE_THRESHOLD` | `0.05` | Lower = more fingerprint matches (more false positives) |
| `FP_PEAKS_PER_SEC` | `10` | Higher = more accurate but slower ingest |
| `MELODY_DTW_MAX_DISTANCE` | `500` | Lower = stricter melody matching |
| `PYIN_FMIN` / `PYIN_FMAX` | C2 / C7 | Adjust if your hum goes very low or high |
