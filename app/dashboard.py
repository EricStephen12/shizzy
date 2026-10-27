"""
dashboard.py — Control Center and Batch Ingestion Dashboard for Song Sort.
Serves a responsive Web Dashboard at /admin and provides API endpoints
for batch link ingestion, R2 browsing, catalog management, and live stats.
"""

import os
import re
import tempfile
import urllib.request
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from app import fingerprint_engine, melody_engine, r2_storage
from app.config import R2_BUCKET_NAME, R2_PUBLIC_URL
from app.db import get_db, Song, SongFingerprint, MelodyContour

router = APIRouter(prefix="/admin", tags=["admin-dashboard"])


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------
class IngestItem(BaseModel):
    title: str
    artist: Optional[str] = "Loveworld Singers"
    url: str

class BatchIngestRequest(BaseModel):
    raw_text: Optional[str] = None
    items: Optional[List[IngestItem]] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_raw_song_text(raw_text: str) -> List[dict]:
    """
    Parses pasted text with any of the following formats:
      1. Title | Artist | URL
      2. Title, Artist, URL
      3. URL (derives title from filename)
    """
    lines = [line.strip() for line in raw_text.strip().split("\n") if line.strip()]
    parsed = []

    for line in lines:
        # Check if line contains pipe '|'
        if "|" in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3:
                parsed.append({"title": parts[0], "artist": parts[1], "url": parts[2]})
            elif len(parts) == 2:
                parsed.append({"title": parts[0], "artist": "Loveworld Singers", "url": parts[1]})
            continue

        # Check if comma-separated
        if "," in line and ("http://" in line or "https://" in line):
            parts = [p.strip() for p in line.split(",")]
            # URL is usually the last part
            url_part = next((p for p in parts if p.startswith("http")), "")
            non_urls = [p for p in parts if not p.startswith("http")]
            title = non_urls[0] if len(non_urls) > 0 else "Unknown Title"
            artist = non_urls[1] if len(non_urls) > 1 else "Loveworld Singers"
            if url_part:
                parsed.append({"title": title, "artist": artist, "url": url_part})
                continue

        # Simple URL only
        if line.startswith("http://") or line.startswith("https://"):
            filename = line.split("?")[0].split("/")[-1]
            clean_name = os.path.splitext(filename)[0]
            # Replace underscores/hyphens and clean up
            clean_name = re.sub(r"[_\-]+", " ", clean_name).strip().title()
            # Remove common prefixes like 'take 1' or timestamps
            clean_name = re.sub(r"\bTake \d+\b", "", clean_name, flags=re.I).strip()
            clean_name = re.sub(r"\b\d+s \d+ms\b", "", clean_name, flags=re.I).strip()
            clean_name = re.sub(r"\b\d{10,}\b", "", clean_name).strip()
            title = clean_name or "Untitled Track"
            parsed.append({"title": title, "artist": "Loveworld Singers", "url": line})

    return parsed


def _ingest_single_url(title: str, artist: str, url: str, db: Session) -> dict:
    """Download, fingerprint, and store a single song in Postgres."""
    # Check if already ingested
    existing = db.query(Song).filter((Song.r2_url == url) | (Song.file_path == url)).first()
    if existing:
        return {
            "status": "already_exists",
            "song_id": existing.id,
            "title": existing.title,
            "artist": existing.artist,
            "hashes": 0,
        }

    # Register song row
    song = Song(
        title=title.strip(),
        artist=artist.strip() if artist else "Loveworld Singers",
        file_path=url,
        r2_key=None,
        r2_url=url,
    )
    db.add(song)
    db.commit()
    db.refresh(song)

    # Download temp file
    ext = os.path.splitext(url.split("?")[0])[1] or ".mp3"
    tmp_path = os.path.join(tempfile.gettempdir(), f"songid_batch_{uuid.uuid4().hex}{ext}")

    try:
        # User-agent header to avoid bot blocks
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=45) as resp, open(tmp_path, "wb") as f:
            f.write(resp.read())

        # Fingerprint
        fp_count = fingerprint_engine.store_fingerprints(song.id, tmp_path, db)
        # Melody
        melody_ok = melody_engine.store_contour(song.id, tmp_path, db)

        return {
            "status": "success",
            "song_id": song.id,
            "title": song.title,
            "artist": song.artist,
            "hashes": fp_count,
            "melody": bool(melody_ok),
        }
    except Exception as exc:
        db.rollback()
        # Clean up song record if audio failed
        db.delete(song)
        db.commit()
        raise exc
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@router.get("/api/stats")
def get_stats(db: Session = Depends(get_db)):
    """Live stats from Postgres and R2."""
    try:
        songs_count = db.query(Song).count()
        fp_count = db.query(SongFingerprint).count()
        mel_count = db.query(MelodyContour).count()
        r2_keys = r2_storage.list_audio_keys()
        return {
            "status": "online",
            "songs_count": songs_count,
            "fingerprints_count": fp_count,
            "melodies_count": mel_count,
            "r2_bucket": R2_BUCKET_NAME,
            "r2_files_count": len(r2_keys),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/api/parse-links")
def parse_links(req: BatchIngestRequest):
    """Preview parsed song links before starting batch ingest."""
    if not req.raw_text:
        return {"items": []}
    items = _parse_raw_song_text(req.raw_text)
    return {"count": len(items), "items": items}


@router.post("/api/ingest-single")
def ingest_single(item: IngestItem, db: Session = Depends(get_db)):
    """Ingest a single song by URL."""
    try:
        res = _ingest_single_url(item.title, item.artist or "Loveworld Singers", item.url, db)
        return res
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/api/batch-ingest-step")
def batch_ingest_step(item: IngestItem, db: Session = Depends(get_db)):
    """Process a single song within a batch run (used by frontend loop)."""
    try:
        res = _ingest_single_url(item.title, item.artist or "Loveworld Singers", item.url, db)
        return res
    except Exception as e:
        return {
            "status": "failed",
            "title": item.title,
            "error": str(e),
            "hashes": 0,
        }


@router.delete("/api/songs/{song_id}")
def delete_song(song_id: int, db: Session = Depends(get_db)):
    """Delete a song and all its fingerprints."""
    song = db.query(Song).filter(Song.id == song_id).first()
    if not song:
        raise HTTPException(status_code=404, detail="Song not found")
    title = song.title
    db.delete(song)
    db.commit()
    return {"status": "deleted", "song_id": song_id, "title": title}


@router.get("/api/r2-library")
def get_r2_library(prefix: str = "", limit: int = 100):
    """List audio files in R2 with public URLs."""
    try:
        all_keys = r2_storage.list_audio_keys(prefix=prefix)
        sliced = all_keys[:limit]
        items = []
        for k in sliced:
            clean = os.path.basename(k)
            name = os.path.splitext(clean)[0].replace("_", " ").title()
            items.append({
                "key": k,
                "title": name,
                "url": r2_storage.get_public_url(k),
            })
        return {"total": len(all_keys), "limit": limit, "items": items}
    except Exception as e:
        return {"error": str(e), "items": []}


# ---------------------------------------------------------------------------
# HTML Dashboard Page
# ---------------------------------------------------------------------------
@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
def admin_dashboard_html():
    """Renders the Song Sort Web Admin Dashboard."""
    return HTMLResponse(content=DASHBOARD_HTML)


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Song Sort — Control Center & Batch Ingest</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Newsreader:ital,opsz,wght@1,6..72,500;1,6..72,600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0C0E12;
      --card: #15181E;
      --card-hover: #1C2028;
      --border: #232730;
      --text: #F3F4F6;
      --muted: #9CA3AF;
      --accent: #2563EB;
      --accent-glow: rgba(37,99,235,0.25);
      --success: #10B981;
      --warning: #F59E0B;
      --danger: #EF4444;
      --font: 'Plus Jakarta Sans', -apple-system, sans-serif;
      --serif: 'Newsreader', Georgia, serif;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: var(--bg);
      color: var(--text);
      font-family: var(--font);
      min-height: 100vh;
      padding: 24px 20px 60px;
    }

    .container { max-width: 1100px; margin: 0 auto; }

    /* Top Nav */
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 24px;
      border-bottom: 1px solid var(--border);
      margin-bottom: 28px;
    }
    .brand { display: flex; align-items: baseline; gap: 12px; }
    .brand h1 {
      font-family: var(--serif);
      font-style: italic;
      font-size: 28px;
      font-weight: 600;
      letter-spacing: -0.5px;
    }
    .badge-alpha {
      background: rgba(37, 99, 235, 0.15);
      border: 1px solid rgba(37, 99, 235, 0.4);
      color: #60A5FA;
      font-size: 11px;
      font-weight: 600;
      padding: 3px 8px;
      border-radius: 20px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    .status-pill {
      display: flex;
      align-items: center;
      gap: 8px;
      background: #10241A;
      border: 1px solid #18472E;
      color: #34D399;
      font-size: 12px;
      padding: 6px 14px;
      border-radius: 20px;
    }
    .status-dot {
      width: 7px; height: 7px; border-radius: 50%;
      background: #10B981;
      box-shadow: 0 0 8px #10B981;
    }

    /* Stats Grid */
    .stats-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      margin-bottom: 32px;
    }
    .stat-card {
      background: var(--card);
      border: 1px solid var(--border);
      padding: 20px;
      border-radius: 14px;
      position: relative;
    }
    .stat-label { font-size: 13px; color: var(--muted); margin-bottom: 8px; font-weight: 500; }
    .stat-value { font-size: 30px; font-weight: 700; color: #fff; letter-spacing: -0.5px; }
    .stat-sub { font-size: 12px; color: var(--muted); margin-top: 4px; }

    /* Tabs */
    .tab-bar {
      display: flex;
      gap: 8px;
      border-bottom: 1px solid var(--border);
      margin-bottom: 24px;
      overflow-x: auto;
    }
    .tab-btn {
      background: none;
      border: none;
      color: var(--muted);
      font-family: var(--font);
      font-size: 14px;
      font-weight: 600;
      padding: 12px 18px;
      cursor: pointer;
      position: relative;
      transition: color 0.2s;
    }
    .tab-btn:hover { color: #fff; }
    .tab-btn.active { color: #fff; }
    .tab-btn.active::after {
      content: '';
      position: absolute;
      bottom: -1px; left: 0; right: 0;
      height: 2px;
      background: var(--accent);
      box-shadow: 0 0 10px var(--accent);
    }

    /* Panels */
    .tab-panel { display: none; }
    .tab-panel.active { display: block; }

    /* Ingest Form / Box */
    .box {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 24px;
      margin-bottom: 32px;
    }
    .box-title { font-size: 17px; font-weight: 600; margin-bottom: 6px; }
    .box-desc { font-size: 13px; color: var(--muted); margin-bottom: 20px; line-height: 1.5; }

    textarea, input[type="text"] {
      width: 100%;
      background: #0E1015;
      border: 1px solid var(--border);
      color: #fff;
      font-family: var(--font);
      font-size: 13px;
      padding: 12px 14px;
      border-radius: 10px;
      outline: none;
      transition: border-color 0.2s;
    }
    textarea:focus, input[type="text"]:focus {
      border-color: var(--accent);
    }
    textarea { min-height: 160px; font-family: monospace; line-height: 1.6; resize: vertical; }

    .input-row { display: flex; gap: 12px; margin-bottom: 14px; }
    .input-col { flex: 1; }
    label { display: block; font-size: 12px; color: var(--muted); margin-bottom: 6px; font-weight: 500; }

    .btn {
      background: var(--accent);
      color: #fff;
      font-family: var(--font);
      font-size: 14px;
      font-weight: 600;
      border: none;
      padding: 12px 22px;
      border-radius: 10px;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      transition: all 0.2s;
    }
    .btn:hover { background: #1D4ED8; }
    .btn:disabled { opacity: 0.5; cursor: not-allowed; }
    .btn-secondary { background: #232730; color: #fff; }
    .btn-secondary:hover { background: #2B303C; }

    /* Progress & Terminal */
    .progress-box {
      display: none;
      margin-top: 20px;
      background: #0E1015;
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 18px;
    }
    .progress-bar-bg {
      height: 8px;
      background: #1C2028;
      border-radius: 4px;
      overflow: hidden;
      margin: 12px 0;
    }
    .progress-fill {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, #2563EB, #10B981);
      transition: width 0.3s ease;
    }
    .terminal-log {
      max-height: 220px;
      overflow-y: auto;
      font-family: monospace;
      font-size: 12px;
      color: #A3A3A3;
      margin-top: 12px;
      padding-top: 12px;
      border-top: 1px solid var(--border);
      line-height: 1.7;
    }
    .terminal-log div.ok { color: #34D399; }
    .terminal-log div.err { color: #F87171; }
    .terminal-log div.skip { color: #FBBF24; }

    /* Songs Table */
    .table-container {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 16px;
      overflow: hidden;
    }
    .table-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 18px 24px;
      border-bottom: 1px solid var(--border);
    }
    .search-input { max-width: 280px; }

    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th {
      text-align: left;
      padding: 12px 24px;
      color: var(--muted);
      font-weight: 600;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      border-bottom: 1px solid var(--border);
    }
    td { padding: 14px 24px; border-bottom: 1px solid var(--border); }
    tr:last-child td { border-bottom: none; }
    tr:hover td { background: var(--card-hover); }

    .track-title { font-weight: 600; color: #fff; font-size: 14px; }
    .track-artist { color: var(--muted); font-size: 12px; margin-top: 2px; }
    .badge {
      display: inline-block;
      padding: 3px 8px;
      border-radius: 6px;
      font-size: 11px;
      font-weight: 600;
    }
    .badge-hashes { background: rgba(16, 185, 129, 0.15); color: #34D399; }
    .btn-del {
      background: none; border: none; color: #EF4444; cursor: pointer;
      font-size: 12px; font-weight: 600; padding: 4px 8px; border-radius: 6px;
    }
    .btn-del:hover { background: rgba(239, 68, 68, 0.15); }
    audio { height: 28px; width: 180px; }
  </style>
</head>
<body>
  <div class="container">
    <!-- Header -->
    <header>
      <div class="brand">
        <h1>Song Sort</h1>
        <span class="badge-alpha">Alpha Ops</span>
      </div>
      <div class="status-pill">
        <span class="status-dot"></span>
        <span id="connStatus">Railway Postgres Connected</span>
      </div>
    </header>

    <!-- Stats -->
    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-label">Songs Indexed</div>
        <div class="stat-value" id="statSongs">—</div>
        <div class="stat-sub">Ready for identification</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Acoustic Hashes</div>
        <div class="stat-value" id="statHashes">—</div>
        <div class="stat-sub">Spectrogram fingerprints</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Melody Contours</div>
        <div class="stat-value" id="statMelodies">—</div>
        <div class="stat-sub">Humming recognition</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Cloudflare R2 Media</div>
        <div class="stat-value" id="statR2">—</div>
        <div class="stat-sub">Audio files in bucket</div>
      </div>
    </div>

    <!-- Navigation Tabs -->
    <div class="tab-bar">
      <button class="tab-btn active" onclick="switchTab('batch')">Batch Link Ingest (800+)</button>
      <button class="tab-btn" onclick="switchTab('single')">Single Add</button>
      <button class="tab-btn" onclick="switchTab('r2')">Browse R2 Files</button>
      <button class="tab-btn" onclick="switchTab('catalog')">Library Catalog</button>
    </div>

    <!-- TAB 1: BATCH INGEST (800+ LINKS) -->
    <div id="tab-batch" class="tab-panel active">
      <div class="box">
        <h2 class="box-title">Batch Ingestion Tool</h2>
        <p class="box-desc">
          Paste your links below. You can paste raw audio URLs (one per line) or use <code>Title | Artist | URL</code>.
          The engine will download, calculate fingerprints, index melodies, and store directly into Railway Postgres.
        </p>

        <textarea id="batchText" placeholder="Paste your links here... Examples:
Still Water | Silver Tide | https://pub-...r2.dev/still_water.mp3
Holocene | Bon Iver | https://pub-...r2.dev/holocene.mp3
https://pub-...r2.dev/take_1_absolutely_powerful.m4a"></textarea>

        <div style="margin-top: 14px; display: flex; gap: 10px;">
          <button class="btn" id="startBatchBtn" onclick="startBatchIngest()">
            ⚡ Start Batch Ingest
          </button>
          <button class="btn btn-secondary" onclick="previewBatch()">
            Parse & Verify Links
          </button>
        </div>

        <!-- Live Progress -->
        <div class="progress-box" id="progressBox">
          <div style="display: flex; justify-content: space-between; font-size: 13px;">
            <span id="progressTitle">Ingesting songs...</span>
            <span id="progressStats" style="font-weight: 600;">0 / 0</span>
          </div>
          <div class="progress-bar-bg">
            <div class="progress-fill" id="progressFill"></div>
          </div>
          <div class="terminal-log" id="terminalLog"></div>
        </div>
      </div>
    </div>

    <!-- TAB 2: SINGLE ADD -->
    <div id="tab-single" class="tab-panel">
      <div class="box">
        <h2 class="box-title">Add Single Song</h2>
        <p class="box-desc">Quickly index a song with a direct audio link.</p>

        <div class="input-row">
          <div class="input-col">
            <label>Song Title</label>
            <input type="text" id="singleTitle" placeholder="e.g. Adonai">
          </div>
          <div class="input-col">
            <label>Artist Name</label>
            <input type="text" id="singleArtist" placeholder="e.g. Loveworld Singers" value="Loveworld Singers">
          </div>
        </div>
        <div style="margin-bottom: 16px;">
          <label>Audio URL (R2 / Cloudflare / Direct MP3 Link)</label>
          <input type="text" id="singleUrl" placeholder="https://pub-...r2.dev/audio/song.mp3">
        </div>
        <button class="btn" id="singleBtn" onclick="submitSingle()">Fingerprint & Save</button>
        <span id="singleStatus" style="margin-left: 14px; font-size: 13px;"></span>
      </div>
    </div>

    <!-- TAB 3: BROWSE R2 -->
    <div id="tab-r2" class="tab-panel">
      <div class="box">
        <h2 class="box-title">Files in Cloudflare R2 Bucket</h2>
        <p class="box-desc">Browse files currently in <code>rehearsalhub-media</code> and click to index.</p>
        <div style="display: flex; gap: 10px; margin-bottom: 16px;">
          <input type="text" id="r2Prefix" placeholder="Folder filter (e.g. zones/zone-001/audio/)" style="max-width: 350px;">
          <button class="btn btn-secondary" onclick="loadR2Files()">Fetch Files</button>
        </div>
        <div id="r2FileList" style="max-height: 400px; overflow-y: auto;">
          <div style="color: var(--muted); font-size: 13px;">Click "Fetch Files" to load from R2.</div>
        </div>
      </div>
    </div>

    <!-- TAB 4: CATALOG -->
    <div id="tab-catalog" class="tab-panel">
      <div class="table-container">
        <div class="table-header">
          <div>
            <h2 style="font-size: 16px; font-weight: 600;">Indexed Songs (<span id="tableCount">0</span>)</h2>
          </div>
          <input type="text" class="search-input" id="catalogSearch" placeholder="Search title or artist..." oninput="filterCatalog()">
        </div>
        <table>
          <thead>
            <tr>
              <th style="width: 50px;">ID</th>
              <th>Track Details</th>
              <th style="width: 200px;">Audio Preview</th>
              <th style="width: 80px; text-align: right;">Action</th>
            </tr>
          </thead>
          <tbody id="catalogBody">
            <tr><td colspan="4" style="text-align: center; color: var(--muted);">Loading library...</td></tr>
          </tbody>
        </table>
      </div>
    </div>

  </div>

  <script>
    let allSongs = [];
    let isIngesting = false;

    // Initialize stats and library
    window.onload = () => {
      fetchStats();
      fetchCatalog();
    };

    function switchTab(name) {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      event.target.classList.add('active');
      document.getElementById('tab-' + name).classList.add('active');
      if (name === 'catalog') fetchCatalog();
    }

    async function fetchStats() {
      try {
        const res = await fetch('/admin/api/stats');
        const data = await res.json();
        document.getElementById('statSongs').innerText = data.songs_count || 0;
        document.getElementById('statHashes').innerText = (data.fingerprints_count || 0).toLocaleString();
        document.getElementById('statMelodies').innerText = data.melodies_count || 0;
        document.getElementById('statR2').innerText = (data.r2_files_count || 0).toLocaleString();
      } catch (err) {
        document.getElementById('connStatus').innerText = 'Connection Error';
      }
    }

    async function fetchCatalog() {
      try {
        const res = await fetch('/songs');
        allSongs = await res.json();
        renderCatalog(allSongs);
      } catch (err) {
        console.error(err);
      }
    }

    function renderCatalog(songs) {
      document.getElementById('tableCount').innerText = songs.length;
      const tbody = document.getElementById('catalogBody');
      if (songs.length === 0) {
        tbody.innerHTML = '<tr><td colspan="4" style="text-align:center; padding: 30px; color: var(--muted);">No songs indexed yet.</td></tr>';
        return;
      }
      tbody.innerHTML = songs.map(s => `
        <tr>
          <td style="color: var(--muted); font-mono;">#${s.id}</td>
          <td>
            <div class="track-title">${s.title}</div>
            <div class="track-artist">${s.artist || 'Loveworld Singers'}</div>
          </td>
          <td>
            ${s.r2_url ? `<audio controls preload="none" src="${s.r2_url}"></audio>` : '<span style="color:var(--muted)">No stream URL</span>'}
          </td>
          <td style="text-align: right;">
            <button class="btn-del" onclick="deleteTrack(${s.id}, '${s.title.replace(/'/g, "\\'")}')">Delete</button>
          </td>
        </tr>
      `).join('');
    }

    function filterCatalog() {
      const q = document.getElementById('catalogSearch').value.toLowerCase();
      const filtered = allSongs.filter(s =>
        s.title.toLowerCase().includes(q) || (s.artist && s.artist.toLowerCase().includes(q))
      );
      renderCatalog(filtered);
    }

    async function deleteTrack(id, title) {
      if (!confirm(`Delete "${title}" and remove its fingerprints from database?`)) return;
      try {
        const res = await fetch(`/admin/api/songs/${id}`, { method: 'DELETE' });
        if (res.ok) {
          fetchStats();
          fetchCatalog();
        }
      } catch (err) {
        alert('Failed to delete track');
      }
    }

    // --- BATCH INGESTION ---
    async function previewBatch() {
      const raw = document.getElementById('batchText').value;
      if (!raw.trim()) { alert('Please paste some links first.'); return; }
      const res = await fetch('/admin/api/parse-links', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ raw_text: raw })
      });
      const data = await res.json();
      alert(`Parsed ${data.count} songs ready for batch ingestion!`);
    }

    async function startBatchIngest() {
      if (isIngesting) return;
      const raw = document.getElementById('batchText').value;
      if (!raw.trim()) { alert('Paste links first.'); return; }

      const parseRes = await fetch('/admin/api/parse-links', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ raw_text: raw })
      });
      const { items } = await parseRes.json();
      if (!items || items.length === 0) { alert('No valid links found.'); return; }

      isIngesting = true;
      document.getElementById('startBatchBtn').disabled = true;
      const box = document.getElementById('progressBox');
      const log = document.getElementById('terminalLog');
      const fill = document.getElementById('progressFill');
      const stats = document.getElementById('progressStats');
      box.style.display = 'block';
      log.innerHTML = `<div>Starting batch ingest of ${items.length} songs...</div>`;

      let done = 0;
      let okCount = 0;
      let skipCount = 0;

      for (let i = 0; i < items.length; i++) {
        const item = items[i];
        stats.innerText = `${i + 1} / ${items.length}`;
        fill.style.width = `${Math.round(((i + 1) / items.length) * 100)}%`;

        try {
          const res = await fetch('/admin/api/batch-ingest-step', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(item)
          });
          const resData = await res.json();

          if (resData.status === 'already_exists') {
            skipCount++;
            log.innerHTML += `<div class="skip">[${i+1}/${items.length}] Skipping "${item.title}" — already indexed</div>`;
          } else if (resData.status === 'success') {
            okCount++;
            log.innerHTML += `<div class="ok">[${i+1}/${items.length}] ✅ "${item.title}" (${resData.hashes} hashes)</div>`;
          } else {
            log.innerHTML += `<div class="err">[${i+1}/${items.length}] ❌ "${item.title}": ${resData.error || 'Failed'}</div>`;
          }
        } catch (err) {
          log.innerHTML += `<div class="err">[${i+1}/${items.length}] ❌ "${item.title}": Network timeout</div>`;
        }

        log.scrollTop = log.scrollHeight;
        done++;
      }

      log.innerHTML += `<div><b>Batch Complete!</b> ${okCount} new songs indexed, ${skipCount} already existed.</div>`;
      document.getElementById('startBatchBtn').disabled = false;
      isIngesting = false;
      fetchStats();
      fetchCatalog();
    }

    // --- SINGLE INGEST ---
    async function submitSingle() {
      const title = document.getElementById('singleTitle').value;
      const artist = document.getElementById('singleArtist').value;
      const url = document.getElementById('singleUrl').value;
      const status = document.getElementById('singleStatus');
      const btn = document.getElementById('singleBtn');

      if (!title || !url) { alert('Please provide Title and Audio URL.'); return; }

      btn.disabled = true;
      status.innerText = 'Fingerprinting... please wait.';
      status.style.color = '#F59E0B';

      try {
        const res = await fetch('/admin/api/ingest-single', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ title, artist, url })
        });
        const data = await res.json();
        if (data.status === 'already_exists') {
          status.innerText = 'Song already indexed!';
          status.style.color = '#F59E0B';
        } else {
          status.innerText = `✅ Ingested (${data.hashes} hashes stored)`;
          status.style.color = '#10B981';
          document.getElementById('singleTitle').value = '';
          document.getElementById('singleUrl').value = '';
          fetchStats();
        }
      } catch (e) {
        status.innerText = 'Failed to ingest audio.';
        status.style.color = '#EF4444';
      } finally {
        btn.disabled = false;
      }
    }

    // --- R2 BROWSER ---
    async function loadR2Files() {
      const prefix = document.getElementById('r2Prefix').value;
      const container = document.getElementById('r2FileList');
      container.innerHTML = '<div style="color:var(--muted)">Scanning R2 bucket...</div>';
      try {
        const res = await fetch(`/admin/api/r2-library?prefix=${encodeURIComponent(prefix)}&limit=80`);
        const data = await res.json();
        if (!data.items || data.items.length === 0) {
          container.innerHTML = '<div style="color:var(--muted)">No files found.</div>';
          return;
        }
        container.innerHTML = data.items.map(item => `
          <div style="display:flex; justify-content:space-between; align-items:center; padding:10px; border-bottom:1px solid var(--border); font-size:13px;">
            <div>
              <div style="font-weight:600; color:#fff;">${item.title}</div>
              <div style="color:var(--muted); font-size:11px;">${item.key}</div>
            </div>
            <button class="btn btn-secondary" style="padding:6px 12px; font-size:12px;" onclick="ingestR2Track('${item.title.replace(/'/g, "\\'")}', '${item.url}')">
              Fingerprint
            </button>
          </div>
        `).join('');
      } catch (err) {
        container.innerHTML = '<div style="color:red">Error loading R2 files</div>';
      }
    }

    async function ingestR2Track(title, url) {
      switchTab('single');
      document.getElementById('singleTitle').value = title;
      document.getElementById('singleUrl').value = url;
      submitSingle();
    }
  </script>
</body>
</html>
"""
