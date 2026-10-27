"""
batch_ingest.py — CLI tool to ingest hundreds of songs from a text or CSV file.

Usage:
    python batch_ingest.py --file songs.txt
    python batch_ingest.py --file songs.csv
    python batch_ingest.py --r2 --limit 100

Format of songs.txt:
    Still Water | Silver Tide | https://pub-...r2.dev/audio/stillwater.mp3
    A King Is Born | Xano | https://pub-...r2.dev/audio/king_is_born.mp3
    https://pub-...r2.dev/audio/take_1_absolutely_powerful.m4a
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

from app.db import init_db, SessionLocal, Song
from app.dashboard import _parse_raw_song_text, _ingest_single_url
from app import r2_storage


def ingest_from_file(file_path: str):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        raw = f.read()

    items = _parse_raw_song_text(raw)
    total = len(items)
    print(f"\n=======================================================")
    print(f" Song Sort — Batch Ingesting {total} Songs")
    print(f"=======================================================\n")

    init_db()
    db = SessionLocal()

    success = 0
    skipped = 0
    failed = 0

    try:
        for i, item in enumerate(items, 1):
            title = item["title"]
            artist = item.get("artist", "Loveworld Singers")
            url = item["url"]

            print(f"[{i}/{total}] {title} ({artist}) ... ", end="", flush=True)

            try:
                res = _ingest_single_url(title, artist, url, db)
                status = res.get("status")
                if status == "already_exists":
                    print(f"already in database (id={res.get('song_id')}) — skipping")
                    skipped += 1
                elif status == "success":
                    print(f"OK! ({res.get('hashes')} fingerprints)")
                    success += 1
                else:
                    print(f"Done.")
                    success += 1
            except Exception as e:
                print(f"FAILED: {e}")
                failed += 1

            time.sleep(0.5)  # slight breather
    finally:
        db.close()

    print(f"\n=======================================================")
    print(f" Batch Complete: {success} newly indexed, {skipped} already existed, {failed} failed.")
    print(f" Total in file: {total}")
    print(f"=======================================================\n")


def ingest_from_r2(limit: int = 0):
    from app.ingest import ingest_from_r2 as r2_ingest
    r2_ingest(limit=limit)


def main():
    parser = argparse.ArgumentParser(description="Song Sort — Batch Song Ingestion")
    parser.add_argument("--file", "-f", help="Path to text or CSV file with song links", default=None)
    parser.add_argument("--r2", action="store_true", help="Ingest directly from Cloudflare R2 bucket")
    parser.add_argument("--limit", "-l", type=int, default=0, help="Max songs to ingest from R2 (0 = unlimited)")

    args = parser.parse_args()

    if args.file:
        ingest_from_file(args.file)
    elif args.r2:
        ingest_from_r2(limit=args.limit)
    else:
        print("Please provide a file with --file <filename> or use --r2 to pull from Cloudflare R2.")
        print("Example: python batch_ingest.py --file songs.txt")


if __name__ == "__main__":
    main()
