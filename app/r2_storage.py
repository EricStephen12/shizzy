"""
r2_storage.py — Read-only interface to Cloudflare R2 for audio file access.

Operations:
  list_audio_keys()         — list all audio files in the bucket
  download_temp(key)        — download one file to a local temp path for processing
  get_public_url(key)       — return the public HTTPS streaming URL
  cleanup_temp(path)        — delete a previously downloaded temp file

The boto3 client is configured with R2's S3-compatible endpoint.
The token is read-only — this module CANNOT delete or upload anything.
"""

import os
import tempfile
import uuid
from typing import Optional

import boto3
from botocore.config import Config

from app.config import (
    R2_ACCOUNT_ID,
    R2_ACCESS_KEY_ID,
    R2_SECRET_KEY,
    R2_BUCKET_NAME,
    R2_PUBLIC_URL,
    R2_ENDPOINT_URL,
)

# Supported audio extensions to pull from the bucket
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}

# ---------------------------------------------------------------------------
# boto3 client (created once, reused)
# ---------------------------------------------------------------------------
_s3 = None

def _get_client():
    global _s3
    if _s3 is None:
        _s3 = boto3.client(
            "s3",
            endpoint_url=R2_ENDPOINT_URL,
            aws_access_key_id=R2_ACCESS_KEY_ID,
            aws_secret_access_key=R2_SECRET_KEY,
            config=Config(signature_version="s3v4"),
            region_name="auto",
        )
    return _s3


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_audio_keys(prefix: str = "") -> list[str]:
    """
    Return all object keys in the bucket that are audio files.
    Optionally filter by a folder prefix (e.g. "songs/").
    """
    client = _get_client()
    paginator = client.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=R2_BUCKET_NAME, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            ext = os.path.splitext(key)[1].lower()
            if ext in AUDIO_EXTENSIONS:
                keys.append(key)
    return keys


def download_temp(key: str) -> str:
    """
    Download an R2 object to a temporary local file.
    Returns the local temp file path.
    Caller must call cleanup_temp(path) when done.
    """
    client = _get_client()
    ext = os.path.splitext(key)[1] or ".audio"
    tmp_path = os.path.join(
        tempfile.gettempdir(),
        f"songid_r2_{uuid.uuid4().hex}{ext}"
    )
    client.download_file(R2_BUCKET_NAME, key, tmp_path)
    return tmp_path


def get_public_url(key: str) -> str:
    """
    Return the public HTTPS URL for streaming a file directly from R2.
    Requires Public Access to be enabled on the bucket.
    """
    base = R2_PUBLIC_URL.rstrip("/")
    return f"{base}/{key}"


def cleanup_temp(path: str) -> None:
    """Delete a temporary file downloaded by download_temp()."""
    try:
        os.remove(path)
    except OSError:
        pass
