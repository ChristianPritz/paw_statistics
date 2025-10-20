#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_model.py

Downloads the model and example data from OSF (placeholders) into the repository:
- models/keypoint_model.pth                (single file)
- examples/demo_images/                    (download zip, extract into this folder)
- examples/paw_data/data.zip               (download zip, do NOT extract)

Usage:
    python download_model.py          # default: download all missing artifacts
    python download_model.py --force  # re-download everything
    python download_model.py --no-model --no-images --no-data  # selectively skip
"""

from __future__ import annotations
import argparse
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

# === PLACEHOLDER download URLs (replace with real OSF '/download' URLs) ===
MODEL_URL = "https://osf.io/PUT_MODEL_LINK/download"  # e.g. https://osf.io/abcd1/files/model.pth/download
IMAGES_ZIP_URL = "https://osf.io/PUT_IMAGES_ZIP/download"  # zip -> extract into examples/demo_images/
DATA_ZIP_URL = "https://osf.io/PUT_DATA_ZIP/download"  # zip -> save into examples/paw_data/data.zip (do NOT extract)
# =======================================================================

# Destination paths
MODEL_DEST = Path("models") / "keypoint_model.pth"
IMAGES_DIR = Path("examples") / "demo_images"
DATA_DIR = Path("examples") / "paw_data"
DATA_ZIP_DEST = DATA_DIR / "data.zip"


def _download_file(url: str, dest: Path, show_progress: bool = True, chunk_size: int = 8192) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".tmp")
    os.close(tmp_fd)
    try:
        with urllib.request.urlopen(url) as resp, open(tmp_path, "wb") as out:
            total = resp.getheader("Content-Length")
            if total is not None:
                total = int(total)
            downloaded = 0
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                if show_progress and total:
                    percent = downloaded * 100 / total
                    print(f"\rDownloading {dest.name}: {percent:5.1f}% ({downloaded}/{total} bytes)", end="", flush=True)
        if show_progress and total:
            print()  # newline after progress
        shutil.move(tmp_path, str(dest))
        print(f"✅ Saved to {dest}")
        return dest
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def download_model(url: str = MODEL_URL, dest: Path = MODEL_DEST, force: bool = False) -> Optional[Path]:
    if dest.exists() and not force:
        print(f"✅ Model already present at {dest}")
        return dest
    print("⬇️  Downloading model weights...")
    return _download_file(url, dest)


def download_and_extract_images(url: str = IMAGES_ZIP_URL, extract_to: Path = IMAGES_DIR, force: bool = False) -> Optional[Path]:
    """
    Download images ZIP and extract into extract_to.
    If extract_to exists and is non-empty and not force, skip.
    """
    if extract_to.exists() and any(extract_to.iterdir()) and not force:
        print(f"✅ Images already present in {extract_to} (skipping).")
        return extract_to

    extract_to.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        tmp_zip = Path(td) / "images.zip"
        print("⬇️  Downloading demo images (zip)...")
        _download_file(url, tmp_zip)
        print(f"📦 Extracting images to {extract_to} ...")
        try:
            with zipfile.ZipFile(tmp_zip, "r") as zf:
                zf.extractall(path=str(extract_to))
            print(f"✅ Images extracted to {extract_to}")
        except zipfile.BadZipFile:
            print("❌ Downloaded file is not a valid zip file.", file=sys.stderr)
            raise
    return extract_to


def download_data_zip(url: str = DATA_ZIP_URL, dest: Path = DATA_ZIP_DEST, force: bool = False) -> Optional[Path]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        print(f"✅ Data zip already present at {dest}")
        return dest
    print("⬇️  Downloading data zip (not extracting)...")
    return _download_file(url, dest)


def parse_args():
    p = argparse.ArgumentParser(description="Download model and example data for paw_statistics (placeholders).")
    p.add_argument("--force", action="store_true", help="Re-download all artifacts, overwriting existing files.")
    p.add_argument("--no-model", action="store_true", help="Skip downloading the model.")
    p.add_argument("--no-images", action="store_true", help="Skip downloading/extracting demo images.")
    p.add_argument("--no-data", action="store_true", help="Skip downloading the data zip.")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        if not args.no_model:
            try:
                download_model(force=args.force)
            except Exception as e:
                print(f"❌ Model download failed: {e}", file=sys.stderr)

        if not args.no_images:
            try:
                download_and_extract_images(force=args.force)
            except Exception as e:
                print(f"❌ Images download/extract failed: {e}", file=sys.stderr)

        if not args.no_data:
            try:
                download_data_zip(force=args.force)
            except Exception as e:
                print(f"❌ Data zip download failed: {e}", file=sys.stderr)

        print("All requested downloads finished.")
    except KeyboardInterrupt:
        print("\nAborted by user.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
