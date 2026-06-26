#!/usr/bin/env python3
"""Download the optional RGB overlay-video bundle from Google Drive.

The core release lets you evaluate the model and produce time–force charts with
no video at all. The *overlay* videos (side-by-side RGB frame | scrolling
prediction-vs-GT) are large (~1.3 GB), so they live on Google Drive instead of
in the repo. Run this once to fetch and unpack them into ``data/videos/``; then
``scripts/evaluate.py --video`` can render the overlays.

  python scripts/download_videos.py            # download + extract into data/
  python scripts/download_videos.py --list     # just print what the bundle holds

Requires ``gdown`` (``pip install gdown``). If you would rather download by hand,
open the share link printed below and extract the archive so that the per-session
folders land under ``data/videos/``.
"""
from __future__ import annotations

import argparse
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ── Bundle location ──────────────────────────────────────────────────────────
# TODO(maintainer): upload data/videos/ as a single .zip/.tar.gz to Google Drive,
# set the file to "Anyone with the link", and paste its file id here.
GDRIVE_FILE_ID = "REPLACE_WITH_GOOGLE_DRIVE_FILE_ID"
ARCHIVE_NAME = "emg2force_videos.zip"          # or .tar.gz
DEST_DIR = ROOT / "data"                        # archive expands to data/videos/...


def share_link() -> str:
    return f"https://drive.google.com/file/d/{GDRIVE_FILE_ID}/view?usp=sharing"


def _extract(archive: Path, dest: Path) -> None:
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
    elif archive.name.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(archive) as t:
            t.extractall(dest)
    else:
        raise SystemExit(f"Don't know how to extract {archive.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="print bundle contents and exit")
    ap.add_argument("--keep-archive", action="store_true", help="don't delete the archive after extracting")
    args = ap.parse_args()

    if args.list:
        print("Overlay-video bundle (optional):")
        print("  • data/videos/<session>/frames.mp4 + timestamps.npy + meta.json")
        print("  • the paired robot sessions needed to render them")
        print(f"\nShare link: {share_link()}")
        return

    if GDRIVE_FILE_ID.startswith("REPLACE_WITH"):
        sys.exit(
            "No Google Drive file id configured yet.\n"
            "Set GDRIVE_FILE_ID in scripts/download_videos.py (maintainers), or "
            "download the bundle by hand and extract it under data/videos/."
        )

    try:
        import gdown
    except ImportError:
        sys.exit("gdown is required: pip install gdown\n(or download by hand: " + share_link() + ")")

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    archive = DEST_DIR / ARCHIVE_NAME
    print(f"Downloading overlay videos → {archive}")
    gdown.download(id=GDRIVE_FILE_ID, output=str(archive), quiet=False)

    print(f"Extracting → {DEST_DIR}/")
    _extract(archive, DEST_DIR)
    if not args.keep_archive:
        archive.unlink(missing_ok=True)
    print(f"Done. Try:  python scripts/evaluate.py --video --split demo_val.txt --limit-sessions 2")


if __name__ == "__main__":
    main()
