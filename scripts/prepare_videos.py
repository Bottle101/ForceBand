#!/usr/bin/env python3
"""Build timestamp-aligned RGB videos for the cleanly-paired sessions.

Source RGB is stored as WebDataset ``.tar`` shards of timestamped JPEG frames
(``<idx>_<unix_ts>.jpg``). For each session marked ``paired`` in the pairing
manifest, this collects its shards, decodes frames in timestamp order, and writes:

    data/videos/<session_stem>/
        frames.mp4        frames in capture order, at the measured FPS
        timestamps.npy    (n_frames,) float64 seconds, relative to session start
        meta.json         {fps, session_start_unix, n_frames, status, shards}

The raw ``.tar`` shards are NOT part of the release; this is the build script
that produced the shipped ``frames.mp4`` files. Re-run it only if you have the
original WebDataset.

  python scripts/prepare_videos.py --split val --raw-data-root /path/to/data
"""
from __future__ import annotations

import argparse
import json
import re
import tarfile
from pathlib import Path

import cv2
import numpy as np

from _common import ROOT

_MEMBER_TS = re.compile(r"_(\d{10}\.\d+)\.jpg$")


def _frame_iter(tar_path: Path):
    """Yield (unix_ts, jpeg_bytes) for each timestamped frame in a tar shard."""
    with tarfile.open(tar_path, "r") as t:
        members = []
        for m in t.getmembers():
            mm = _MEMBER_TS.search(m.name)
            if mm:
                members.append((float(mm.group(1)), m))
        for ts, m in sorted(members, key=lambda x: x[0]):
            f = t.extractfile(m)
            if f is not None:
                yield ts, f.read()


def build_session_video(stem: str, shards: list[Path], session_start: float,
                        out_dir: Path, status: str) -> bool:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = out_dir / "frames.mp4"

    # First collect global timestamp order across shards (shards are sequential).
    shards = sorted(shards, key=lambda p: p.name)
    written_ts: list[float] = []
    writer = None
    size = None
    try:
        for shard in shards:
            if not shard.exists():
                print(f"  ! missing shard {shard}")
                continue
            for ts, jpg in _frame_iter(shard):
                img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                if writer is None:
                    h, w = img.shape[:2]
                    size = (w, h)
                    # Provisional FPS; rewritten in meta after counting all frames.
                    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), 16.0, size)
                elif (img.shape[1], img.shape[0]) != size:
                    img = cv2.resize(img, size)
                writer.write(img)
                written_ts.append(ts)
    finally:
        if writer is not None:
            writer.release()

    if not written_ts:
        print(f"  ! no frames for {stem}")
        return False

    ts = np.asarray(written_ts, dtype=np.float64)
    fps = float((len(ts) - 1) / (ts[-1] - ts[0])) if len(ts) > 1 and ts[-1] > ts[0] else 16.0
    np.save(out_dir / "timestamps.npy", ts - session_start)
    (out_dir / "meta.json").write_text(json.dumps({
        "fps": round(fps, 3), "session_start_unix": session_start,
        "n_frames": int(len(ts)), "status": status,
        "shards": [s.name for s in shards],
    }, indent=2))
    print(f"  ✓ {stem}: {len(ts)} frames @ {fps:.1f} fps  ({out_mp4.stat().st_size/1e6:.1f} MB)")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val", choices=["val", "train"])
    ap.add_argument("--raw-data-root", default="/media/botao/botao_ws/emg2force/data",
                    help="root holding the WebDataset rgb/*.tar shards")
    ap.add_argument("--status", nargs="+", default=["paired"],
                    help="manifest statuses to include (e.g. paired ECCV)")
    ap.add_argument("--manifest", default=str(ROOT / "data/emg_rgb_pairing.json"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    raw_root = Path(args.raw_data_root)
    manifest = json.loads(Path(args.manifest).read_text())
    rows = [r for r in manifest
            if r.get("split") == args.split
            and any(r["status"].startswith(s) for s in args.status)
            and r.get("shards")]
    if args.limit:
        rows = rows[:args.limit]
    print(f"{len(rows)} {args.split} sessions to process (status={args.status})")

    ok = 0
    for r in rows:
        stem = Path(r["file"]).stem
        shards = [raw_root / s for s in r["shards"]]
        out_dir = ROOT / "data/videos" / stem
        if (out_dir / "frames.mp4").exists() and (out_dir / "timestamps.npy").exists():
            print(f"  · {stem}: already built, skipping")
            ok += 1
            continue
        if build_session_video(stem, shards, float(r["start"]), out_dir, r["status"]):
            ok += 1
    print(f"\nDone: {ok}/{len(rows)} sessions have video under data/videos/")


if __name__ == "__main__":
    main()
