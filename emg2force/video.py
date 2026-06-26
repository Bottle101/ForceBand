"""Optional per-session media: raw-array dumps and side-by-side overlay videos.

Consumes the self-describing video layout produced by ``scripts/prepare_videos.py``:

    data/videos/<session_stem>/
        frames.mp4        RGB recording, frames in capture order
        timestamps.npy    (n_frames,) float64 seconds, relative to session start
        meta.json         {fps, session_start_unix, n_frames, ...}

Frames are aligned to the model's per-sample predictions by timestamp, so each
RGB frame is shown next to the prediction/GT at its capture time. Requires
``opencv-python``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
_FINGER_COLORS = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e"]


def _hdf5_rel_timestamps(hdf5_path: Path, n: int) -> np.ndarray:
    import h5py
    with h5py.File(hdf5_path, "r") as f:
        ts = f["forceband"]["timeseries"]
        times = np.asarray(ts["timestamps"] if isinstance(ts, h5py.Group) else ts["time"],
                           dtype=np.float64)
    times = times[:n]
    return times - times[0] if times.size else times


def save_raw_session_data(stem, hdf5_path, preds, gts, video_root, out_dir: Path) -> Path:
    """Dump per-session arrays as .npz and copy the source video next to them."""
    import shutil
    out_dir.mkdir(parents=True, exist_ok=True)
    n = int(preds.shape[0])
    np.savez(
        out_dir / "force.npz",
        preds=preds.astype(np.float32), gts=gts.astype(np.float32),
        timestamps=_hdf5_rel_timestamps(Path(hdf5_path), n),
        sample_rate_hz=np.float64(1000.0), hdf5_source=str(hdf5_path),
    )
    if video_root is not None:
        src_dir = Path(video_root) / stem
        for name in ("frames.mp4", "timestamps.npy", "meta.json"):
            src = src_dir / name
            if src.exists():
                shutil.copy2(src, out_dir / name)
    log.info("Saved raw artifacts → %s (T=%d)", out_dir, n)
    return out_dir


def render_overlay(stem, hdf5_path, preds, gts, video_root, out_path: Path,
                   window_sec: float = 4.0, max_seconds: float = 0.0, dpi: int = 100) -> None:
    """Render a side-by-side mp4: RGB frame | scrolling pred-vs-GT plot."""
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    src_dir = Path(video_root) / stem
    mp4_path, ts_path, meta_path = src_dir / "frames.mp4", src_dir / "timestamps.npy", src_dir / "meta.json"
    if not (mp4_path.exists() and ts_path.exists()):
        log.warning("No paired video for %s in %s; skipping", stem, src_dir)
        return

    frame_rel = np.load(ts_path).astype(np.float64)          # seconds since session start
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    n_pred = int(preds.shape[0])
    hdf5_rel = _hdf5_rel_timestamps(Path(hdf5_path), n_pred)
    sample_idx = np.clip(np.searchsorted(hdf5_rel, frame_rel), 0, n_pred - 1)

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        log.warning("Cannot open %s; skipping", mp4_path)
        return
    vw, vh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(meta.get("fps") or cap.get(cv2.CAP_PROP_FPS) or 15.0)

    n_render = min(n_video, len(frame_rel))
    if max_seconds > 0:
        n_render = min(n_render, int(max_seconds * fps))

    half = int(window_sec * 1000)  # half-window in 1 kHz samples
    y_max = float(max(gts.max(), preds.max()) * 1.05) or 1.0
    y_min = float(min(gts.min(), preds.min(), 0.0))

    fig, axes = plt.subplots(5, 1, figsize=(max(640, vw) / dpi, vh / dpi), dpi=dpi, sharex=True)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.97, bottom=0.08, hspace=0.3)
    fig.patch.set_facecolor("white")
    lines_gt, lines_pr = [], []
    for c in range(5):
        ax = axes[c]
        ax.set_ylim(y_min, y_max); ax.set_xlim(-window_sec, window_sec)
        ax.set_ylabel(FINGER_NAMES[c], fontsize=9); ax.grid(True, linestyle="--", alpha=0.3)
        ax.tick_params(axis="both", labelsize=7)
        (lg,) = ax.plot([], [], color=_FINGER_COLORS[c], lw=1.5, label="GT")
        (lp,) = ax.plot([], [], color="black", lw=1.0, alpha=0.8, label="Pred")
        ax.axvline(0, color="red", lw=0.8, alpha=0.6)
        lines_gt.append(lg); lines_pr.append(lp)
        if c == 0:
            ax.legend(loc="upper right", fontsize=8, ncol=2)
    axes[-1].set_xlabel("time (s) around current frame", fontsize=8)
    fig.canvas.draw()
    plot_w, plot_h = fig.canvas.get_width_height()

    out_w, out_h = vw + plot_w, max(vh, plot_h)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
    if not writer.isOpened():
        cap.release(); plt.close(fig)
        raise RuntimeError(f"Cannot open writer for {out_path}")

    for f_idx in tqdm(range(n_render), desc=f"render {stem[:30]}"):
        ret, frame = cap.read()
        if not ret:
            break
        i_center = int(sample_idx[f_idx])
        lo, hi = max(0, i_center - half), min(n_pred, i_center + half + 1)
        t_local = (np.arange(lo, hi) - i_center) / 1000.0
        for c in range(5):
            lines_gt[c].set_data(t_local, gts[lo:hi, c])
            lines_pr[c].set_data(t_local, preds[lo:hi, c])
        fig.canvas.draw()
        plot_bgr = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba())[:, :, :3], cv2.COLOR_RGB2BGR)
        canvas = np.full((out_h, out_w, 3), 255, dtype=np.uint8)
        canvas[:vh, :vw] = frame
        canvas[:plot_h, vw:vw + plot_w] = plot_bgr
        cv2.putText(canvas, f"frame {f_idx}  t={f_idx / fps:5.2f}s", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        writer.write(canvas)

    writer.release(); cap.release(); plt.close(fig)
    log.info("Wrote %s (%d frames @ %g fps)", out_path, n_render, fps)


def export_session_media(session_results, cfg, out_dir: Path) -> None:
    """Driver: raw dumps and/or overlay videos for each session in the results."""
    video_root = Path(cfg.video_root) if cfg.video_root is not None else None
    limit = cfg.video_session_limit or len(session_results)
    for stem, hdf5_path, preds, gts in session_results[:limit]:
        if cfg.save_raw_session_data:
            try:
                save_raw_session_data(stem, hdf5_path, preds, gts, video_root,
                                      out_dir / "per_session_raw" / stem)
            except Exception as e:  # noqa: BLE001
                log.warning("Raw dump failed for %s: %s", stem, e)
        if cfg.render_overlay_video and video_root is not None:
            try:
                render_overlay(stem, hdf5_path, preds, gts, video_root,
                               out_dir / "per_session_video" / f"{stem}.mp4",
                               window_sec=cfg.video_window_sec, max_seconds=cfg.video_max_seconds)
            except Exception as e:  # noqa: BLE001
                log.warning("Overlay video failed for %s: %s", stem, e)
