#!/usr/bin/env python3
"""Evaluate a checkpoint on a session split; write metrics + time–force charts.

Defaults to the released DINOv2 checkpoint on the validation split (the held-out
pretraining chunks in data/splits/val.txt) — the ~5-minute checkpoint check.
Each window is scored teacher-forced: the ground-truth previous force is fed as
the model's anchor. (An autoregressive ``--mode rollout`` is also available.)

Writes overall_metrics.csv, per_session_metrics.csv, and pred_vs_gt / scatter
plots under <out>/<mode>/.

Examples:
  python scripts/evaluate.py                       # DINOv2, val split
  python scripts/evaluate.py --config dinov2       # Apache-2.0 DINOv2 checkpoint
  python scripts/evaluate.py --split demo_val.txt  # the robot-demo validation set
  python scripts/evaluate.py --checkpoint <finetuned.ckpt> --split demo_val.txt   # a calibrated model
  python scripts/evaluate.py --video --limit-sessions 2   # + overlay videos (needs data/videos)
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from _common import ROOT, read_split
from emg2force import DEFAULT_MODEL, ValConfig, get_config, list_models, run_validation


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DEFAULT_MODEL,
                    help=f"named model config {list_models()} — selects checkpoint + backbone")
    ap.add_argument("--checkpoint", default=None, help="explicit checkpoint path (overrides --config)")
    ap.add_argument("--split", default="val.txt", help="split name under data/splits or a path")
    ap.add_argument("--out", default=str(ROOT / "results/eval"))
    ap.add_argument("--mode", choices=["teacher_forced", "rollout"], default="teacher_forced",
                    help="teacher_forced (default, the reported metric) or autoregressive rollout")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--limit-sessions", type=int, default=0, help="0 = all")
    ap.add_argument("--video", action="store_true", help="render overlay videos (needs data/videos)")
    ap.add_argument("--save-raw", action="store_true", help="dump per-session npz + copy source video")
    ap.add_argument("--backbone", default=None,
                    help="timm backbone id; default = read from checkpoint config (DINOv2)")
    args = ap.parse_args()

    # Explicit --checkpoint wins; otherwise resolve the named --config.
    checkpoint = args.checkpoint or str(get_config(args.config).checkpoint_path)
    model_label = args.checkpoint or args.config

    sessions = read_split(args.split)
    if args.limit_sessions:
        sessions = sessions[:args.limit_sessions]
    print(f"Evaluating {len(sessions)} sessions from '{args.split}'  (model: {model_label})")

    want_media = args.video or args.save_raw
    base = ValConfig(
        checkpoint=checkpoint, out_dir=args.out, test_sessions=sessions,
        use_prev_force=True, batch_size=args.batch_size, num_workers=args.num_workers,
        backbone=args.backbone,
        video_root=(ROOT / "data/videos") if want_media else None,
        render_overlay_video=args.video, save_raw_session_data=args.save_raw,
    )
    cfg = replace(base, eval_mode=args.mode, out_dir=str(Path(args.out) / args.mode))
    run_validation(cfg)


if __name__ == "__main__":
    main()
