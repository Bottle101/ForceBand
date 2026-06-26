#!/usr/bin/env python3
"""Fine-tune (calibrate) the released checkpoint on robot-demonstration data.

Loads the released BasicTransformer weights, then continues training (short
warmup, low LR) on a small calibration set. The defaults reproduce the
calibration tutorial in the README: fit on the April robot demos 0–35
(``calib.txt``) and validate on the held-out demos 36–93 (``demo_val.txt``).

  # Calibrate the default (DINOv2) checkpoint on demos 0–35, validate on 36–93:
  python scripts/finetune.py

  # Same, but starting from the Apache-2.0 DINOv2 checkpoint:
  python scripts/finetune.py --config dinov2

  # Your own sessions (HDF5 layout as documented in the README):
  python scripts/finetune.py --train-split my_train.txt --val-split my_val.txt

After training it writes a clean, self-describing checkpoint (state_dict +
backbone id) that ``scripts/evaluate.py`` can load directly.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from _common import ROOT, read_split
from emg2force import DEFAULT_MODEL, Emg2ForceDataModule, Emg2ForceModule, get_config, list_models
from emg2force.model import build_force_model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DEFAULT_MODEL,
                    help=f"named model config {list_models()} — picks the pretrained checkpoint + backbone")
    ap.add_argument("--pretrained", default=None, help="explicit checkpoint to fine-tune from (overrides --config)")
    ap.add_argument("--backbone", default=None, help="explicit backbone id (else read from the checkpoint)")
    ap.add_argument("--train-split", default="calib.txt", help="calibration sessions (default: demos 0–35)")
    ap.add_argument("--val-split", default="demo_val.txt", help="held-out demos to validate on (default: 36–93)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--window-length", type=int, default=10000)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "lightning_logs"))
    ap.add_argument("--name", default="emg2force_finetune")
    args = ap.parse_args()
    cfg = get_config(args.config)
    pretrained = args.pretrained or str(cfg.checkpoint_path)

    pl.seed_everything(42, workers=True)
    train_sessions, val_sessions = read_split(args.train_split), read_split(args.val_split)
    print(f"fine-tune: train={len(train_sessions)} val={len(val_sessions)} sessions")

    datamodule = Emg2ForceDataModule(
        window_length=args.window_length, batch_size=args.batch_size, num_workers=args.num_workers,
        train_sessions=train_sessions, val_sessions=val_sessions, test_sessions=val_sessions,
    )

    # Resolve the pretrained checkpoint + its backbone, then build a matching
    # model and load the weights (backbone built untrained, then overwritten — no
    # download needed). --backbone overrides; otherwise use the checkpoint's own
    # stored backbone id, falling back to the --config default.
    ckpt = torch.load(pretrained, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    backbone = args.backbone
    if backbone is None and isinstance(ckpt, dict):
        backbone = ckpt.get("config", {}).get("backbone_timm_id")
    backbone = backbone or cfg.backbone
    print(f"backbone: {backbone}  | pretrained: {pretrained}")
    net = build_force_model(use_prev_force=True, backbone=backbone, pretrained_backbone=False)
    model = Emg2ForceModule(net, lr=args.lr, use_scheduler=True,
                            warmup_epochs=1, max_epochs=args.epochs)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"loaded pretrained weights ({len(state)} tensors; "
          f"missing={len(missing)}, unexpected={len(unexpected)})")

    callbacks = [
        ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1,
                        filename="best-{epoch:02d}-{val_loss:.2f}", save_last=True),
        EarlyStopping(monitor="val_loss", mode="min", patience=20, verbose=True),
    ]
    trainer = pl.Trainer(
        max_epochs=args.epochs, callbacks=callbacks,
        logger=TensorBoardLogger(args.out, name=args.name),
        accelerator="gpu", devices=1, log_every_n_steps=10,
        num_sanity_val_steps=0, precision="32-true",
    )
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)

    # Re-save the best epoch as a clean, self-describing checkpoint (state_dict +
    # backbone id) — same format as the released checkpoints, so evaluate.py can
    # auto-detect the backbone for the calibrated model.
    best_path = trainer.checkpoint_callback.best_model_path
    if best_path:
        state = torch.load(best_path, map_location="cpu")["state_dict"]
        clean_ckpt = Path(best_path).parent / "emg2force_finetuned.ckpt"
        torch.save({
            "state_dict": state,
            "config": {"backbone_timm_id": backbone},
            "metadata": {"finetuned_from": pretrained, "epochs": args.epochs, "lr": args.lr},
        }, clean_ckpt)
        print(f"\nSaved calibrated checkpoint → {clean_ckpt}")
        print("Evaluate + visualize it on the held-out demos:")
        print(f"  python scripts/evaluate.py --checkpoint {clean_ckpt} "
              f"--split {args.val_split} --out results/finetune_eval")


if __name__ == "__main__":
    main()
