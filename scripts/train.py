#!/usr/bin/env python3
"""Train the force model from scratch — the pretraining recipe.

Reproduces the recipe that produced the released checkpoint:
BasicTransformer(dino, use_prev_force), loss = MAE + 0.01*MSE, AdamW +
warmup→cosine, 10 s windows @ 1 kHz, over data/splits/train.txt.

NOTE: the full pretraining dataset (and its train.txt split) is **under
preparation** and is not part of this release — only a small held-out validation
subset ships (data/splits/val.txt). This script is included so the training
recipe is fully reproducible once the dataset is available. To adapt the model
to your own data today, use scripts/finetune.py (calibration) instead.

  python scripts/train.py --epochs 300 --batch-size 16
"""
from __future__ import annotations

import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from _common import ROOT, read_split
from emg2force import DEFAULT_MODEL, Emg2ForceDataModule, Emg2ForceModule, get_config, list_models
from emg2force.model import build_force_model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-split", default="train.txt")
    ap.add_argument("--val-split", default="val.txt")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--config", default=DEFAULT_MODEL,
                    help=f"named model config {list_models()} — sets the backbone")
    ap.add_argument("--backbone", default=None,
                    help="explicit timm spectrogram ViT id (overrides --config)")
    ap.add_argument("--window-length", type=int, default=10000)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "lightning_logs"))
    ap.add_argument("--name", default="emg2force_prevforce")
    args = ap.parse_args()

    pl.seed_everything(42, workers=True)
    train_sessions, val_sessions = read_split(args.train_split), read_split(args.val_split)
    print(f"train={len(train_sessions)} val={len(val_sessions)} sessions")

    datamodule = Emg2ForceDataModule(
        window_length=args.window_length, batch_size=args.batch_size, num_workers=args.num_workers,
        train_sessions=train_sessions, val_sessions=val_sessions, test_sessions=val_sessions,
    )
    # Initialise the spectrogram ViT backbone from public weights when training
    # from scratch. --backbone overrides the named --config.
    backbone = args.backbone or get_config(args.config).backbone
    print(f"backbone: {backbone}  (config: {args.config})")
    net = build_force_model(use_prev_force=True, backbone=backbone, pretrained_backbone=True)
    print(f"BasicTransformer: {sum(p.numel() for p in net.parameters()):,} params")
    model = Emg2ForceModule(net, lr=args.lr, use_scheduler=True,
                            warmup_epochs=10, max_epochs=args.epochs)

    callbacks = [
        ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1,
                        filename="best-{epoch:02d}-{val_loss:.2f}", save_last=True),
        ModelCheckpoint(filename="checkpoint-{epoch:02d}", every_n_epochs=10, save_top_k=-1),
        EarlyStopping(monitor="val_loss", mode="min", patience=50, verbose=True),
    ]
    trainer = pl.Trainer(
        max_epochs=args.epochs, callbacks=callbacks,
        logger=TensorBoardLogger(args.out, name=args.name),
        accelerator="gpu", devices=1, log_every_n_steps=10,
        num_sanity_val_steps=0, precision="32-true",
    )
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)


if __name__ == "__main__":
    main()
