"""PyTorch-Lightning module wrapping the force-prediction network.

Keeps the training recipe that produced the released checkpoint:
loss = ``MAE + 0.01 * MSE`` on the 5 forces, AdamW (lr, weight_decay=1e-2),
optional linear-warmup → cosine-decay schedule.

The wrapped ``ForcePredictionModule`` is stored as ``self.model`` so its
parameters appear under ``model.network.*`` — matching the released
``state_dict``. The network object is **not** saved into the checkpoint
hyperparameters, so re-trained checkpoints stay free of pickled module classes.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import pytorch_lightning as pl
import torch
from torch import nn

log = logging.getLogger(__name__)


class Emg2ForceModule(pl.LightningModule):
    def __init__(self, network: nn.Module, lr: float = 1e-4, weight_decay: float = 1e-2,
                 use_scheduler: bool = False, warmup_epochs: int = 10,
                 max_epochs: int = 300, min_lr: float = 1e-6):
        super().__init__()
        self.save_hyperparameters(ignore=["network"])
        self.model = network
        self.mae_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def forward(self, batch: Mapping[str, torch.Tensor]):
        return self.model(batch, provide_initial_pos=False)

    def _step(self, batch: Mapping[str, torch.Tensor], stage: str) -> torch.Tensor:
        preds, targets, _ = self.forward(batch)
        assert torch.isfinite(preds).all(), "NaN/Inf in prediction"
        assert torch.isfinite(targets).all(), "NaN/Inf in target"
        mae = self.mae_loss(preds, targets)
        mse = self.mse_loss(preds, targets)
        loss = mae + 0.01 * mse
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_mae", mae, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_mse", mse, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        if not self.hparams.use_scheduler:
            return opt
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        warmup = LinearLR(opt, start_factor=1e-2, end_factor=1.0,
                          total_iters=self.hparams.warmup_epochs)
        cosine = CosineAnnealingLR(opt, T_max=max(1, self.hparams.max_epochs - self.hparams.warmup_epochs),
                                   eta_min=self.hparams.min_lr)
        sched = SequentialLR(opt, schedulers=[warmup, cosine],
                             milestones=[self.hparams.warmup_epochs])
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
