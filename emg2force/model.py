"""Temporal force-prediction model (DINO spectrogram encoder + Transformer).

``BasicTransformer`` is the released architecture: it fuses a DINO EMG
spectrogram embedding with the raw 18-channel EMG+IMU stream, runs a small
Transformer encoder, and regresses 5 fingertip forces. It optionally conditions
on the previous window's last force (``use_prev_force``) so the trajectory stays
continuous across windows.

``ForcePredictionModule`` is a thin wrapper that adapts the batch dict to the
network and resamples the prediction to the input length.

Module/attribute names are kept identical to the network that produced the
released checkpoint, so its ``state_dict`` loads without remapping.
"""

from __future__ import annotations

import inspect

import torch
import torch.nn.functional as F
from torch import nn

from emg2force.encoder import DEFAULT_DINO_MODEL, SpectroDINOEncoder


class Conv1dBlock(nn.Module):
    """1D conv → ReLU → dropout, followed by a channel LayerNorm.

    The convolution uses no padding, so the sequence shortens by
    ``kernel_size - 1`` (before striding); this matches the released model.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int, dropout: float = 0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        # LayerNorm over the channel axis (BCT -> normalise C).
        return self.norm(x.swapaxes(-1, -2)).swapaxes(-1, -2)


class BasicTransformer(nn.Module):
    """DINO spectrogram encoder + conv subsampler + Transformer + linear head.

    Args:
        d_model, n_layers, n_heads: Transformer encoder size.
        num_forces: number of output force channels (5 fingertips).
        use_prev_force: condition each window on the previous window's last force.
        prev_force_noise: train-time Gaussian noise added to ``prev_force`` to
            reduce exposure bias when rolled out autoregressively.
        out_len: internal sequence length the head is resampled to before the
            wrapper resamples to the true input length (kept at 2000 to match
            the released checkpoint).
        pretrained_backbone / freeze_backbone: forwarded to the encoder.
    """

    def __init__(self, d_model: int = 256, n_layers: int = 4, n_heads: int = 8,
                 num_forces: int = 5, use_prev_force: bool = False,
                 prev_force_noise: float = 0.0, out_len: int = 2000,
                 backbone: str = DEFAULT_DINO_MODEL,
                 pretrained_backbone: bool = False, freeze_backbone: bool = False):
        super().__init__()
        self.use_prev_force = use_prev_force
        self.num_forces = num_forces
        self.prev_force_noise = prev_force_noise
        self.out_len = out_len

        self.spectro_encoder = SpectroDINOEncoder(
            sample_rate=1000, n_fft=512, hop_length=10, win_length=512,
            dino_model_name=backbone,
            pretrained_backbone=pretrained_backbone, freeze_backbone=freeze_backbone,
        )
        enc_dim = self.spectro_encoder.num_features  # 384 for ViT-S

        self.conv_blocks = nn.Sequential(
            Conv1dBlock(in_channels=enc_dim + 18, out_channels=512, kernel_size=5, stride=2),
            Conv1dBlock(in_channels=512, out_channels=d_model, kernel_size=5, stride=2),
        )

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, num_forces)

        if self.use_prev_force:
            # Project the previous force and add it as a global bias to every
            # transformer token, anchoring the window to where the last one ended.
            self.prev_force_proj = nn.Sequential(
                nn.Linear(num_forces, d_model),
                nn.LayerNorm(d_model),
            )

    def forward(self, x: torch.Tensor, prev_force: torch.Tensor | None = None) -> torch.Tensor:
        x_emg = self.spectro_encoder(x[:, :8, :])     # [B, enc_dim, T]
        x = torch.cat([x_emg, x], dim=1)              # [B, enc_dim+18, T]

        x = self.conv_blocks(x)                        # [B, d_model, T_sub]
        x = x.transpose(1, 2)                          # [B, T_sub, d_model]

        if self.use_prev_force:
            if prev_force is None:
                prev_force = x.new_zeros(x.size(0), self.num_forces)
            prev_force = prev_force.to(dtype=x.dtype)
            if self.training and self.prev_force_noise > 0:
                prev_force = prev_force + self.prev_force_noise * torch.randn_like(prev_force)
            x = x + self.prev_force_proj(prev_force).unsqueeze(1)  # broadcast over time

        x = self.encoder(x)                            # [B, T_sub, d_model]
        y = self.head(x).transpose(1, 2)               # [B, num_forces, T_sub]
        return F.interpolate(y, size=self.out_len, mode="linear", align_corners=False)


class ForcePredictionModule(nn.Module):
    """Adapt the batch dict to ``network`` and resample to the input length.

    forward(batch) -> (pred, target, mask) where:
        pred   : [B, out_channels, T] forces at the input sample rate
        target : [B, out_channels, T] ground-truth forces (None if not provided)
        mask   : [B, T] all-ones (kept for API compatibility)

    ``batch["target"]`` is optional, so the same call works for pure inference.
    """

    def __init__(self, network: nn.Module, out_channels: int = 5):
        super().__init__()
        self.network = network
        self.out_channels = out_channels
        # Only feed prev_force to networks whose forward() accepts it.
        self._net_accepts_prev_force = (
            "prev_force" in inspect.signature(self.network.forward).parameters
        )

    def forward(self, batch: dict[str, torch.Tensor], provide_initial_pos: bool = False):
        emg_imu = batch["input"]              # [B, 18, T]
        target = batch.get("target", None)    # [B, 5, T] or None (inference)

        if self._net_accepts_prev_force:
            pred = self.network(emg_imu, prev_force=batch.get("prev_force", None))
        else:
            pred = self.network(emg_imu)

        if pred.shape[-1] != emg_imu.shape[-1]:
            pred = F.interpolate(pred, size=emg_imu.shape[-1], mode="linear", align_corners=False)

        mask = torch.ones(emg_imu.shape[0], emg_imu.shape[2], dtype=torch.bool, device=emg_imu.device)
        return pred, target, mask


def build_force_model(use_prev_force: bool = True, out_channels: int = 5,
                      backbone: str = DEFAULT_DINO_MODEL,
                      pretrained_backbone: bool = False, freeze_backbone: bool = False,
                      **net_kwargs) -> ForcePredictionModule:
    """Construct the released model: BasicTransformer wrapped for force output.

    ``backbone`` selects the timm spectrogram ViT (default: the Apache-2.0
    DINOv2 ViT-S/14, which matches the released checkpoint).
    """
    net = BasicTransformer(
        use_prev_force=use_prev_force, num_forces=out_channels, backbone=backbone,
        pretrained_backbone=pretrained_backbone, freeze_backbone=freeze_backbone,
        **net_kwargs,
    )
    return ForcePredictionModule(net, out_channels=out_channels)
