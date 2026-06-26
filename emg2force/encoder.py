"""EMG spectrogram encoder backed by a DINO ViT.

The 8-channel EMG window is turned into a per-channel magnitude spectrogram,
projected from 8 → 3 channels, resized to the ViT patch grid, and passed
through a ``timm`` DINO ViT-S backbone (DINOv2 ViT-S/14 by default). The
patch-token sequence is then linearly upsampled back to the EMG sample rate so
it can be concatenated with the raw EMG+IMU stream before the temporal model.

This is a self-contained reimplementation with no external research-code
dependencies. The forward pass is kept numerically identical to the network that
produced the released checkpoint, so the ``backbone.*`` / ``proj8to3.*`` weights
load directly.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError as e:  # pragma: no cover
    raise ImportError("The DINO backbone requires `timm` — `pip install timm`.") from e

#: Default timm backbone — DINOv2 ViT-S/14 with registers (Apache-2.0; num_features = 384).
DEFAULT_DINO_MODEL = "vit_small_patch14_reg4_dinov2.lvd142m"
DINOV2_VITS14 = "vit_small_patch14_reg4_dinov2.lvd142m"
#: DINOv3 ViT-S/16 (num_features = 384). NOT shipped — custom (non-Apache) license; see the README roadmap.
DINOV3_VITS16 = "vit_small_patch16_dinov3.lvd1689m"


class SpectroDINOEncoder(nn.Module):
    """Map an 8-channel EMG window to a per-timestep feature sequence.

    Args:
        sample_rate: EMG sample rate in Hz (informational; STFT is in samples).
        n_fft, hop_length, win_length: STFT parameters.
        dino_model_name: timm backbone id.
        pretrained_backbone: if True, initialise the ViT from the public backbone
            weights (used when training from scratch). Set False when you are
            about to load the released checkpoint — the architecture is identical
            either way, and this avoids the extra download.
        freeze_backbone: if True, the backbone parameters do not require grad.

    Input:  ``x_emg`` of shape ``[B, 8, T]`` (or ``[B, T, 8]``).
    Output: ``[B, num_features, T]`` features at the EMG sample rate.
    """

    def __init__(
        self,
        sample_rate: int = 1000,
        n_fft: int = 512,
        hop_length: int = 10,
        win_length: int = 512,
        dino_model_name: str = DEFAULT_DINO_MODEL,
        pretrained_backbone: bool = False,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        # STFT analysis window (not part of the state_dict).
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

        # 1x1 projection 8 EMG channels -> 3, so the pretrained 3-channel ViT
        # patch-embed can be used unchanged.
        self.proj8to3 = nn.Conv2d(8, 3, kernel_size=1, bias=False)

        # ``dynamic_img_size`` lets the ViT accept the (non-square, non-default)
        # spectrogram grid and interpolates positional embeddings accordingly.
        # DINOv2's PatchEmbed otherwise asserts a fixed 518x518 input. Some
        # backbones (e.g. some EVA-style impls) don't take the kwarg — fall back.
        try:
            self.backbone = timm.create_model(
                dino_model_name, pretrained=pretrained_backbone, num_classes=0,
                dynamic_img_size=True,
            )
        except TypeError:
            self.backbone = timm.create_model(
                dino_model_name, pretrained=pretrained_backbone, num_classes=0,
            )
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Patch size, used to pad the spectrogram up to a whole number of patches.
        patch = self.backbone.patch_embed.patch_size
        self.patch_h, self.patch_w = patch if isinstance(patch, tuple) else (patch, patch)

    @property
    def num_features(self) -> int:
        return self.backbone.num_features

    def forward(self, x_emg: torch.Tensor) -> torch.Tensor:
        # Accept [B, 8, T] or [B, T, 8]; canonicalise to [B, T, 8].
        if not (x_emg.dim() == 3 and x_emg.size(2) == 8):
            x_emg = x_emg.transpose(1, 2)
        B, T, C = x_emg.shape

        # Per-channel magnitude spectrogram, vectorised over batch*channel.
        x = x_emg.transpose(1, 2).reshape(B * C, T)  # [B*8, T]
        X = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(x.device),
            return_complex=True,
            center=True,
            pad_mode="reflect",
        )  # [B*8, F, frames]
        spec = torch.log1p(X.abs())

        # Per-(channel,sample) standardisation.
        spec = spec - spec.mean(dim=(1, 2), keepdim=True)
        spec = spec / (spec.std(dim=(1, 2), keepdim=True) + 1e-6)

        f_bins, frames = spec.size(1), spec.size(2)
        spec = spec.view(B, C, f_bins, frames)

        # Pad (freq, time) up to whole patches so the ViT grid is exact.
        h_needed = math.ceil(f_bins / self.patch_h) * self.patch_h
        w_needed = math.ceil(frames / self.patch_w) * self.patch_w
        spec = F.interpolate(spec, size=(h_needed, w_needed), mode="bilinear", align_corners=False)

        x3 = self.proj8to3(spec)                  # [B, 3, H, W]
        feats = self.backbone.forward_features(x3)  # [B, N_tokens, D]
        # Treat the token axis as "time" and upsample back to the EMG grid.
        return self._upsample_to_emg_rate(feats.permute(0, 2, 1), target_t=T)  # [B, D, T]

    @torch.no_grad()
    def _upsample_to_emg_rate(self, seq: torch.Tensor, target_t: int, mode: str = "linear") -> torch.Tensor:
        """Linearly resample the token sequence to ``target_t`` steps.

        NOTE: wrapped in ``no_grad`` to exactly match the released model. A side
        effect is that no gradient flows back into the backbone during training,
        i.e. the backbone weights are effectively frozen regardless of
        ``freeze_backbone``. Kept intentionally for faithful reproduction.
        """
        return F.interpolate(seq, size=target_t, mode=mode, align_corners=False)
