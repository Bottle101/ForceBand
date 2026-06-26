"""emg2force — fingertip force estimation from inertial sEMG.

Public API:
    load_model(config="dinov2"|..., device=...) -> ready-to-eval module
    MODELS / get_config / list_models           -> named model configs
    build_force_model(...)                   -> untrained ForcePredictionModule
    BasicTransformer, ForcePredictionModule, SpectroDINOEncoder
    Emg2ForceModule (Lightning), Emg2ForceDataModule, Emg2ForceDataset, Emg2ForceSessionData
    ValConfig, run_validation               -> evaluation driver
"""

from __future__ import annotations

import torch

from emg2force.configs import DEFAULT_MODEL, MODELS, ModelConfig, get_config, list_models
from emg2force.data import Emg2ForceDataModule, Emg2ForceDataset, Emg2ForceSessionData
from emg2force.encoder import SpectroDINOEncoder
from emg2force.lightning import Emg2ForceModule
from emg2force.model import BasicTransformer, ForcePredictionModule, build_force_model

__version__ = "0.1.0"

__all__ = [
    "load_model", "build_force_model",
    "MODELS", "DEFAULT_MODEL", "ModelConfig", "get_config", "list_models",
    "BasicTransformer", "ForcePredictionModule", "SpectroDINOEncoder",
    "Emg2ForceModule", "Emg2ForceDataModule", "Emg2ForceDataset", "Emg2ForceSessionData",
    "ValConfig", "run_validation",
]


def load_model(checkpoint: str | None = None, device: str | torch.device = "cpu",
               use_prev_force: bool = True, eval_mode: bool = True,
               backbone: str | None = None, config: str | None = None) -> Emg2ForceModule:
    """Build the released model and load a checkpoint's weights.

    Returns an ``Emg2ForceModule`` whose ``forward(batch)`` takes a dict with
    ``input`` [B,18,T] (and optional ``prev_force`` [B,5]) and returns
    ``(pred, target, mask)``. The ViT backbone is built untrained and then
    overwritten by the checkpoint, so no backbone download is needed.

    Selecting the model (in priority order):
      * ``config="dinov2"`` — a named config (``configs/models.json``),
        resolving both the checkpoint and the backbone.
      * ``checkpoint=<path>`` — an explicit checkpoint; the backbone is read from
        the checkpoint's own ``config`` block.
      * neither — the default config (``DEFAULT_MODEL``).
    An explicit ``backbone`` always overrides.
    """
    from emg2force.encoder import DEFAULT_DINO_MODEL
    if config is None and checkpoint is None:
        config = DEFAULT_MODEL
    if config is not None:
        c = get_config(config)
        checkpoint = checkpoint or str(c.checkpoint_path)
        backbone = backbone or c.backbone
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if backbone is None and isinstance(ckpt, dict):
        backbone = ckpt.get("config", {}).get("backbone_timm_id")
    net = build_force_model(use_prev_force=use_prev_force,
                            backbone=backbone or DEFAULT_DINO_MODEL,
                            pretrained_backbone=False)
    module = Emg2ForceModule(net)
    missing, unexpected = module.load_state_dict(state, strict=False)
    if missing:
        import warnings
        warnings.warn(f"load_model: {len(missing)} missing keys (e.g. {missing[:3]})")
    if unexpected:
        import warnings
        warnings.warn(f"load_model: {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")
    module.to(device)
    if eval_mode:
        module.eval()
    return module


def __getattr__(name: str):  # lazy heavy imports (matplotlib/scipy) only when used
    if name in ("ValConfig", "run_validation"):
        from emg2force import eval as _eval
        return getattr(_eval, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
