"""Named model configurations — quick switch between released checkpoints.

The source of truth is ``configs/models.json`` at the repo root (edit it to add
your own fine-tuned checkpoints). A built-in copy is baked in as a fallback so
the package still works when installed without the json file.

Each entry maps a short name → checkpoint path + backbone timm id, e.g.::

    from emg2force import get_config, load_model
    cfg = get_config("dinov2")          # -> ModelConfig(checkpoint=..., backbone=...)
    model = load_model(config="dinov2")  # builds the DINOv2 model + loads its weights

The CLI scripts expose the same switch via ``--config dinov2``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# Repo root = parent of the package directory (emg2force/configs.py -> repo root).
_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_FILE = _ROOT / "configs" / "models.json"

# Fallback used if configs/models.json is missing or unreadable.
_BUILTIN = {
    "default": "dinov2",
    "models": {
        "dinov2": {
            "checkpoint": "checkpoints/emg2force_prevforce_dinov2.ckpt",
            "backbone": "vit_small_patch14_reg4_dinov2.lvd142m",
            "backbone_license": "Apache-2.0",
            "description": "Released model — DINOv2 ViT-S/14 (registers) backbone, Apache-2.0.",
        },
    },
}

_FIELDS = ("checkpoint", "backbone", "backbone_license", "description")


@dataclass(frozen=True)
class ModelConfig:
    name: str
    checkpoint: str
    backbone: str | None = None
    backbone_license: str = "unspecified"
    description: str = ""

    @property
    def checkpoint_path(self) -> Path:
        """Absolute checkpoint path (relative entries resolve against the repo root)."""
        p = Path(self.checkpoint)
        return p if p.is_absolute() else _ROOT / p


def _load_data() -> dict:
    if _CONFIG_FILE.exists():
        try:
            return json.loads(_CONFIG_FILE.read_text())
        except Exception as e:  # noqa: BLE001
            log.warning("Could not read %s (%s); using built-in defaults.", _CONFIG_FILE, e)
    return _BUILTIN


_DATA = _load_data()
DEFAULT_MODEL: str = _DATA.get("default", "dinov2")
MODELS: dict[str, ModelConfig] = {
    name: ModelConfig(name=name, **{k: spec[k] for k in _FIELDS if k in spec})
    for name, spec in _DATA.get("models", {}).items()
}


def list_models() -> list[str]:
    """Names of the available model configs."""
    return list(MODELS)


def get_config(name: str) -> ModelConfig:
    """Look up a model config by name (e.g. ``"dinov2"``)."""
    if name not in MODELS:
        raise KeyError(f"Unknown model config {name!r}. Available: {list_models()}")
    return MODELS[name]
