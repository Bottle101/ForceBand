"""Evaluation: load a checkpoint, score it on a session list, write metrics/plots.

Two regimes for the prev-force model:
  * ``teacher_forced`` — the dataset's ground-truth previous force at every
    window (optimistic anchor; matches training).
  * ``rollout``        — autoregressive: each session is processed window-by-
    window, feeding the model's own last prediction back as ``prev_force``
    (the realistic deployment number).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytorch_lightning as pl  # noqa: E402
import torch  # noqa: E402
import torch.utils.data  # noqa: E402
from scipy import stats as sp_stats  # noqa: E402

from emg2force.data import Emg2ForceDataModule  # noqa: E402
from emg2force.encoder import DEFAULT_DINO_MODEL  # noqa: E402
from emg2force.lightning import Emg2ForceModule  # noqa: E402
from emg2force.model import build_force_model  # noqa: E402

log = logging.getLogger(__name__)

FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]


@dataclass
class ValConfig:
    checkpoint: str
    out_dir: str
    test_sessions: list[Path]
    use_prev_force: bool = True
    eval_mode: str = "teacher_forced"            # "teacher_forced" | "rollout"
    # timm backbone id. None -> read from the checkpoint's config, else the default.
    backbone: str | None = None
    window_length: int = 10000
    batch_size: int = 8
    num_workers: int = 4
    limit_batches: int = 0                        # 0 = all
    max_points_per_image: int = 200_000
    # Optional per-session overlay video / raw dump (see emg2force.video). When
    # set, `video_root` is the release `data/videos` directory.
    video_root: Path | None = None
    render_overlay_video: bool = False
    save_raw_session_data: bool = False
    video_window_sec: float = 4.0
    video_max_seconds: float = 0.0
    video_session_limit: int = 0


# ── checkpoint ──────────────────────────────────────────────────────────────

def load_into_module(model: Emg2ForceModule, checkpoint: str) -> None:
    """Load a (clean or original) checkpoint's state_dict into the module."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        log.warning(f"Missing keys ({len(missing)}): {missing[:5]} ...")
    if unexpected:
        log.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")


# ── inference helpers ───────────────────────────────────────────────────────

def _move(x, device):
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: _move(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_move(v, device) for v in x)
    return x


def _to_Nx5(arr: torch.Tensor) -> torch.Tensor:
    if arr.ndim == 1:
        arr = arr.unsqueeze(0)
    if arr.shape[-1] != 5:
        ch = next(i for i, s in enumerate(arr.shape) if s == 5)
        perm = [d for d in range(arr.ndim) if d != ch] + [ch]
        arr = arr.permute(*perm).contiguous()
    return arr.reshape(-1, 5)


def _run_batches(model, loader, device, limit: int = 0):
    preds, gts = [], []
    for i, batch in enumerate(loader):
        if limit and i >= limit:
            break
        y_hat = model(_move(batch, device))
        pred, tgt = (y_hat[0], y_hat[1]) if isinstance(y_hat, tuple) else (y_hat, batch["target"])
        pred, tgt = _to_Nx5(pred.detach().cpu()), _to_Nx5(tgt.detach().cpu())
        n = min(pred.shape[0], tgt.shape[0])
        preds.append(pred[:n]); gts.append(tgt[:n])
    return torch.cat(preds).numpy(), torch.cat(gts).numpy()


@torch.no_grad()
def collect_predictions(model, datamodule, device, limit_batches: int = 0):
    model.eval().to(device)
    return _run_batches(model, datamodule.test_dataloader(), device, limit_batches)


@torch.no_grad()
def collect_predictions_per_session(model, datamodule, device, cfg: ValConfig):
    model.eval().to(device)
    results = []
    for ds in datamodule.test_dataset.datasets:
        loader = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                                             num_workers=cfg.num_workers, pin_memory=True)
        preds, gts = _run_batches(model, loader, device)
        real = sum(e - s for s, e in ds.windows)  # drop the zero-padded tail of short sessions
        results.append((Path(ds.hdf5_path).stem, Path(ds.hdf5_path), preds[:real], gts[:real]))
    return results


@torch.no_grad()
def collect_predictions_per_session_rollout(model, datamodule, device, cfg: ValConfig):
    """Autoregressive rollout: feed each window's last prediction as the next
    window's prev_force. Windows are contiguous (stride == window_length)."""
    model.eval().to(device)
    num_forces = getattr(model.model, "out_channels", 5)
    results = []
    for ds in datamodule.test_dataset.datasets:
        preds, gts = [], []
        prev = torch.zeros(1, num_forces, device=device)  # session starts at rest
        for idx in range(len(ds)):
            sample = ds[idx]
            batch = _move({
                "input": sample["input"].unsqueeze(0),
                "target": sample["target"].unsqueeze(0),
                "prev_force": prev,
            }, device)
            y_hat = model(batch)
            pred = y_hat[0] if isinstance(y_hat, tuple) else y_hat
            tgt = y_hat[1] if isinstance(y_hat, tuple) else batch["target"]
            prev = pred[:, :, -1].detach().reshape(1, num_forces)
            preds.append(_to_Nx5(pred.detach().cpu()))
            gts.append(_to_Nx5(tgt.detach().cpu()))
        real = sum(e - s for s, e in ds.windows)  # drop the zero-padded tail of short sessions
        results.append((Path(ds.hdf5_path).stem, Path(ds.hdf5_path),
                        torch.cat(preds).numpy()[:real], torch.cat(gts).numpy()[:real]))
    return results


# ── metrics ─────────────────────────────────────────────────────────────────

_METRIC_KEYS = ["mae", "mse", "rmse", "r2", "pearson_r", "nrmse", "error_std", "max_ae"]
_METRIC_HEADERS = ["MAE", "MSE", "RMSE", "R²", "Pearson r", "NRMSE", "Err STD", "Max AE"]


def filter_inactive(preds, gts, threshold: float = 20.0):
    """Keep only time steps where some GT force channel is >= threshold."""
    active = np.any(gts >= threshold, axis=1)
    return preds[active], gts[active]


def compute_metrics(preds: np.ndarray, gts: np.ndarray) -> dict:
    assert preds.shape == gts.shape and preds.shape[1] == 5
    eps = 1e-8
    diff = preds - gts
    mae = np.mean(np.abs(diff), axis=0)
    mse = np.mean(diff ** 2, axis=0)
    rmse = np.sqrt(mse)
    # R² and Pearson r are only meaningful where the ground truth actually varies.
    # A finger that is never actuated in a session (constant ≈ 0) has no variance,
    # so its R²/r are reported as NaN and excluded from the finger-mean — without
    # this guard a near-constant channel produces absurd (e.g. -1e15) values.
    gt_std = gts.std(axis=0)
    pred_std = preds.std(axis=0)
    active = gt_std > 1e-6
    ss_res = np.sum(diff ** 2, axis=0)
    ss_tot = np.sum((gts - gts.mean(axis=0, keepdims=True)) ** 2, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.where(active & (ss_tot > eps), 1.0 - ss_res / ss_tot, np.nan)
    pearson_r = np.full(5, np.nan)
    for c in range(5):
        if gt_std[c] > 1e-6 and pred_std[c] > 1e-6:
            pearson_r[c] = sp_stats.pearsonr(gts[:, c], preds[:, c])[0]
    rng = gts.max(axis=0) - gts.min(axis=0)
    nrmse = np.where(rng > eps, rmse / rng, np.nan)
    return dict(mae=mae, mse=mse, rmse=rmse, r2=r2, pearson_r=pearson_r,
                nrmse=nrmse, error_std=np.std(diff, axis=0), max_ae=np.max(np.abs(diff), axis=0))


def print_metrics_table(metrics: dict, title: str = "Overall Metrics"):
    fw, cw = 10, 12
    sep = "-" * (fw + len(_METRIC_KEYS) * cw + 4)
    log.info("\n%s\n%s", title, sep)
    log.info(f"{'Finger':<{fw}}" + "".join(f"{h:>{cw}}" for h in _METRIC_HEADERS))
    log.info(sep)
    for i, name in enumerate(FINGER_NAMES):
        log.info(f"{name:<{fw}}" + "".join(f"{metrics[k][i]:>{cw}.4f}" for k in _METRIC_KEYS))
    log.info(f"{'Mean':<{fw}}" + "".join(f"{np.nanmean(metrics[k]):>{cw}.4f}" for k in _METRIC_KEYS))
    log.info(sep)


def save_metrics_csv(metrics: dict, path: Path):
    lines = ["finger," + ",".join(_METRIC_KEYS)]
    for i, name in enumerate(FINGER_NAMES):
        lines.append(f"{name}," + ",".join(f"{metrics[k][i]:.6f}" for k in _METRIC_KEYS))
    lines.append("Mean," + ",".join(f"{np.nanmean(metrics[k]):.6f}" for k in _METRIC_KEYS))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    log.info("Saved metrics CSV: %s", path)


def save_per_session_csv(session_rows, path: Path):
    lines = ["session,finger," + ",".join(_METRIC_KEYS)]
    for name, sm in session_rows:
        for i, fname in enumerate(FINGER_NAMES):
            lines.append(f"{name},{fname}," + ",".join(f"{sm[k][i]:.6f}" for k in _METRIC_KEYS))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    log.info("Saved per-session CSV: %s", path)


# ── plots ───────────────────────────────────────────────────────────────────

def plot_pred_vs_gt(preds, gts, out_path, title_suffix=""):
    x = np.arange(len(preds))
    fig, axes = plt.subplots(5, 1, figsize=(30, 14), sharex=True)
    fig.suptitle(f"Prediction vs Ground Truth{title_suffix}", fontsize=14)
    for c in range(5):
        axes[c].plot(x, gts[:, c], label="GT", linewidth=2.5, color="C0")
        axes[c].plot(x, preds[:, c], label="Pred", linewidth=2.0, alpha=0.9, color="C1")
        axes[c].set_ylabel(FINGER_NAMES[c]); axes[c].grid(True, linestyle="--", alpha=0.3)
        axes[c].legend(loc="upper right", ncol=2, fontsize=9)
    axes[-1].set_xlabel("Sample index")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_scatter(preds, gts, out_path):
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle("Predicted vs Actual", fontsize=14)
    for c in range(5):
        n = len(preds)
        idx = np.random.choice(n, min(n, 20000), replace=False)
        axes[c].scatter(gts[idx, c], preds[idx, c], s=1, alpha=0.15, color=f"C{c}")
        lo = min(gts[:, c].min(), preds[:, c].min()); hi = max(gts[:, c].max(), preds[:, c].max())
        axes[c].plot([lo, hi], [lo, hi], "k--", linewidth=0.8)
        axes[c].set_title(FINGER_NAMES[c]); axes[c].set_xlabel("Actual")
        axes[c].set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("Predicted")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150); plt.close(fig)


# ── driver ──────────────────────────────────────────────────────────────────

def build_eval_model(cfg: ValConfig) -> Emg2ForceModule:
    log.info("Loading checkpoint: %s", cfg.checkpoint)
    ckpt = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    # Pick the backbone: explicit cfg > checkpoint config > default backbone.
    backbone = cfg.backbone
    if backbone is None and isinstance(ckpt, dict):
        backbone = ckpt.get("config", {}).get("backbone_timm_id")
    backbone = backbone or DEFAULT_DINO_MODEL
    log.info("Backbone: %s", backbone)
    net = build_force_model(use_prev_force=cfg.use_prev_force, backbone=backbone, pretrained_backbone=False)
    model = Emg2ForceModule(net)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        log.warning("Missing keys (%d): %s ...", len(missing), missing[:5])
    if unexpected:
        log.warning("Unexpected keys (%d): %s ...", len(unexpected), unexpected[:5])
    return model


def run_validation(cfg: ValConfig):
    logging.basicConfig(level=logging.INFO)
    pl.seed_everything(42, workers=True)

    datamodule = Emg2ForceDataModule(
        window_length=cfg.window_length, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
        train_sessions=[], val_sessions=cfg.test_sessions, test_sessions=cfg.test_sessions,
    )
    datamodule.setup("test")

    model = build_eval_model(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.out_dir)

    if cfg.use_prev_force and cfg.eval_mode == "rollout":
        log.info("Eval mode: autoregressive rollout")
        session_results = collect_predictions_per_session_rollout(model, datamodule, device, cfg)
    else:
        log.info("Eval mode: teacher-forced")
        session_results = collect_predictions_per_session(model, datamodule, device, cfg)
    # Headline metrics are the pooled per-session predictions (padding already
    # trimmed), so short sessions don't contribute fabricated zeros.
    preds = np.concatenate([s[2] for s in session_results])
    gts = np.concatenate([s[3] for s in session_results])
    log.info("Collected %d samples", preds.shape[0])

    metrics = compute_metrics(preds, gts)
    print_metrics_table(metrics, title="Overall Validation Metrics")
    save_metrics_csv(metrics, out_dir / "overall_metrics.csv")

    n = preds.shape[0]
    plot_pred_vs_gt(preds, gts, out_dir / "pred_vs_gt.png")
    parts = (n + cfg.max_points_per_image - 1) // cfg.max_points_per_image
    if parts > 1:
        for p in range(parts):
            s, e = p * cfg.max_points_per_image, min((p + 1) * cfg.max_points_per_image, n)
            plot_pred_vs_gt(preds[s:e], gts[s:e], out_dir / f"pred_vs_gt_part{p+1:02d}.png",
                            title_suffix=f" (part {p+1}/{parts})")
    plot_scatter(preds, gts, out_dir / "scatter.png")

    session_rows = []
    for name, _path, s_preds, s_gts in session_results:
        sm = compute_metrics(s_preds, s_gts)
        session_rows.append((name, sm))
        plot_pred_vs_gt(s_preds, s_gts, out_dir / "per_session" / f"{name}.png",
                        title_suffix=f" — {name}")
    save_per_session_csv(session_rows, out_dir / "per_session_metrics.csv")

    if cfg.render_overlay_video or cfg.save_raw_session_data:
        from emg2force import video
        video.export_session_media(session_results, cfg, out_dir)

    log.info("\nValidation complete → %s", out_dir)
