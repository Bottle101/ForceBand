"""Data loading for EMG+IMU → fingertip-force prediction.

Three layers:
  * ``Emg2ForceSessionData`` — reads one processed HDF5 session.
  * ``Emg2ForceDataset``     — windows a session into fixed-length samples.
  * ``Emg2ForceDataModule``  — a Lightning datamodule over many sessions.

Self-contained: only ``h5py``/``numpy``/``torch``/``pytorch_lightning``.

Expected HDF5 layout (1 kHz, upsampled), under group ``forceband/timeseries``:
  - ``timestamps`` (N,)      float64, seconds
  - ``emg``        (N, 8)    float32   (if stored with >8 columns, cols 1:9 are used)
  - ``imu``        (N, 10)   float32   (quaternion[4], gyro[3], acc[3])
  - ``force``      (N, 5)    float32   (thumb, index, middle, ring, pinky)
A legacy compound-dataset layout (fields ``time/emg/imu/force``) is also read.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import ConcatDataset, DataLoader

log = logging.getLogger(__name__)


@dataclass
class Emg2ForceSessionData:
    """Lazy reader for one processed HDF5 session."""

    HDF5_GROUP = "forceband"
    TIMESERIES = "timeseries"
    EMG = "emg"
    IMU = "imu"
    FORCE = "force"
    TIMESTAMPS = "timestamps"

    hdf5_path: Path

    def __post_init__(self) -> None:
        self._file = h5py.File(self.hdf5_path, "r")
        group = self._file[self.HDF5_GROUP]
        ts = group[self.TIMESERIES]

        if isinstance(ts, h5py.Dataset) and ts.dtype.fields:
            # Legacy compound dataset with named fields.
            self.TIMESTAMPS = "time"
            for field in (self.TIMESTAMPS, self.EMG, self.IMU, self.FORCE):
                assert field in ts.dtype.fields, f"Missing '{field}' field in {self.hdf5_path}"
            self.timeseries = ts
        else:
            # Group of separate datasets — load into a structured array.
            for key in (self.TIMESTAMPS, self.EMG, self.IMU, self.FORCE):
                assert key in ts.keys(), f"Missing '{key}' dataset in {self.hdf5_path}"
            n = len(ts[self.TIMESTAMPS])
            dtype = np.dtype([
                ("timestamps", np.float64),
                ("emg", np.float32, (8,)),
                ("imu", np.float32, (10,)),
                ("force", np.float32, (5,)),
            ])
            data = np.empty(n, dtype=dtype)
            data["timestamps"] = ts[self.TIMESTAMPS]
            emg = ts[self.EMG]
            data["emg"] = emg[:, 1:9] if emg.shape[1] > 8 else emg
            data["imu"] = ts[self.IMU]
            data["force"] = ts[self.FORCE]
            self.timeseries = data

    def __len__(self) -> int:
        return len(self.timeseries)

    def get_data_range(self, start: int, end: int) -> np.ndarray:
        return self.timeseries[start:end]

    def close(self) -> None:
        if getattr(self, "_file", None) is not None:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class Emg2ForceDataset(torch.utils.data.Dataset):
    """Slice a session into fixed-length windows.

    Each item is a dict with ``input`` [18, T], ``target`` [5, T] and
    ``prev_force`` [5] (the force sample immediately before the window; zeros at
    the start of a session). Windows shorter than ``window_length`` are
    zero-padded at the end.
    """

    def __init__(self, hdf5_path: Path, window_length: int = 10000, stride: int | None = None,
                 allowed_intervals: list[tuple[int, int]] | None = None):
        self.hdf5_path = hdf5_path
        self.window_length = window_length
        self.stride = stride if stride is not None else window_length
        self.allowed_intervals = allowed_intervals

        self.session = Emg2ForceSessionData(hdf5_path)
        self.session_length = len(self.session)
        self.windows = self._calculate_windows()

    def _calculate_windows(self) -> list[tuple[int, int]]:
        intervals = self.allowed_intervals or [(0, self.session_length)]
        windows: list[tuple[int, int]] = []
        for iv_start, iv_end in intervals:
            iv_start = max(0, iv_start)
            iv_end = min(self.session_length, iv_end)
            last_start = iv_end - self.window_length
            if last_start < iv_start:
                if iv_end > iv_start:  # shorter than one window — keep, zero-padded
                    windows.append((iv_start, iv_end))
                continue
            windows.extend(
                (s, s + self.window_length) for s in range(iv_start, last_start + 1, self.stride)
            )
        return windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start, end = self.windows[idx]
        window = self.session.get_data_range(start, end)
        emg = window[self.session.EMG]      # (T, 8)
        imu = window[self.session.IMU]      # (T, 10)
        force = window[self.session.FORCE]  # (T, 5)

        n_force = force.shape[1]
        if start > 0:
            prev = self.session.get_data_range(start - 1, start)[self.session.FORCE]
            prev_force = np.asarray(prev, dtype=np.float32).reshape(n_force)
        else:
            prev_force = np.zeros(n_force, dtype=np.float32)

        pad = self.window_length - emg.shape[0]
        if pad > 0:
            emg = np.pad(emg, ((0, pad), (0, 0)))
            imu = np.pad(imu, ((0, pad), (0, 0)))
            force = np.pad(force, ((0, pad), (0, 0)))

        inp = np.concatenate([emg, imu], axis=1)  # (T, 18)
        return {
            "input": torch.as_tensor(inp.copy(), dtype=torch.float32).T,    # (18, T)
            "target": torch.as_tensor(force.copy(), dtype=torch.float32).T,  # (5, T)
            "prev_force": torch.as_tensor(prev_force, dtype=torch.float32),  # (5,)
            "window_start_idx": start,
            "window_end_idx": end,
        }

    def __del__(self):
        if hasattr(self, "session"):
            self.session.close()


class Emg2ForceDataModule(pl.LightningDataModule):
    """Lightning datamodule over lists of session HDF5 paths.

    Train windows use a 1/4-window stride (overlapping); val/test use
    non-overlapping windows. Set ``train_minute_bins`` to randomly subsample
    each training session to N one-minute bins.
    """

    def __init__(self, window_length: int, batch_size: int, num_workers: int,
                 train_sessions: Sequence[Path], val_sessions: Sequence[Path],
                 test_sessions: Sequence[Path], val_test_window_length: int | None = None,
                 train_minute_bins: int | None = None, bin_seconds: float = 60.0,
                 sample_rate: int = 1000, bin_seed: int = 42):
        super().__init__()
        self.window_length = window_length
        self.val_test_window_length = val_test_window_length or window_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_sessions = list(train_sessions)
        self.val_sessions = list(val_sessions)
        self.test_sessions = list(test_sessions)
        self.train_minute_bins = train_minute_bins
        self.bin_seconds = bin_seconds
        self.sample_rate = sample_rate
        self.bin_seed = bin_seed
        self.train_dataset = self.val_dataset = self.test_dataset = None

    def _pick_train_intervals(self, session_path: Path) -> list[tuple[int, int]] | None:
        if self.train_minute_bins is None:
            return None
        import random
        with h5py.File(session_path, "r") as f:
            ts = f["forceband"]["timeseries"]
            n = len(ts["timestamps"]) if isinstance(ts, h5py.Group) else len(ts)
        bin_len = int(round(self.bin_seconds * self.sample_rate))
        total = n // bin_len
        if total <= 0:
            raise ValueError(f"{Path(session_path).name} shorter than one bin ({n} < {bin_len}).")
        k = min(self.train_minute_bins, total)
        rng = random.Random(self.bin_seed + hash(str(session_path)) % (2**31))
        ids = sorted(rng.sample(range(total), k))
        return [(b * bin_len, (b + 1) * bin_len) for b in ids]

    def _make(self, sessions, window_length, stride=None, minute_bins=False) -> ConcatDataset:
        datasets = [
            Emg2ForceDataset(
                hdf5_path=p, window_length=window_length, stride=stride,
                allowed_intervals=self._pick_train_intervals(p) if minute_bins else None,
            )
            for p in sessions
        ]
        return ConcatDataset(datasets)

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = self._make(
                self.train_sessions, self.window_length,
                stride=self.window_length // 4, minute_bins=True,
            )
            self.val_dataset = self._make(self.val_sessions, self.val_test_window_length)
            if self.test_sessions:
                self.test_dataset = self._make(self.test_sessions, self.val_test_window_length)
        elif stage == "test":
            if self.test_dataset is None:
                self.test_dataset = self._make(self.test_sessions, self.val_test_window_length)

    def _loader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle,
                          num_workers=self.num_workers, pin_memory=True)

    def train_dataloader(self):
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._loader(self.test_dataset, shuffle=False)
