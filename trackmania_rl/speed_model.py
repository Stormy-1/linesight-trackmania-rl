"""Learned speed model: track features -> the speed a pro carries at that point.

A small MLP (features standardised inside the model, so a saved file is
self-contained). Trained on the official replays of non-test maps; it replaces
the hand-set corner rule ``max_speed / (1 + gain * |curvature|)`` and the 70 m/s cap.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

HIDDEN_UNITS = 64
EPOCHS = 40
BATCH_SIZE = 1024
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 1e-4
HUBER_DELTA_MPS = 5.0


class SpeedModel(nn.Module):
    def __init__(
        self,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        feature_names: tuple[str, ...],
        target_mean: float = 0.0,
        target_std: float = 1.0,
    ):
        super().__init__()
        self.feature_names = tuple(feature_names)
        self.register_buffer("mean", torch.as_tensor(feature_mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(np.maximum(feature_std, 1e-6), dtype=torch.float32))
        self.register_buffer("target_mean", torch.tensor(float(target_mean)))
        self.register_buffer("target_std", torch.tensor(max(float(target_std), 1e-6)))
        n = len(feature_names)
        self.net = nn.Sequential(
            nn.Linear(n, HIDDEN_UNITS), nn.ReLU(), nn.Linear(HIDDEN_UNITS, HIDDEN_UNITS), nn.ReLU(), nn.Linear(HIDDEN_UNITS, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.target_mean + self.target_std * self.net((x - self.mean) / self.std).squeeze(-1)

    def predict(self, features: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self(torch.as_tensor(features, dtype=torch.float32)).numpy().astype(float)


def train_speed_model(
    features: np.ndarray, target: np.ndarray, feature_names: tuple[str, ...], seed: int = 0, epochs: int = EPOCHS
) -> SpeedModel:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SpeedModel(features.mean(axis=0), features.std(axis=0), feature_names, target.mean(), target.std())
    optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    loss_fn = nn.HuberLoss(delta=HUBER_DELTA_MPS / max(float(target.std()), 1e-6))  # loss on the standardised target
    x_all = torch.as_tensor(features, dtype=torch.float32)
    y_all = torch.as_tensor(target, dtype=torch.float32)
    for _ in range(epochs):
        order = torch.as_tensor(rng.permutation(len(target)))
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start : start + BATCH_SIZE]
            optimiser.zero_grad()
            scaled = (model(x_all[batch]) - model.target_mean) / model.target_std
            loss_fn(scaled, (y_all[batch] - model.target_mean) / model.target_std).backward()
            optimiser.step()
        schedule.step()
    return model.eval()


def save_speed_model(model: SpeedModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "feature_names": list(model.feature_names)}, path)


def load_speed_model(path: Path) -> SpeedModel:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    names = tuple(payload["feature_names"])
    model = SpeedModel(np.zeros(len(names)), np.ones(len(names)), names)
    model.load_state_dict(payload["state_dict"])
    return model.eval()


def learned_base_speed(geometry, model: SpeedModel, scale: float) -> np.ndarray:
    """Per-point target from the model, times a global safety scale (one value for every map)."""

    from trackmania_rl.speed_features import track_point_features

    if not 0.0 < scale <= 1.5:
        raise ValueError(f"speed scale {scale} out of range")
    return scale * model.predict(track_point_features(geometry))
