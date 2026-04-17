"""
PyTorch neural network architectures and BaseDLModel wrappers.

Architectures
-------------
MLPNet   — Multi-layer perceptron (works for both binary and multiclass).
CNN1DNet — 1D convolutional network (treats features as a 1-D sequence).

Wrappers (implement BaseDLModel so trainers can use either interchangeably)
---------------------------------------------------------------------------
MLPModel
CNN1DModel
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ciciot_ids.models.base import BaseDLModel

logger = logging.getLogger(__name__)


# ============================================================================
# PyTorch architectures
# ============================================================================

def _torch():
    """Lazy import torch to avoid hard dependency when not needed."""
    try:
        import torch
        return torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for DL models. Install with: pip install torch"
        ) from exc


def _nn():
    return _torch().nn


class MLPNet(_nn().__class__):  # type: ignore[misc]
    """
    MLP: Input → 256 → 128 → 64 → num_classes
    BatchNorm + Dropout after each hidden layer.

    For binary (num_classes=1) use BCEWithLogitsLoss.
    For multiclass (num_classes>1) use CrossEntropyLoss.
    """

    def __new__(cls, *args, **kwargs):
        # Deferred import so class body can reference nn.Module
        import torch.nn as nn

        class _MLPNet(nn.Module):
            def __init__(self, input_dim: int = 39, num_classes: int = 8, dropout: float = 0.3):
                super().__init__()
                self.net = nn.Sequential(
                    nn.BatchNorm1d(input_dim),
                    nn.Linear(input_dim, 256),
                    nn.ReLU(),
                    nn.BatchNorm1d(256),
                    nn.Dropout(dropout),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.BatchNorm1d(128),
                    nn.Dropout(dropout),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.BatchNorm1d(64),
                    nn.Dropout(dropout),
                    nn.Linear(64, num_classes),
                )

            def forward(self, x):
                return self.net(x)

        # Store on module-level so it's accessible
        global MLPNet
        MLPNet = _MLPNet
        return _MLPNet(*args, **kwargs)


class CNN1DNet(_nn().__class__):  # type: ignore[misc]
    """
    1D CNN: treats the feature vector as a 1-D sequence.
    Conv1D(1→64→128) + GlobalAvgPool + 128 → 64 → num_classes.
    """

    def __new__(cls, *args, **kwargs):
        import torch.nn as nn

        class _CNN1DNet(nn.Module):
            def __init__(self, input_dim: int = 39, num_classes: int = 8, dropout: float = 0.3):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.BatchNorm1d(64),
                    nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.BatchNorm1d(128),
                    nn.AdaptiveAvgPool1d(1),
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(64, num_classes),
                )

            def forward(self, x):
                x = x.unsqueeze(1)   # (B, input_dim) → (B, 1, input_dim)
                return self.classifier(self.conv(x))

        global CNN1DNet
        CNN1DNet = _CNN1DNet
        return _CNN1DNet(*args, **kwargs)


# ============================================================================
# BaseDLModel wrappers
# ============================================================================

class _DLModelBase(BaseDLModel):
    """
    Shared logic for MLP and CNN wrappers.
    Concrete classes only need to supply _build_net().
    """

    def __init__(
        self,
        input_dim: int = 39,
        num_classes: int = 8,
        dropout: float = 0.3,
    ) -> None:
        import torch

        self._input_dim = input_dim
        self._num_classes = num_classes
        self._dropout = dropout
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._net = self._build_net().to(self._device)

    def _build_net(self):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # BaseDLModel interface
    # ------------------------------------------------------------------

    def fit(self, X_train, y_train, **kwargs):
        raise NotImplementedError(
            "Use DLTrainer.train() instead of calling .fit() directly."
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch

        self._net.eval()
        with torch.no_grad():
            logits = self._net(
                torch.tensor(X, dtype=torch.float32).to(self._device)
            )
            if self._num_classes == 1:
                preds = (torch.sigmoid(logits).squeeze() > 0.5).long()
            else:
                preds = logits.argmax(dim=1)
        return preds.cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch

        self._net.eval()
        with torch.no_grad():
            logits = self._net(
                torch.tensor(X, dtype=torch.float32).to(self._device)
            )
            if self._num_classes == 1:
                prob = torch.sigmoid(logits).squeeze().cpu().numpy()
                return np.stack([1 - prob, prob], axis=-1)
            else:
                return torch.softmax(logits, dim=1).cpu().numpy()

    def save(self, path: Path) -> None:
        import torch

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self._net.state_dict(),
                "input_dim": self._input_dim,
                "num_classes": self._num_classes,
                "dropout": self._dropout,
            },
            path,
        )
        logger.info("%s saved → %s", type(self).__name__, path)

    @classmethod
    def load(cls, path: Path) -> "_DLModelBase":
        import torch

        ckpt = torch.load(path, map_location="cpu")
        obj = cls(
            input_dim=ckpt["input_dim"],
            num_classes=ckpt["num_classes"],
            dropout=ckpt["dropout"],
        )
        obj._net.load_state_dict(ckpt["state_dict"])
        logger.info("%s loaded ← %s", cls.__name__, path)
        return obj

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self._net.parameters() if p.requires_grad)

    @property
    def net(self):
        """Expose underlying nn.Module for the trainer."""
        return self._net

    @property
    def device(self):
        return self._device


class MLPModel(_DLModelBase):
    """MLP wrapper implementing BaseDLModel."""

    def _build_net(self):
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self, input_dim, num_classes, dropout):
                super().__init__()
                self.net = nn.Sequential(
                    nn.BatchNorm1d(input_dim),
                    nn.Linear(input_dim, 256), nn.ReLU(),
                    nn.BatchNorm1d(256), nn.Dropout(dropout),
                    nn.Linear(256, 128), nn.ReLU(),
                    nn.BatchNorm1d(128), nn.Dropout(dropout),
                    nn.Linear(128, 64), nn.ReLU(),
                    nn.BatchNorm1d(64), nn.Dropout(dropout),
                    nn.Linear(64, num_classes),
                )

            def forward(self, x):
                return self.net(x)

        return Net(self._input_dim, self._num_classes, self._dropout)


class CNN1DModel(_DLModelBase):
    """1D CNN wrapper implementing BaseDLModel."""

    def _build_net(self):
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self, input_dim, num_classes, dropout):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv1d(1, 64, kernel_size=3, padding=1),
                    nn.ReLU(), nn.BatchNorm1d(64),
                    nn.Conv1d(64, 128, kernel_size=3, padding=1),
                    nn.ReLU(), nn.BatchNorm1d(128),
                    nn.AdaptiveAvgPool1d(1),
                )
                self.head = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(128, 64), nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(64, num_classes),
                )

            def forward(self, x):
                return self.head(self.conv(x.unsqueeze(1)))

        return Net(self._input_dim, self._num_classes, self._dropout)
