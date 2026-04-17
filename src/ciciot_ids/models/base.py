"""
Abstract base classes for all models (SOLID: O — open for extension,
D — high-level modules depend on these abstractions, not concrete classes).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseModel(ABC):
    """Minimal contract every model must fulfil."""

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> "BaseModel":
        """Train the model in-place and return self."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return integer class predictions."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist model artefact to *path*."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseModel":
        """Load a previously saved model from *path*."""


class BaseMLModel(BaseModel, ABC):
    """Extended contract for sklearn-compatible models (adds predict_proba)."""

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability matrix of shape (n_samples, n_classes)."""


class BaseDLModel(BaseModel, ABC):
    """Extended contract for PyTorch models."""

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return raw softmax / sigmoid probabilities."""

    @property
    @abstractmethod
    def num_params(self) -> int:
        """Total trainable parameter count."""
