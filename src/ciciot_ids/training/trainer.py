"""
Training orchestration (Single Responsibility: orchestrate training only,
not data loading or evaluation).

MLTrainer  — fits any BaseMLModel, returns timing info.
DLTrainer  — runs a full PyTorch epoch loop with val-F1 early stopping.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from ciciot_ids.models.base import BaseMLModel, BaseDLModel

logger = logging.getLogger(__name__)


class MLTrainer:
    """
    Trains a BaseMLModel-compatible sklearn model.

    Parameters
    ----------
    eval_set_on_fit : bool
        If True and the model's fit() accepts X_val/y_val, pass them.
        (Used by XGBoostModel for early stopping eval set.)
    """

    def __init__(self, eval_set_on_fit: bool = True) -> None:
        self._eval_set_on_fit = eval_set_on_fit

    def train(
        self,
        model: BaseMLModel,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> tuple[BaseMLModel, float]:
        """
        Fit *model* and return (model, elapsed_seconds).
        """
        logger.info("Starting ML training: %s", type(model).__name__)
        t0 = time.time()

        if self._eval_set_on_fit and X_val is not None:
            model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        else:
            model.fit(X_train, y_train)

        elapsed = time.time() - t0
        logger.info(
            "%s training complete in %.1fs", type(model).__name__, elapsed
        )
        return model, elapsed


class DLTrainer:
    """
    PyTorch training loop with:
    - Per-epoch train loss + val F1-macro logging
    - Best-weights checkpointing (by val F1-macro)
    - ReduceLROnPlateau scheduler
    - Optional model save path

    Parameters
    ----------
    epochs : int
    learning_rate : float
    batch_size : int
    patience : int
        LR patience for ReduceLROnPlateau.
    task : {'binary', 'multiclass'}
        Determines loss function.
    checkpoint_path : Path | None
        If provided, best weights are saved here after training.
    """

    def __init__(
        self,
        epochs: int = 30,
        learning_rate: float = 1e-3,
        batch_size: int = 2048,
        patience: int = 3,
        task: str = "multiclass",
        checkpoint_path: Path | None = None,
    ) -> None:
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.patience = patience
        self.task = task
        self.checkpoint_path = checkpoint_path

    def train(
        self,
        model: BaseDLModel,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> tuple[BaseDLModel, dict[str, list[float]], float]:
        """
        Train *model* for self.epochs epochs.

        Returns
        -------
        model      — with best val-F1 weights restored
        history    — {'train_loss': [...], 'val_f1': [...]}
        elapsed_s  — total wall time in seconds
        """
        import torch
        import torch.nn as nn
        from sklearn.metrics import f1_score
        from torch.utils.data import DataLoader, TensorDataset

        device = model.device
        net = model.net

        # ── DataLoaders ────────────────────────────────────────────────
        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long if self.task == "multiclass"
                         else torch.float32),
        )
        val_ds = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long if self.task == "multiclass"
                         else torch.float32),
        )
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.batch_size, shuffle=False
        )

        # ── Loss / optimizer / scheduler ───────────────────────────────
        if self.task == "binary":
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(net.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=self.patience
        )

        # ── Training loop ──────────────────────────────────────────────
        best_val_f1 = 0.0
        best_weights: dict[str, Any] = {}
        history: dict[str, list[float]] = {"train_loss": [], "val_f1": []}

        logger.info(
            "DLTrainer starting — %s | %d epochs | batch %d | device %s",
            type(model).__name__, self.epochs, self.batch_size, device,
        )
        header = f"{'Epoch':>6} {'Train Loss':>12} {'Val F1-Mac':>12} {'LR':>10}"
        logger.info(header)

        t0 = time.time()

        for epoch in range(1, self.epochs + 1):
            # ── Train ──
            net.train()
            total_loss = 0.0
            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                out = net(X_b)
                if self.task == "binary":
                    loss = criterion(out.squeeze(), y_b)
                else:
                    loss = criterion(out, y_b)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)

            # ── Validate ──
            net.eval()
            val_preds: list[np.ndarray] = []
            with torch.no_grad():
                for X_b, _ in val_loader:
                    out = net(X_b.to(device))
                    if self.task == "binary":
                        preds = (torch.sigmoid(out).squeeze() > 0.5).long()
                    else:
                        preds = out.argmax(dim=1)
                    val_preds.append(preds.cpu().numpy())

            y_val_pred = np.concatenate(val_preds)
            val_f1 = float(f1_score(y_val, y_val_pred, average="macro", zero_division=0))
            scheduler.step(val_f1)

            history["train_loss"].append(avg_loss)
            history["val_f1"].append(val_f1)

            current_lr = optimizer.param_groups[0]["lr"]
            flag = " ← best" if val_f1 > best_val_f1 else ""
            logger.info(
                "%6d %12.4f %12.4f %10.2e%s",
                epoch, avg_loss, val_f1, current_lr, flag,
            )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_weights = {
                    k: v.cpu().clone() for k, v in net.state_dict().items()
                }

        # ── Restore best weights ───────────────────────────────────────
        net.load_state_dict(best_weights)
        elapsed = time.time() - t0

        logger.info(
            "Training complete — best val F1-macro: %.4f | elapsed: %.1fs",
            best_val_f1, elapsed,
        )

        # ── Optional checkpoint ────────────────────────────────────────
        if self.checkpoint_path is not None:
            model.save(self.checkpoint_path)

        return model, history, elapsed
