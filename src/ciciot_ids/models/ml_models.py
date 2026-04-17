"""
Sklearn-compatible ML model wrappers (Open/Closed: extend BaseMLModel,
never modify it; Liskov: RF and XGBoost are interchangeable here).
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from ciciot_ids.models.base import BaseMLModel

logger = logging.getLogger(__name__)


class RandomForestModel(BaseMLModel):
    """
    Random Forest wrapper.

    Parameters
    ----------
    **params
        Forwarded to sklearn.ensemble.RandomForestClassifier.
        Sensible defaults match notebook 03/07.
    """

    _DEFAULTS: dict[str, Any] = dict(
        n_estimators=100,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
        verbose=0,
    )

    def __init__(self, **params: Any) -> None:
        from sklearn.ensemble import RandomForestClassifier

        cfg = {**self._DEFAULTS, **params}
        self._model = RandomForestClassifier(**cfg)
        logger.debug("RandomForestModel config: %s", cfg)

    # ------------------------------------------------------------------
    # BaseMLModel interface
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        **kwargs: Any,
    ) -> "RandomForestModel":
        logger.info("Fitting Random Forest on %d samples …", len(X_train))
        self._model.fit(X_train, y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self._model, fh)
        logger.info("Random Forest saved → %s", path)

    @classmethod
    def load(cls, path: Path) -> "RandomForestModel":
        obj = cls.__new__(cls)
        with open(path, "rb") as fh:
            obj._model = pickle.load(fh)
        logger.info("Random Forest loaded ← %s", path)
        return obj

    # ------------------------------------------------------------------
    # Extra
    # ------------------------------------------------------------------

    @property
    def feature_importances_(self) -> np.ndarray:
        return self._model.feature_importances_


class XGBoostModel(BaseMLModel):
    """
    XGBoost classifier wrapper.

    Parameters
    ----------
    task : {'binary', 'multiclass'}
        Sets the objective and num_class automatically.
    num_classes : int
        Required when task='multiclass'.
    **params
        Additional params forwarded to xgb.XGBClassifier.
    """

    _DEFAULTS_BINARY: dict[str, Any] = dict(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )

    _DEFAULTS_MULTI: dict[str, Any] = dict(
        objective="multi:softprob",
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=42,
        verbosity=1,
    )

    def __init__(
        self,
        task: str = "binary",
        num_classes: int = 2,
        **params: Any,
    ) -> None:
        import xgboost as xgb

        self._task = task
        if task == "binary":
            cfg = {**self._DEFAULTS_BINARY, **params}
        else:
            cfg = {**self._DEFAULTS_MULTI, "num_class": num_classes, **params}

        self._model = xgb.XGBClassifier(**cfg)
        logger.debug("XGBoostModel config: %s", cfg)

    # ------------------------------------------------------------------
    # BaseMLModel interface
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        **kwargs: Any,
    ) -> "XGBoostModel":
        logger.info("Fitting XGBoost on %d samples …", len(X_train))
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self._model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self._model, fh)
        logger.info("XGBoost saved → %s", path)

    @classmethod
    def load(cls, path: Path) -> "XGBoostModel":
        obj = cls.__new__(cls)
        obj._task = "unknown"
        with open(path, "rb") as fh:
            obj._model = pickle.load(fh)
        logger.info("XGBoost loaded ← %s", path)
        return obj

    # ------------------------------------------------------------------
    # Extra
    # ------------------------------------------------------------------

    @property
    def feature_importances_(self) -> np.ndarray:
        return self._model.feature_importances_
