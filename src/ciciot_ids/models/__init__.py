"""Model package — exposes base interfaces and concrete implementations."""

from ciciot_ids.models.base import BaseModel, BaseMLModel, BaseDLModel
from ciciot_ids.models.ml_models import RandomForestModel, XGBoostModel
try:
    from ciciot_ids.models.dl_models import MLPModel, CNN1DModel
except ImportError:
    MLPModel = None  # type: ignore[assignment,misc]
    CNN1DModel = None  # type: ignore[assignment,misc]

__all__ = [
    "BaseModel",
    "BaseMLModel",
    "BaseDLModel",
    "RandomForestModel",
    "XGBoostModel",
    "MLPModel",
    "CNN1DModel",
]
