"""Leakage-safe probabilistic Formula 1 race forecasting.

Public entry points are imported explicitly - never with a wildcard - so the
package namespace is auditable and name collisions cannot occur silently.
"""

from __future__ import annotations

from .config import DEFAULT_MODEL_CONFIG, ModelConfig, regime_for
from .leakage import LeakageError, assert_stage_allowed, shifted_expanding
from .pipeline import Forecast, Panel, build_panel, forecast_race, resolve_circuit

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_MODEL_CONFIG",
    "Forecast",
    "LeakageError",
    "ModelConfig",
    "Panel",
    "assert_stage_allowed",
    "build_panel",
    "forecast_race",
    "regime_for",
    "resolve_circuit",
    "shifted_expanding",
    "__version__",
]
