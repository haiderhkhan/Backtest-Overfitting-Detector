"""Backtest Overfitting Detector — "how much do I trust this backtest?"

Pipeline: log_trial -> deflated_sharpe_ratio + compute_pbo
          -> walk_forward_validate -> generate_health_report
"""

from .deflated_sharpe import deflated_sharpe_ratio
from .pbo import compute_pbo
from .report import DEFAULT_THRESHOLDS, generate_health_report
from .trial_log import get_all_trials, get_returns_matrix, get_trial_returns, log_trial
from .walk_forward import walk_forward_validate

__all__ = [
    "log_trial",
    "get_all_trials",
    "get_trial_returns",
    "get_returns_matrix",
    "deflated_sharpe_ratio",
    "compute_pbo",
    "walk_forward_validate",
    "generate_health_report",
    "DEFAULT_THRESHOLDS",
]
