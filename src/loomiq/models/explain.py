"""SHAP explanations for the productivity model.

* Global: which features matter most across all lines (summary / bar plot).
* Local: why one specific line is predicted to miss its target.

``shap`` is imported inside functions so ``import loomiq`` stays fast.
"""

from pathlib import Path
from typing import Any

import pandas as pd

from loomiq.utils.logger import get_logger

logger = get_logger(__name__)


def build_explainer(model: Any, background: pd.DataFrame) -> Any:
    """Create a SHAP explainer for a tree model.

    Args:
        model: Fitted RandomForest or XGBoost model.
        background: Sample of training data used as the reference.
    """
    # TODO: shap.TreeExplainer(model, data=background)
    raise NotImplementedError("build_explainer is not implemented yet")


def global_importance(explainer: Any, X: pd.DataFrame) -> pd.DataFrame:
    """Return mean absolute SHAP value per feature, sorted high to low."""
    # TODO: Compute shap_values, take mean(|values|) per column, sort descending.
    raise NotImplementedError("global_importance is not implemented yet")


def explain_line(explainer: Any, line_row: pd.DataFrame, top_n: int = 5) -> list[dict[str, Any]]:
    """Explain one line's prediction in plain terms.

    Args:
        explainer: SHAP explainer from :func:`build_explainer`.
        line_row: A single-row DataFrame for the line.
        top_n: Number of strongest drivers to return.

    Returns:
        List of ``{"feature", "value", "shap_value", "direction"}`` dicts,
        where direction is "helps" or "hurts" meeting the target.
        This structure is also fed to the LLM assistant.
    """
    # TODO: Compute SHAP for the row, sort by |shap|, keep top_n, label direction.
    raise NotImplementedError("explain_line is not implemented yet")


def save_summary_plot(explainer: Any, X: pd.DataFrame, path: Path) -> None:
    """Save a SHAP summary plot (e.g. to ``reports/figures/shap_summary.png``)."""
    # TODO: shap.summary_plot(..., show=False); plt.savefig(path, bbox_inches="tight")
    raise NotImplementedError("save_summary_plot is not implemented yet")
