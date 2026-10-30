"""SHAP explanations for the productivity model.

* Global: which factors matter most across all lines.
* Local: why one specific line is predicted to miss its target.

All SHAP values here point the same way: **positive = pushes the line towards
missing its target (more risk)**, negative = pushes it towards meeting it.
One-hot columns (``team_7``, ``day_Monday``, ...) are grouped back into the
original factor so explanations read like "Team 7" instead of "team_7 = 1".
"""

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from loomiq.features.build_features import CATEGORICAL_COLUMNS
from loomiq.utils.logger import get_logger

logger = get_logger(__name__)

MISSED = 0

FEATURE_LABELS = {
    "targeted_productivity": "target level",
    "smv": "SMV (garment complexity)",
    "wip": "work in progress",
    "over_time": "overtime minutes",
    "idle_time": "idle minutes",
    "idle_men": "idle workers",
    "no_of_style_change": "style changes",
    "no_of_workers": "team size",
    "overtime_per_worker": "overtime per worker",
    "wip_per_worker": "WIP per worker",
    "idle_ratio": "share of idle workers",
    "incentive": "incentive",
    "incentive_per_worker": "incentive per worker",
    "department": "department",
    "day": "day of week",
    "quarter": "week of month",
    "team": "team",
}

# The category dropped by one-hot encoding (drop_first=True): all zeros means this value
BASE_CATEGORIES = {"department": "sewing", "day": "Saturday", "quarter": "Quarter1", "team": "1"}


def _import_shap() -> Any:
    """Import shap, hiding a known harmless warning from its own colormap code."""
    with warnings.catch_warnings():
        # shap calls matplotlib's Colormap.set_bad/set_over/set_under, which matplotlib
        # plans to deprecate. Nothing in LoomIQ uses them; remove once shap is updated.
        warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
        import shap
    return shap


def original_column(column: str) -> str:
    """Map an encoded column back to its source, e.g. ``team_7`` -> ``team``."""
    for categorical in CATEGORICAL_COLUMNS:
        if column.startswith(f"{categorical}_"):
            return categorical
    return column


def readable_label(column: str) -> str:
    """Human-friendly name for any input column, e.g. ``day_Thursday`` -> ``day of week: Thursday``."""
    factor = original_column(column)
    label = FEATURE_LABELS.get(factor, factor)
    if factor != column:
        return f"{label}: {column.removeprefix(factor + '_')}"
    return label


class RiskExplainer:
    """SHAP explainer whose values always measure push towards *missing* the target."""

    def __init__(self, model: Any, background: pd.DataFrame) -> None:
        """Create the explainer.

        Args:
            model: Fitted model (tree models use the fast TreeExplainer).
            background: Sample of training inputs, used as the reference for
                non-tree models.
        """
        shap = _import_shap()

        self.is_tree = _is_tree_model(model)
        if self.is_tree:
            self._explainer = shap.TreeExplainer(model)
        else:
            self._explainer = shap.Explainer(
                lambda data: model.predict_proba(data)[:, MISSED], background
            )

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Return SHAP values of shape (rows, features); positive = more risk."""
        if not self.is_tree:
            return self._explainer(X).values
        values = np.asarray(self._explainer.shap_values(X))
        if values.ndim == 3:  # one set of values per class (random forest)
            return values[:, :, MISSED]
        return -values  # single output explains "met" (XGBoost), so flip it


def _is_tree_model(model: Any) -> bool:
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

    return isinstance(model, (RandomForestClassifier, XGBClassifier))


def build_explainer(model: Any, background: pd.DataFrame, max_background: int = 100) -> RiskExplainer:
    """Create a :class:`RiskExplainer` for a fitted model.

    Args:
        model: Fitted model from the saved bundle.
        background: Training inputs; a random sample is used as the reference.
        max_background: Maximum background rows (keeps non-tree explainers fast).
    """
    sample = background.sample(min(len(background), max_background), random_state=42)
    return RiskExplainer(model, sample)


def global_importance(explainer: RiskExplainer, X: pd.DataFrame, group_one_hot: bool = True) -> pd.DataFrame:
    """Return mean absolute SHAP value per factor, sorted high to low.

    Args:
        explainer: From :func:`build_explainer`.
        X: Model inputs to explain (e.g. the test set).
        group_one_hot: Add up one-hot columns into their original factor.

    Returns:
        DataFrame with ``feature``, ``label`` and ``mean_abs_shap``.
    """
    values = pd.DataFrame(np.abs(explainer.shap_values(X)), columns=X.columns)
    importance = values.mean()
    if group_one_hot:
        importance = importance.groupby(original_column).sum()
    table = importance.sort_values(ascending=False).rename("mean_abs_shap").reset_index()
    table.columns = ["feature", "mean_abs_shap"]
    table.insert(1, "label", table["feature"].map(lambda f: FEATURE_LABELS.get(f, f)))
    return table


def _active_category(row: pd.Series, factor: str) -> str:
    """Return the category that is "on" for a one-hot encoded factor in one row."""
    prefix = f"{factor}_"
    for column, value in row.items():
        if column.startswith(prefix) and value == 1:
            return column.removeprefix(prefix)
    return BASE_CATEGORIES[factor]


def explain_line(explainer: RiskExplainer, line_row: pd.DataFrame, top_n: int = 5) -> list[dict[str, Any]]:
    """Explain one line's prediction in plain terms.

    Args:
        explainer: From :func:`build_explainer`.
        line_row: A single-row DataFrame of model inputs for the line.
        top_n: Number of strongest drivers to return.

    Returns:
        List of dicts with ``feature``, ``label``, ``value``, ``shap_value`` and
        ``direction`` ("raises risk" or "lowers risk"), strongest first.
        This structure is also passed to the LLM assistant.
    """
    if len(line_row) != 1:
        raise ValueError("line_row must contain exactly one row")

    row = line_row.iloc[0]
    contributions = pd.Series(explainer.shap_values(line_row)[0], index=line_row.columns)
    grouped = contributions.groupby(original_column).sum()

    drivers = []
    for factor, shap_value in grouped.reindex(grouped.abs().sort_values(ascending=False).index).head(top_n).items():
        if factor in CATEGORICAL_COLUMNS:
            value: Any = _active_category(row, factor)
        else:
            value = round(float(row[factor]), 3)
        drivers.append(
            {
                "feature": factor,
                "label": FEATURE_LABELS.get(factor, factor),
                "value": value,
                "shap_value": round(float(shap_value), 4),
                "direction": "raises risk" if shap_value > 0 else "lowers risk",
            }
        )
    return drivers


def save_importance_plot(importance: pd.DataFrame, path: Path, top_n: int = 12) -> Path:
    """Save a horizontal bar chart of grouped global importance."""
    import matplotlib.pyplot as plt

    from loomiq.utils.plotting import DEPARTMENT_COLORS, apply_style

    apply_style()
    top = importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(top) + 1))
    ax.barh(top["label"], top["mean_abs_shap"], color=DEPARTMENT_COLORS["sewing"], height=0.6)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("average impact on risk of missing target (mean |SHAP|)")
    ax.set_title("What drives the risk of missing target")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_summary_plot(explainer: RiskExplainer, X: pd.DataFrame, path: Path, max_display: int = 12) -> Path:
    """Save a SHAP beeswarm plot (each dot is one team-day; right = more risk)."""
    import matplotlib.pyplot as plt

    shap = _import_shap()

    values = explainer.shap_values(X)
    labels = [readable_label(c) for c in X.columns]
    shap.summary_plot(values, X, feature_names=labels, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.axes[0].set_xlabel("SHAP value (right = pushes towards missing target)")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_global_explanations() -> list[Path]:
    """Explain the saved model on the test days and save both SHAP figures.

    Returns:
        Paths of the saved figures in ``reports/figures/``.
    """
    from loomiq.config import get_path, load_config
    from loomiq.data.clean import load_clean_productivity
    from loomiq.features.build_features import build_features
    from loomiq.models.predict import load_model, prepare_features
    from loomiq.models.train_productivity import time_based_split

    bundle = load_model()
    train_df, test_df = time_based_split(load_clean_productivity(), load_config()["productivity_model"]["test_size"])
    X_train, _ = build_features(train_df, bundle["use_incentive"])
    X_test = prepare_features(bundle, test_df)

    explainer = build_explainer(bundle["model"], X_train.reindex(columns=bundle["feature_names"], fill_value=0))
    figures = get_path("figures")
    paths = [
        save_importance_plot(global_importance(explainer, X_test), figures / "shap_importance.png"),
        save_summary_plot(explainer, X_test, figures / "shap_summary.png"),
    ]
    logger.info("Saved SHAP figures: %s", [p.name for p in paths])
    return paths


if __name__ == "__main__":
    save_global_explanations()
