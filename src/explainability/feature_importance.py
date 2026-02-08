"""
Global Feature Importance from tree-based models.

Extracts and visualises feature importances from:
  - Random Forest (Gini / mean decrease in impurity)
  - XGBoost (gain-based importance)

Produces a combined bar chart and returns a sorted DataFrame.
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from src.utils.config import get_config, PROJECT_ROOT
from src.utils.logger import get_logger
from src.features.extractor import FeatureExtractor

logger = get_logger(__name__)


def load_feature_names(config: dict | None = None) -> list[str]:
    """Return the ordered list of feature column names."""
    if config is None:
        config = get_config()
    fe = FeatureExtractor(config)
    return fe.feature_names()


def compute_importances(models_dir: Path, feature_names: list[str]) -> pd.DataFrame:
    """Load RF and XGBoost models and extract feature importances.

    Returns a DataFrame with columns:
        feature, rf_importance, xgb_importance, avg_importance
    sorted by avg_importance descending.
    """
    rf = joblib.load(models_dir / "random_forest.joblib")
    xgb = joblib.load(models_dir / "xgboost.joblib")

    rf_imp = rf.feature_importances_
    xgb_imp = xgb.feature_importances_

    df = pd.DataFrame({
        "feature": feature_names,
        "rf_importance": rf_imp,
        "xgb_importance": xgb_imp,
    })
    df["avg_importance"] = (df["rf_importance"] + df["xgb_importance"]) / 2
    df = df.sort_values("avg_importance", ascending=False).reset_index(drop=True)

    return df


def plot_feature_importance(
    importance_df: pd.DataFrame,
    output_path: Path,
    top_n: int = 15,
):
    """Side-by-side bar chart of RF vs XGBoost importances for top N features."""
    top = importance_df.head(top_n).copy()
    # Reverse so highest-importance feature is on top in horizontal bar chart
    top = top.iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 8))

    y_pos = np.arange(len(top))
    bar_height = 0.35

    ax.barh(y_pos + bar_height / 2, top["rf_importance"], bar_height,
            label="Random Forest", color="tab:green", alpha=0.8)
    ax.barh(y_pos - bar_height / 2, top["xgb_importance"], bar_height,
            label="XGBoost", color="tab:purple", alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top["feature"], fontsize=10)
    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances — RF vs XGBoost", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Feature importance plot saved to {output_path}")


def run_feature_importance(config: dict | None = None) -> pd.DataFrame:
    """Full pipeline: compute importances, save CSV + plot."""
    if config is None:
        config = get_config()

    models_dir = PROJECT_ROOT / config["model"]["output_dir"]
    output_dir = models_dir / "explainability"
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_names = load_feature_names(config)
    importance_df = compute_importances(models_dir, feature_names)

    # Save CSV
    csv_path = output_dir / "feature_importance.csv"
    importance_df.to_csv(csv_path, index=False)
    logger.info(f"Feature importance table saved to {csv_path}")

    # Save plot
    plot_feature_importance(importance_df, output_dir / "feature_importance.png")

    # Log top 10
    logger.info("\nTop 10 features by average importance:")
    for _, row in importance_df.head(10).iterrows():
        logger.info(f"  {row['feature']:30s}  RF={row['rf_importance']:.4f}  "
                     f"XGB={row['xgb_importance']:.4f}  Avg={row['avg_importance']:.4f}")

    return importance_df
