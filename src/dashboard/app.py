"""
Lightweight Streamlit dashboard for RandFusion artifacts.

Usage:
    streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

# Make project root importable when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import get_config  # noqa: E402


def _artifact(path: Path) -> Path:
    return path


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _show_missing(path: Path):
    st.warning(f"Missing artifact: {path.relative_to(PROJECT_ROOT)}")


def _render_confusion_matrix(cm: list[list[int]], model_name: str):
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Weak", "Strong"],
        yticklabels=["Weak", "Strong"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix: {model_name}")
    st.pyplot(fig, clear_figure=True)


def _render_per_generator(df: pd.DataFrame):
    plot_df = df.copy()
    plot_df["errors"] = plot_df["n_samples"] - plot_df["correct"]
    plot_df = plot_df.sort_values("errors", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.barplot(data=plot_df, x="generator", y="errors", hue="true_label", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_ylabel("Error Count")
    ax.set_xlabel("Generator")
    ax.set_title("Per-Generator Error Count (higher means harder)")
    ax.legend(title="True Label", labels=["Weak (0)", "Strong (1)"])
    st.pyplot(fig, clear_figure=True)

    show_df = plot_df[["generator", "n_samples", "correct", "accuracy", "errors", "true_label"]]
    st.dataframe(show_df, use_container_width=True)


def _render_shap_table(shap_csv: Path):
    if not shap_csv.exists():
        _show_missing(shap_csv)
        return

    shap_df = pd.read_csv(shap_csv)
    mean_abs = shap_df.abs().mean().sort_values(ascending=False).head(12)

    fig, ax = plt.subplots(figsize=(9, 4.6))
    sns.barplot(x=mean_abs.values, y=mean_abs.index, orient="h", ax=ax)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_ylabel("Feature")
    ax.set_title("Top Features by Mean Absolute SHAP Value")
    st.pyplot(fig, clear_figure=True)

    table_df = pd.DataFrame({"feature": mean_abs.index, "mean_abs_shap": mean_abs.values})
    st.dataframe(table_df, use_container_width=True)


def main():
    st.set_page_config(page_title="RandFusion Dashboard", layout="wide")

    config = get_config()
    models_dir = PROJECT_ROOT / config["model"]["output_dir"]
    eval_dir = models_dir / "evaluation"
    exp_dir = models_dir / "explainability"
    comparison_dir = models_dir / "comparison"

    results_path = models_dir / "results.json"
    per_gen_path = eval_dir / "per_generator_accuracy.csv"
    calib_img = eval_dir / "calibration_curves.png"
    cm_img = eval_dir / "confusion_matrices.png"
    shap_bar_img = exp_dir / "shap_summary_bar.png"
    shap_bee_img = exp_dir / "shap_summary_beeswarm.png"
    shap_csv = exp_dir / "shap_values.csv"
    comparison_path = comparison_dir / "comparison_results.json"

    results = _read_json(results_path)
    comparison = _read_json(comparison_path)

    st.title("RandFusion Dashboard")
    st.caption("Lightweight visualization for model performance, errors, explainability, and calibration.")

    with st.sidebar:
        st.header("Artifacts")
        for p in [
            results_path,
            comparison_path,
            per_gen_path,
            cm_img,
            calib_img,
            shap_bar_img,
            shap_bee_img,
            shap_csv,
        ]:
            status = "✅" if p.exists() else "❌"
            st.write(f"{status} {p.relative_to(PROJECT_ROOT)}")

    tabs = st.tabs([
        "Overview",
        "Confusion Matrix",
        "Per-Generator Errors",
        "SHAP Summaries",
        "Calibration",
    ])

    with tabs[0]:
        st.subheader("Model Test Metrics")
        if not results:
            _show_missing(results_path)
        else:
            test_results = results.get("test_results", {})
            if not test_results:
                st.warning("No test_results key found in results.json")
            else:
                model_names = list(test_results.keys())
                selected = st.selectbox("Select model", model_names, index=0)
                m = test_results[selected]

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Accuracy", f"{m.get('accuracy', 0):.4f}")
                c2.metric("Precision", f"{m.get('precision', 0):.4f}")
                c3.metric("Recall", f"{m.get('recall', 0):.4f}")
                c4.metric("F1", f"{m.get('f1', 0):.4f}")
                c5.metric("ROC-AUC", f"{m.get('roc_auc', 0):.4f}")

                df = pd.DataFrame(test_results).T.reset_index().rename(columns={"index": "model"})
                st.dataframe(df, use_container_width=True)

        st.subheader("Baseline vs ML Comparison")
        if comparison:
            comp_df = pd.DataFrame(comparison).T.reset_index().rename(columns={"index": "method"})
            keep_cols = [
                c
                for c in [
                    "method",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "roc_auc",
                    "recall_weak",
                ]
                if c in comp_df.columns
            ]
            st.dataframe(comp_df[keep_cols], use_container_width=True)
        else:
            _show_missing(comparison_path)

    with tabs[1]:
        st.subheader("Confusion Matrix by Model")
        if not results:
            _show_missing(results_path)
        else:
            test_results = results.get("test_results", {})
            names = list(test_results.keys())
            if names:
                selected = st.selectbox("Model", names, index=min(1, len(names) - 1), key="cm_model")
                cm = test_results[selected].get("confusion_matrix")
                if cm:
                    _render_confusion_matrix(cm, selected)
                else:
                    st.warning("Confusion matrix missing for selected model.")

        st.subheader("Combined Confusion Matrix Artifact")
        if cm_img.exists():
            st.image(str(cm_img), caption="All models confusion matrices", use_container_width=True)
        else:
            _show_missing(cm_img)

    with tabs[2]:
        st.subheader("Per-Generator Error Analysis")
        if not per_gen_path.exists():
            _show_missing(per_gen_path)
        else:
            df = pd.read_csv(per_gen_path)
            _render_per_generator(df)

            hard_df = (df.assign(errors=lambda d: d["n_samples"] - d["correct"])\
                        .sort_values("errors", ascending=False)\
                        .head(3))
            st.info(
                "Hardest generators in current test split: "
                + ", ".join(hard_df["generator"].tolist())
            )

    with tabs[3]:
        st.subheader("SHAP Summary Visuals")
        c1, c2 = st.columns(2)
        with c1:
            if shap_bar_img.exists():
                st.image(str(shap_bar_img), caption="SHAP Summary Bar", use_container_width=True)
            else:
                _show_missing(shap_bar_img)
        with c2:
            if shap_bee_img.exists():
                st.image(str(shap_bee_img), caption="SHAP Bee-swarm", use_container_width=True)
            else:
                _show_missing(shap_bee_img)

        st.subheader("Top Features from SHAP Values")
        _render_shap_table(shap_csv)

    with tabs[4]:
        st.subheader("Calibration")
        if calib_img.exists():
            st.image(str(calib_img), caption="Reliability and probability distribution", use_container_width=True)
        else:
            _show_missing(calib_img)

        st.markdown(
            "**Interpretation guide:** A perfectly calibrated model follows the diagonal. "
            "If calibration deviates strongly in the 0.5-0.7 range, treat borderline scores with caution."
        )


if __name__ == "__main__":
    main()
