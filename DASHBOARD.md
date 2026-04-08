# RandFusion Dashboard

A lightweight Streamlit app to inspect key RandFusion outputs without opening multiple files manually.

## File Location

- App entrypoint: `src/dashboard/app.py`

## Launch

```bash
streamlit run src/dashboard/app.py
```

## What the Dashboard Covers

### 1. Overview

- Reads `models/results.json`
- Displays per-model test metrics:
  - Accuracy
  - Precision
  - Recall
  - F1
  - ROC-AUC
- Reads `models/comparison/comparison_results.json` for baseline vs ML table

### 2. Confusion Matrix

- Renders matrix for selected model from `models/results.json`
- Also displays pre-generated combined image:
  - `models/evaluation/confusion_matrices.png`

### 3. Per-Generator Errors

- Reads `models/evaluation/per_generator_accuracy.csv`
- Computes error counts per generator (`n_samples - correct`)
- Plots and tabulates hardest generator categories

### 4. SHAP Summaries

- Displays:
  - `models/explainability/shap_summary_bar.png`
  - `models/explainability/shap_summary_beeswarm.png`
- Reads `models/explainability/shap_values.csv`
- Computes top features by mean absolute SHAP value

### 5. Calibration

- Displays `models/evaluation/calibration_curves.png`
- Includes guidance for interpreting reliability behavior

## Artifact Prerequisites

The dashboard is artifact-driven, so run these first:

```bash
python -m src.generators.generate_dataset
python -m src.features.extract_features
python -m src.models.train
python -m src.baselines.compare
python -m src.explainability.run_explainability
python -m src.evaluation.evaluate
```

## Troubleshooting

### Missing artifact warnings

If the dashboard warns that files are missing, verify the corresponding pipeline stage was run and completed successfully.

### Streamlit command not found

Install dependencies:

```bash
pip install -r requirements.txt
```

Then relaunch:

```bash
streamlit run src/dashboard/app.py
```

### Wrong paths in multi-folder setup

Run the command from project root (the folder containing `src/`, `models/`, and `configs/`).

## Notes

- The app is read-only and does not retrain models.
- It is designed for lightweight review and paper/report support.
