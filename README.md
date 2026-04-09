# RandFusion

**ML-Based Aggregation Framework for Cryptographic Randomness Evaluation**

RandFusion uses machine learning to combine the results of multiple classical statistical randomness tests into a single, reliable verdict on whether a set of cryptographic tokens was generated with strong or weak randomness.

---

## Why RandFusion?

**The problem:** When software generates things like session tokens, API keys, or one-time codes, it needs randomness — truly unpredictable, evenly distributed bits. If the randomness is weak (biased, predictable, repetitive), an attacker could guess the token and break in.

There are existing statistical tests (like NIST's suite) that check randomness quality, but they each work alone and just give you "pass" or "fail." If you run 10 tests and 8 pass but 2 are borderline, is the randomness good or bad? That's left to human judgment.

**What we're building:** RandFusion takes all those test results — p-values, entropy scores, compression ratios, autocorrelation numbers — and feeds them as features into a machine learning model. The ML model learns the pattern of what "good randomness" looks like across all tests simultaneously, and gives you:

1. A single verdict: strong or weak randomness
2. A confidence score: how sure it is
3. An explanation: which specific test results drove the decision (via SHAP)

---

## How It Works

```
Token Batches (hex / base64 / binary)
        │
        ▼
  Preprocessing & Decoding
        │
        ▼
  Statistical Randomness Tests
  (NIST, Entropy, Autocorrelation, Compression, ...)
        │
        ▼
  Feature Vector (p-values, statistics, ratios)
        │
        ▼
  ML Ensemble (Logistic Regression + Random Forest + XGBoost → Stacking)
        │
        ▼
  Verdict: STRONG / WEAK  +  Confidence  +  SHAP Explanation
```

---

## Features

- **Multiple statistical tests** — NIST SP 800-22 subset, Shannon entropy, min-entropy, run-length stats, autocorrelation, compression ratio
- **Stacking ensemble** — combines Logistic Regression, Random Forest, and Gradient Boosted Trees via a meta-classifier
- **Explainability** — SHAP-based feature importance and per-sample decision explanations
- **Baseline comparison** — measures ML improvement over simple entropy thresholds and pass-all-tests logic
- **Reproducible** — fixed seeds, versioned configs, deterministic dataset generation

---

## Project Structure

```
RandFusion/
├── configs/            # YAML configuration files
├── data/
│   ├── raw/            # Generated token batches
│   └── processed/      # Feature matrices
├── models/             # Trained model artifacts
├── notebooks/          # EDA, training, and evaluation notebooks
├── src/
│   ├── generators/     # Strong and weak token generators
│   ├── features/       # Statistical test implementations & feature extractor
│   ├── models/         # ML training, ensemble, and inference
│   ├── baselines/      # Naive baseline classifiers
│   ├── explainability/ # SHAP integration and explanation utilities
│   ├── dashboard/      # Streamlit dashboard for results visualization
│   └── evaluation/     # Final evaluation, plots, stress tests, reporting
├── tests/              # Unit and integration tests
├── DASHBOARD.md        # Dashboard usage and troubleshooting
├── HOSTING_STREAMLIT.md # Streamlit deployment runbook
├── runtime.txt         # Python runtime pin for cloud hosting
├── PRD.md              # Product Requirements Document
├── PLAN.md             # Phased development plan
├── README.md           # This file
└── requirements.txt    # Python dependencies
```

---

## Tech Stack

| Component          | Tool                               |
|--------------------|------------------------------------|
| Language           | Python 3.10+                       |
| Statistical tests  | SciPy, custom implementations      |
| ML                 | scikit-learn, XGBoost / LightGBM   |
| Explainability     | SHAP                               |
| Data               | NumPy, Pandas                      |
| Visualization      | Matplotlib, Seaborn                |
| Configuration      | PyYAML                             |
| Notebooks          | Jupyter                            |

---

## Getting Started

### Prerequisites
- Python 3.10 or higher
- pip

### Installation

```bash
git clone https://github.com/<your-username>/RandFusion.git
cd RandFusion
python -m venv venv
venv\Scripts\activate    # On Windows
pip install -r requirements.txt
```

### Quick Start (once implemented)

```bash
# Generate synthetic dataset
python -m src.generators.generate_dataset

# Extract features
python -m src.features.extract_features

# Train models
python -m src.models.train

# Compare ML vs baselines
python -m src.baselines.compare

# Run explainability pipeline (SHAP, feature importance)
python -m src.explainability.run_explainability

# Final evaluation & reporting
python -m src.evaluation.evaluate

# Launch dashboard
streamlit run src/dashboard/app.py

# Run all tests
python -m pytest tests/ -v
```

---

## Dashboard

RandFusion includes a lightweight Streamlit dashboard for artifact-driven analysis.

### What it shows

1. Model overview metrics from `models/results.json`
2. Confusion matrix (per model + combined artifact)
3. Per-generator error analysis from `models/evaluation/per_generator_accuracy.csv`
4. SHAP summary visuals and top mean absolute SHAP features
5. Calibration reliability view from `models/evaluation/calibration_curves.png`

### Run

```bash
streamlit run src/dashboard/app.py
```

### Prerequisite

Run the core pipeline first so artifacts exist:

```bash
python -m src.generators.generate_dataset
python -m src.features.extract_features
python -m src.models.train
python -m src.baselines.compare
python -m src.explainability.run_explainability
python -m src.evaluation.evaluate
```

For detailed dashboard notes, see [DASHBOARD.md](DASHBOARD.md).

For deployment steps, see [HOSTING_STREAMLIT.md](HOSTING_STREAMLIT.md).

---

## Host the Dashboard

Recommended path: Streamlit Community Cloud.

Minimal steps:

1. Push this repository to GitHub.
2. Ensure required `models/` artifacts are committed for full visuals.
3. Create a new app at https://share.streamlit.io/.
4. Set main file to `src/dashboard/app.py`.

Full guide: [HOSTING_STREAMLIT.md](HOSTING_STREAMLIT.md).

---

## Development Phases

| Phase | Description                              | Status      |
|-------|------------------------------------------|-------------|
| 0     | Project setup & environment              | COMPLETED   |
| 1     | Synthetic dataset generation             | COMPLETED   |
| 2     | Statistical randomness tests             | COMPLETED   |
| 3     | Feature dataset construction             | COMPLETED   |
| 4     | ML model training & ensemble             | COMPLETED   |
| 5     | Baseline comparison                      | COMPLETED   |
| 6     | Explainability & interpretability        | COMPLETED   |
| 7     | Evaluation, reporting & documentation    | COMPLETED   |

See [PLAN.md](PLAN.md) for detailed breakdown of each phase.

---

## Key Concepts

- **Token batch**: A group of 1000 cryptographic tokens (128 bits each), evaluated as one sample
- **Feature vector**: Numerical outputs from all statistical tests for one batch — this is what the ML model sees
- **Stacking ensemble**: Base models make predictions, then a meta-classifier combines those predictions into a final decision
- **SHAP explanation**: Shows which statistical features pushed the decision toward "strong" or "weak"

---

## Limitations

- This is an **evaluation and auditing tool**, not a cryptographic certification system
- ML output is heuristic — it does not constitute proof of randomness quality
- The model's accuracy depends on the diversity of training generators
- Results should complement, not replace, established standards like NIST SP 800-22

---

## License

TBD

---

## References

- [NIST SP 800-22: A Statistical Test Suite for Random and Pseudorandom Number Generators](https://csrc.nist.gov/publications/detail/sp/800-22/rev-1a/final)
- [SHAP: SHapley Additive exPlanations](https://github.com/shap/shap)
- [scikit-learn: Stacking Classifier](https://scikit-learn.org/stable/modules/ensemble.html#stacking)
