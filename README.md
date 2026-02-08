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
│   └── explainability/ # SHAP integration and explanation utilities
├── tests/              # Unit and integration tests
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

# Evaluate
python -m src.models.evaluate
```

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
| 7     | Evaluation, reporting & documentation    | IN PROGRESS |

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
