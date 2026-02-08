# PLAN.md
## RandFusion – Development Plan

---

## Phase 0: Project Setup & Environment
**Goal:** Establish a clean, reproducible development environment.

### Tasks
- [x] Initialize Python project structure (src/, tests/, data/, notebooks/, configs/)
- [x] Set up virtual environment and `requirements.txt` / `pyproject.toml`
- [x] Core dependencies: NumPy, SciPy, scikit-learn, XGBoost, SHAP, matplotlib
- [x] Set up logging, configuration management (YAML-based), and random seed control
- [x] Create `.gitignore`, linting config, and formatting rules
- [x] Write a minimal "hello world" pipeline to verify end-to-end tooling

### Deliverables
- Working project skeleton that can be cloned and run with zero friction
- Config file (`config.yaml`) with all tunable parameters (token length, batch size, seeds, etc.)

---

## Phase 1: Synthetic Dataset Generation
**Goal:** Create labeled datasets of strong and weak random token batches.

### Tasks
- [x] Implement **strong randomness generators**
  - Python `secrets` module (CSPRNG)
  - `os.urandom`
- [x] Implement **weak randomness generators** (intentionally flawed)
  - Linear Congruential Generator (LCG) with small state
  - Biased coin (non-uniform bit distribution)
  - Truncated / low-entropy sources (e.g., repeating short seed)
  - Time-seeded `random.Random` with predictable seeds
  - XOR-collapsed outputs (reducing effective entropy)
- [x] Each generator produces batches of 1000 tokens × 128 bits
- [x] Label each batch: `1 = strong`, `0 = weak`
- [x] Generate balanced dataset (e.g., 500 strong + 500 weak batches)
- [x] Save dataset in a structured format (CSV/Parquet with metadata)
- [x] Add a data validation step to catch degenerate batches

### Deliverables
- `data/raw/` directory with labeled token batches
- Generator scripts under `src/generators/`
- A notebook demonstrating sample batches visually

---

## Phase 2: Statistical Randomness Tests (Feature Generators)
**Goal:** Run classical randomness tests on each token batch and extract numerical features.

### Tasks
- [x] Implement or wrap the following tests:
  - **NIST SP 800-22 subset:** Frequency (monobit), Block Frequency, Runs, Longest Run of Ones, Serial, Approximate Entropy, Cumulative Sums
  - **Shannon entropy** of the bitstream
  - **Min-entropy** estimation (most-common-value estimator)
  - **Run-length statistics** (mean, variance, max run)
  - **Autocorrelation** at multiple lags (e.g., lag 1, 2, 4, 8, 16)
  - **Compression ratio** (zlib compressed size / original size)
- [x] Each test returns one or more numerical values (p-values, test statistics, ratios)
- [x] Build a `FeatureExtractor` class that takes a token batch → returns a feature vector
- [x] Handle edge cases: all-zero batches, very short streams, NaN outputs
- [x] Document each feature with name, description, expected range

### Deliverables
- `src/features/` module with individual test implementations
- `FeatureExtractor` class producing a consistent feature vector per batch
- Feature documentation table

---

## Phase 3: Feature Dataset Construction
**Goal:** Convert raw token batches into a structured feature matrix ready for ML.

### Tasks
- [ ] Run `FeatureExtractor` on every batch from Phase 1
- [ ] Produce a feature matrix: rows = batches, columns = features, last column = label
- [ ] Perform exploratory data analysis (EDA)
  - Feature distributions for strong vs. weak classes
  - Correlation heatmap between features
  - Identify redundant or uninformative features
- [ ] Save processed dataset (`data/processed/features.csv`)

### Deliverables
- Clean feature dataset
- EDA notebook with visualizations
- Initial feature selection insights

---

## Phase 4: ML Model Training & Ensemble
**Goal:** Train individual classifiers and build the stacking ensemble.

### Tasks
- [ ] Split data: 70% train / 15% validation / 15% test (stratified)
- [ ] Train **base models**:
  - Logistic Regression (with regularization)
  - Random Forest
  - Gradient Boosted Trees (XGBoost or LightGBM)
- [ ] Evaluate each base model individually on validation set
- [ ] Build **stacking ensemble**:
  - Base model predictions (probabilities) become meta-features
  - Meta-classifier: Logistic Regression (keeps it interpretable)
  - Use cross-validated predictions to avoid data leakage
- [ ] Hyperparameter tuning (grid search or Optuna) on validation set
- [ ] Final evaluation on held-out test set

### Deliverables
- Trained models saved under `models/`
- Training scripts under `src/models/`
- Validation results table (accuracy, precision, recall, F1, ROC-AUC)

---

## Phase 5: Baseline Comparison
**Goal:** Quantify how much the ML ensemble improves over naive approaches.

### Tasks
- [ ] Implement **Baseline 1 – Entropy Threshold**: classify as weak if Shannon entropy < threshold
- [ ] Implement **Baseline 2 – All-Tests-Pass**: classify as weak if any NIST test p-value < 0.01
- [ ] Evaluate both baselines on the same test set
- [ ] Compare: accuracy, recall on weak class, ROC-AUC
- [ ] Produce comparison table and ROC curve overlay plot

### Deliverables
- Baseline implementations in `src/baselines/`
- Side-by-side comparison table and plots
- Written analysis of where ML ensemble outperforms baselines

---

## Phase 6: Explainability & Interpretability
**Goal:** Make every ML decision transparent and auditable.

### Tasks
- [ ] Compute **global feature importance** (from Random Forest + XGBoost)
- [ ] Generate **SHAP values** for the ensemble
  - Summary plot (feature importance across all samples)
  - Dependence plots for top features
  - Force plots for individual sample explanations
- [ ] Build a `explain_decision(sample)` function that returns:
  - Predicted class + confidence
  - Top contributing features (positive and negative)
  - Natural-language summary sentence
- [ ] Document interpretation guidelines for end users

### Deliverables
- SHAP integration in `src/explainability/`
- Explanation notebook with example outputs
- User-facing interpretation guide

---

## Phase 7: Evaluation, Reporting & Documentation
**Goal:** Produce final results, write-up, and polish the project.

### Tasks
- [ ] Generate **confusion matrix** on test set
- [ ] Plot **ROC curve** and **Precision-Recall curve**
- [ ] Compute **calibration curve** (are confidence scores reliable?)
- [ ] Compile all results into a structured report / notebook
- [ ] Stress-test with edge cases:
  - Generators not seen during training
  - Varying token lengths (64, 256 bits)
  - Very small batch sizes
- [ ] Finalize README, code comments, and docstrings
- [ ] Clean up repository structure

### Deliverables
- Final evaluation report (notebook or PDF)
- Polished repository ready for sharing
- Summary of findings and limitations

---

## Phase 8 (Optional): Extensions & Hardening
**Goal:** Stretch goals for deeper work or future iterations.

### Ideas
- [ ] Add more weak generators (Mersenne Twister weaknesses, PHP `rand()`, etc.)
- [ ] Integrate real-world token samples (JWTs, session IDs) as additional test inputs
- [ ] Build a simple CLI tool: `randfusion evaluate --input tokens.hex`
- [ ] Add a lightweight web dashboard (Streamlit) for interactive evaluation
- [ ] Experiment with neural network–based classifiers (MLP, 1D-CNN on raw bits)
- [ ] Cross-validation study on generalization to unseen generator families
- [ ] Package as an installable Python library

---

## Technology Stack

| Component             | Tool / Library                        |
|----------------------|---------------------------------------|
| Language             | Python 3.10+                          |
| Statistical tests    | SciPy, custom NIST implementations    |
| ML models            | scikit-learn, XGBoost / LightGBM      |
| Explainability       | SHAP                                  |
| Data handling        | NumPy, Pandas                         |
| Visualization        | Matplotlib, Seaborn                   |
| Configuration        | PyYAML                                |
| Compression test     | zlib (stdlib)                         |
| Notebooks            | Jupyter                               |
| Version control      | Git                                   |

