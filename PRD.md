# PRD.md  
## RandFusion – ML-Based Aggregation Framework for Cryptographic Randomness Evaluation

---

## 1. Product Overview

RandFusion is a machine learning–assisted evaluation framework designed to assess the quality of randomness used in cryptographic token generation. It combines outputs from classical statistical randomness tests into a unified decision using an ensemble ML model.

The system is intended as a **randomness auditing and evaluation tool**, not as a replacement for cryptographic standards or statistical test suites.

---

## 2. Problem Definition

Cryptographic security relies on high-quality randomness for generating session tokens, nonces, identifiers, and keys. Existing randomness evaluation approaches rely on independent statistical tests with fixed thresholds, producing fragmented outputs that require expert interpretation.

These approaches face limitations when applied to cryptographic tokens:
- Tokens are short and discrete rather than long contiguous streams
- Statistical tests operate independently and ignore inter-test correlations
- Binary pass/fail decisions discard valuable information
- Borderline randomness failures are difficult to interpret consistently

RandFusion addresses these issues by treating randomness evaluation as a **classification problem**, where multiple statistical indicators are combined using machine learning.

---

## 3. Product Goals

### Primary Goals
- Evaluate randomness quality in cryptographic token generation
- Aggregate multiple statistical randomness indicators into a single decision
- Detect weak randomness missed by individual tests
- Provide interpretable results suitable for auditing and analysis

### Non-Goals
- Designing new cryptographic random number generators
- Replacing NIST or other standardized randomness test suites
- Proving cryptographic security or unpredictability
- Performing cryptanalysis or breaking encryption schemes

---

## 4. Target Users

- Cryptography and network security researchers
- Security auditors and analysts
- Backend and platform engineers validating token generation
- Academic users studying applied cryptographic evaluation

---

## 5. System Scope

### Included
- Cryptographic token randomness evaluation
- Statistical randomness testing integration
- Feature extraction from test outputs
- Supervised ML classification and ensemble learning
- Explainability and decision analysis

### Excluded
- Encryption/decryption workflows
- Hardware RNG certification
- Cryptographic proof systems
- Production security enforcement

---

## 6. High-Level Architecture

Token Generator Output
↓
Token Preprocessing & Decoding
↓
Statistical Randomness Tests
↓
Feature Vector Construction
↓
ML Ensemble (RandFusion)
↓
Randomness Quality Decision + Explanation


---

## 7. Functional Requirements

---

### FR1: Token Dataset Input

The system shall support cryptographic token sequences as input.

**Requirements**
- Fixed token length (default: 128 bits)
- Support for encoded formats (hex, base64)
- Batch-based evaluation (default: 1000 tokens per sample)

---

### FR2: Token Preprocessing

The system shall normalize token inputs before analysis.

**Requirements**
- Detect token encoding format
- Decode tokens to raw binary
- Concatenate tokens into fixed-size bitstreams
- Validate consistency and integrity

---

### FR3: Statistical Randomness Evaluation

The system shall compute classical randomness metrics.

**Required Tests**
- NIST SP 800-22 statistical tests
- Shannon entropy
- Min-entropy estimation
- Run-length statistics
- Autocorrelation metrics
- Compression-based indicators

Statistical tests shall be treated as **feature generators**, not final decision points.

---

### FR4: Feature Extraction

The system shall convert statistical outputs into structured feature vectors.

**Feature Categories**
- Raw p-values from statistical tests
- Test statistics where available
- Aggregated metrics for multi-output tests
- Entropy and complexity measures
- Token-specific distribution metrics

Each evaluation sample shall correspond to one feature vector.

---

### FR5: Machine Learning Classification

The system shall classify randomness quality using supervised ML.

**Base Models**
- Logistic Regression
- Random Forest
- Gradient Boosted Trees

**Ensemble Method**
- Stacking ensemble with a meta-classifier
- Output includes class label and confidence score

---

### FR6: Baseline Comparison

The system shall support comparison against classical baselines.

**Baselines**
- Entropy threshold-based classification
- Pass-all-statistical-tests decision logic

---

### FR7: Evaluation Metrics

The system shall report:
- Accuracy
- Precision
- Recall (priority on weak randomness detection)
- F1 score
- ROC-AUC
- Confusion matrix
- Calibration metrics

---

### FR8: Explainability

The system shall provide interpretable explanations.

**Requirements**
- Feature importance ranking
- SHAP or equivalent explanation output
- Per-sample decision contribution analysis

---

## 8. Non-Functional Requirements

### Interpretability
All ML decisions must be explainable using statistical features.

### Reproducibility
- Deterministic dataset generation
- Fixed random seeds
- Versioned configuration files

### Performance
- Feature extraction must be lightweight
- ML inference must support batch evaluation

### Extensibility
- New statistical tests can be added as features
- New RNG/token sources can be integrated

---

## 9. Dataset Specification

### Token Parameters
- Token length: 128 bits
- Batch size: 1000 tokens per evaluation sample

### Randomness Classes
- Strong randomness sources (CSPRNG-based)
- Weak randomness sources (biased, predictable, truncated, low-state generators)

Datasets may be synthetic or derived from standard RNG outputs.

---

## 10. Risks and Constraints

### Risks
- ML overfitting to specific generators
- False confidence on unseen RNGs
- Dataset bias due to synthetic data

### Constraints
- ML output is heuristic, not cryptographic proof
- Final decisions must be interpreted conservatively
- Results complement, not override, existing standards

---

## 11. Expected Outcomes

- Demonstration of ML as an effective aggregation layer
- Empirical comparison with classical randomness testing
- Identification of scenarios where aggregation improves detection
- Clear articulation of ML limitations in cryptographic evaluation

---

## 12. Product Identity

**Name:** RandFusion  
**Definition:** A machine learning–based decision aggregation framework for evaluating cryptographic token randomness by fusing classical statistical test outputs into a unified assessment.

---

