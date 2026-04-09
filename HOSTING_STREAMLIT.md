# Streamlit Hosting Guide

This guide explains exactly how to host the RandFusion dashboard and what you need to do manually.

## Hosting Target (Recommended)

- Platform: Streamlit Community Cloud
- App entrypoint: `src/dashboard/app.py`
- Python runtime: `python-3.11` (via `runtime.txt`)

## What Is Already Prepared

The repo now includes:

- `src/dashboard/app.py` dashboard app
- `requirements.txt` with `streamlit`
- `.streamlit/config.toml` for app UI/server defaults
- `runtime.txt` pinned to Python 3.11

## You Must Do (Manual Steps)

### 1. Push the repository to GitHub

If not already pushed:

```bash
git add .
git commit -m "Add Streamlit dashboard and hosting config"
git push origin <your-branch>
```

### 2. Ensure required artifacts are in the repo

The dashboard reads generated files from `models/`.

At minimum, commit these if you want a fully populated hosted dashboard:

- `models/results.json`
- `models/comparison/comparison_results.json`
- `models/evaluation/per_generator_accuracy.csv`
- `models/evaluation/confusion_matrices.png`
- `models/evaluation/calibration_curves.png`
- `models/explainability/shap_summary_bar.png`
- `models/explainability/shap_summary_beeswarm.png`
- `models/explainability/shap_values.csv`

If these files are not present, dashboard sections will still load and show warnings for missing artifacts.

### 3. Deploy on Streamlit Community Cloud

1. Open https://share.streamlit.io/
2. Sign in with GitHub
3. Click **New app**
4. Select your repository and branch
5. Set **Main file path** to:
   - `src/dashboard/app.py`
6. Leave advanced settings default, then deploy

### 4. First deploy behavior

- Build usually takes a few minutes
- You will get a public URL (for example `https://<app-name>.streamlit.app`)

## If Build Fails

### Common issue: dependency installation timeout

Cause: large stack in `requirements.txt`.

Fix options:

1. Retry once (often succeeds on second build)
2. Use a lighter deployment branch with dashboard-only requirements
3. Move heavy training dependencies to optional extras

## Alternative: Host on Your Own VM

Use this if you want full control and avoid cloud build limits.

### On server

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
streamlit run src/dashboard/app.py --server.address 0.0.0.0 --server.port 8501
```

Then reverse-proxy with Nginx and HTTPS.

## Local Verification

```bash
streamlit run src/dashboard/app.py
```

Open:

- `http://localhost:8501`

## Notes

- This dashboard is read-only; it does not train models.
- Regenerate artifacts anytime using the existing pipeline and redeploy.
