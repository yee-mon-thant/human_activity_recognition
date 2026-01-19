# Project Checklist

## Scoring Breakdown (16 points total)

### ✅ Problem Description (2 points)
- [x] README has clear problem description
- [x] Explains what we're solving and why it matters
- [x] Lists the 6 activity classes

### ✅ EDA (2 points)
- [x] Checked for missing values
- [x] Looked at min/max values
- [x] Made plots of class distribution
- [x] Checked feature distributions
- [x] Made correlation heatmap
- [x] Analyzed feature importance

### ✅ Model Training (3 points)
- [x] Trained Random Forest with different parameters
  - Tried 100 trees
  - Tried 200 trees with max_depth=30
- [x] Trained XGBoost with different configurations
  - Tried 3 different setups
  - Changed n_estimators, max_depth, learning_rate
- [x] Compared all models

### ✅ Training Script (1 point)
- [x] Created train.py
- [x] Exports the training logic from notebook
- [x] Can run with `python train.py`

### ✅ Reproducibility (1 point)
- [x] Dataset is in the repo
- [x] Can re-run notebook without errors
- [x] Can re-run train.py without errors
- [x] Clear instructions in README

### ✅ Model Deployment (1 point)
- [x] Built Streamlit web app
- [x] Can upload CSV files for predictions
- [x] Can do manual predictions
- [x] Shows results

### ✅ Dependencies (2 points)
- [x] pyproject.toml has all dependencies
- [x] requirements.txt for Streamlit deployment
- [x] README explains how to install with uv
- [x] README shows how to activate venv

### ✅ Containerization (2 points)
- [x] Dockerfile created
- [x] README shows how to build: `docker build -t har-app .`
- [x] README shows how to run: `docker run -p 8501:8501 har-app`

### ✅ Cloud Deployment (2 points)
- [x] README has Streamlit Cloud instructions
- [x] Step-by-step guide included
- [ ] TODO: Actually deploy and add URL

## Total: 15/16 points
(Will be 16/16 after deploying to Streamlit Cloud)

---

## Quick Test Commands

```bash
# 1. Setup
cd human_activity_recognition
uv sync
source .venv/bin/activate

# 2. Train model
python train.py

# 3. Test Streamlit app
cd deployment/streamlit
streamlit run app.py

# 4. (Optional) Test Docker
docker build -t har-app .
docker run -p 8501:8501 har-app
```

---

## Files Created

- `README.md` - Main documentation
- `notebook/analysis.ipynb` - EDA and model training
- `train.py` - Training script
- `deployment/streamlit/app.py` - Web app
- `Dockerfile` - Container setup
- `pyproject.toml` - Dependencies
- `data/` - Dataset folder

All requirements met! ✨
