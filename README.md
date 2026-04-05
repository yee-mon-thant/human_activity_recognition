# Human Activity Recognition Project

## What is This?

This project uses machine learning (XGBoost) to predict what activity someone is doing based on their smartphone sensor data.

**The 6 activities:**
1. Walking
2. Walking Upstairs
3. Walking Downstairs
4. Sitting
5. Standing
6. Laying down

## Why This is Useful

- Fitness apps can automatically track your workouts
- Health apps can detect if elderly people have fallen
- Phones can adapt to what you're doing (like turning off notifications when you're exercising)

## Dataset

Using the UCI HAR dataset - 30 people wore smartphones and did different activities while sensors recorded their movement.

- 7,352 training samples
- 2,947 test samples
- 561 features from accelerometer and gyroscope
- Data is in `data/UCI HAR Dataset/`

## What I Did

### 1. Explored the Data (EDA)

See `notebook/analysis.ipynb` for details. Found:
- No missing data (good!)
- All values normalized between -1 and 1
- Classes are balanced
- Body acceleration features are most important

### 2. Trained Models

Tried a few different models:
- **Random Forest** (100 trees): ~92% accuracy
- **Random Forest** (200 trees): ~94% accuracy
- **XGBoost** (100 trees): ~93% accuracy
- **XGBoost** (200 trees): **~93% accuracy** ✅ (this is what we're using)

XGBoost worked best overall!

### 3. Built Deployments

Created two ways to use the model:

**Option 1: Streamlit App** (easier, has a UI)
- Upload CSV files or use sliders
- See predictions with charts
- Good for demos

**Option 2: FastAPI** (for developers)
- REST API with endpoints
- Can integrate into other apps
- Good for production

## How to Run This

### Setup

```bash
# Install uv (fast package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Activate environment
source .venv/bin/activate  # Mac/Linux
# or
.venv\Scripts\activate  # Windows
```

### Train the Model

```bash
python train.py
```

This takes about 2-3 minutes and saves the model to `notebook/models/`.

### Run the Streamlit App (Local)

```bash
streamlit run deployment/streamlit/app.py
```

Then open http://localhost:8501 in your browser.

### Run the FastAPI (Local)

```bash
cd deployment/fastapi
uvicorn main:app --reload
```

API will be at http://localhost:8000
- Docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

## Deploy to Cloud

### Streamlit (Easiest!)

1. Push to GitHub:
   ```bash
   git add .
   git commit -m "Ready to deploy"
   git push origin main
   ```

2. Go to https://share.streamlit.io

3. New app → Select your repo

4. Main file: `deployment/streamlit/app.py`

5. Click Deploy!

You'll get a URL like: `https://your-app.streamlit.app`

### FastAPI (Docker)

1. Build Docker image:
   ```bash
   docker build -f Dockerfile.fastapi -t har-api .
   ```

2. Run locally:
   ```bash
   docker run -p 8000:8000 har-api
   ```

3. Deploy to Render.com (free):
   - Go to render.com
   - New Web Service
   - Connect GitHub repo
   - Runtime: Docker
   - Dockerfile: `Dockerfile.fastapi`
   - Deploy!

## Project Structure

```
human_activity_recognition/
├── data/                      # Dataset
├── notebook/
│   ├── analysis.ipynb         # EDA and model experiments
│   ├── models/                # Trained models (created by train.py)
│   └── performance/           # Model results
├── deployment/
│   ├── streamlit/
│   │   ├── app.py            # Streamlit web app
│   │   └── requirements.txt
│   └── fastapi/
│       ├── main.py           # FastAPI REST API
│       ├── test_model.py     # Test model performance
│       ├── test_api.py       # Test API endpoints
│       └── requirements.txt
├── train.py                   # Train the model
├── pyproject.toml            # Dependencies (uv)
├── Dockerfile.fastapi        # Docker for FastAPI
└── README.md                 # This file
```

## Results

**Model Performance:**
- Overall Accuracy: **92.84%**
- Best at detecting: LAYING (100%), WALKING (97.6%)
- Gets confused: SITTING vs STANDING (82%)

**Per Activity:**
- WALKING: 97.6%
- WALKING_UPSTAIRS: 91.5%
- WALKING_DOWNSTAIRS: 91.4%
- SITTING: 82.5%
- STANDING: 93.1%
- LAYING: 100%

Pretty good! The model sometimes mixes up SITTING and STANDING because they look similar in the sensor data.

## Tools Used

- **XGBoost** - main model
- **pandas, numpy** - data handling
- **scikit-learn** - metrics and preprocessing
- **matplotlib, seaborn** - visualizations
- **Streamlit** - web app UI
- **FastAPI** - REST API
- **uv** - fast dependency manager
- **Docker** - containerization

## What Could Be Better

- Try LSTM neural networks (might work better with time-series data)
- Add more activities (running, cycling, etc.)
- Get real-time predictions from phone
- Deploy with CI/CD pipeline

## Notes

This was a capstone project. The dataset is already well-prepared so the hard work was:
- Understanding what features matter
- Trying different models
- Building a working deployment

Model is saved as `notebook/models/model.pkl` and can be loaded with joblib.

## Links

- Dataset: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
- XGBoost: https://xgboost.readthedocs.io/
- Streamlit: https://docs.streamlit.io/
- FastAPI: https://fastapi.tiangolo.com/

---

**Questions?** Open an issue or reach out!
