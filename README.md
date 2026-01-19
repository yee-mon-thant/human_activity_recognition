# Human Activity Recognition Project

## Problem Description

This project uses machine learning to predict human activities from smartphone sensor data.

### Why This Matters

Smartphones have accelerometers and gyroscopes that can detect movement. By analyzing this sensor data, we can automatically recognize what activity a person is doing without them having to manually log it. This is useful for:

- **Fitness apps** - automatically track workouts
- **Health monitoring** - detect if elderly people have fallen
- **Research** - study human behavior patterns

### What We're Trying to Do

The goal is to build a model that can classify activities into 6 categories:
1. Walking
2. Walking Upstairs
3. Walking Downstairs
4. Sitting
5. Standing
6. Laying down

The model takes sensor readings (acceleration and rotation) and predicts which activity the person is performing.

## Dataset

We're using the UCI Human Activity Recognition dataset. It has data from 30 people aged 19-48 who wore a smartphone while doing different activities.

**Dataset Details:**
- Training data: 7,352 samples
- Test data: 2,947 samples
- Features: 561 (calculated from sensor readings)
- Activities: 6 different types
- All data is already cleaned and normalized

The dataset is included in this repo under `data/UCI HAR Dataset/`.

## Exploratory Data Analysis

I did EDA in the Jupyter notebook `notebook/analysis.ipynb`. Here's what I found:

### Basic Checks
- No missing values in the dataset
- All values are normalized between -1 and 1
- Classes are fairly balanced (each activity has similar number of samples)

### Important Findings
- Body acceleration features are most important for predictions
- Some features are highly correlated (which is expected for sensor data)
- Walking activities have different patterns than stationary activities (sitting, standing)

## Models

I tried multiple models to find what works best:

### Random Forest
- First tried with 100 trees - got around 92% accuracy
- Then tried 200 trees with max_depth=30 - got around 94% accuracy
- Found that body acceleration and gravity features are most important

### XGBoost
- Tried 3 different configurations
- Best config: 200 trees, max_depth=10, learning_rate=0.1
- Got around 95% accuracy

### Results
XGBoost performed slightly better than Random Forest. All models did pretty well (>90% accuracy) because the features are already well-preprocessed.

## How to Run

### Setup

1. Install uv (it's faster than pip):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Install dependencies:
```bash
uv sync
```

3. Activate environment:
```bash
source .venv/bin/activate  # On Mac/Linux
# or
.venv\Scripts\activate  # On Windows
```

### Train the Model

```bash
python train.py
```

This will train the XGBoost model and save it to `notebook/models/`.

### Run the Notebook

```bash
jupyter notebook notebook/analysis.ipynb
```

The notebook shows all the analysis and model experiments.

### Run the Web App

```bash
cd deployment/streamlit
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## Deployment

### Docker

Build and run with Docker:

```bash
# Build
docker build -t har-app .

# Run
docker run -p 8501:8501 har-app
```

Then go to http://localhost:8501

### Streamlit Cloud

To deploy on Streamlit Cloud for free:

1. Push code to GitHub:
```bash
git init
git add .
git commit -m "HAR project"
git push origin main
```

2. Go to share.streamlit.io
3. Connect your GitHub repo
4. Set main file: `deployment/streamlit/app.py`
5. Deploy!

You'll get a public URL to share.

## Project Structure

```
human_activity_recognition/
├── data/
│   └── UCI HAR Dataset/       # Dataset files
├── notebook/
│   ├── analysis.ipynb          # Main notebook with EDA and models
│   ├── models/                 # Saved models (created after training)
│   └── performance/            # Model results (created after training)
├── deployment/
│   └── streamlit/
│       ├── app.py              # Web app
│       └── requirements.txt    # Dependencies for deployment
├── train.py                    # Script to train model
├── pyproject.toml              # Dependencies
├── Dockerfile                  # For Docker deployment
└── README.md                   # This file
```

## Dependencies

All dependencies are in `pyproject.toml`. Main ones:
- pandas - data manipulation
- numpy - numerical operations
- scikit-learn - Random Forest model
- xgboost - XGBoost model
- streamlit - web app
- matplotlib, seaborn - visualizations
- jupyter - notebooks

To install: `uv sync`

## Results

The final XGBoost model achieves **~95% accuracy** on the test set.

**Confusion Matrix shows:**
- Walking activities are sometimes confused with each other (walking vs walking upstairs)
- Stationary activities (sitting, standing, laying) are well separated
- Overall the model works quite well

**Feature Importance:**
- Top features are related to body acceleration magnitude
- Gravity-related features help distinguish stationary vs moving activities

## Future Improvements

Things I could add later:
- Try neural networks (LSTM or CNN)
- Do more hyperparameter tuning
- Add real-time prediction from phone
- Deploy to cloud platform

## Notes

This project was built for a machine learning capstone. The dataset is already well-prepared which makes the ML part straightforward. The main work was:
- Understanding the data through EDA
- Trying different models
- Finding best parameters
- Building a deployment pipeline

## References

- UCI HAR Dataset: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
- XGBoost documentation: https://xgboost.readthedocs.io/
- Streamlit docs: https://docs.streamlit.io/
