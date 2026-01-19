# FastAPI Deployment

REST API for the Human Activity Recognition model.

## What You Get

- **GET /** - API info
- **GET /health** - Check if model is loaded
- **GET /activities** - List all 6 activities
- **POST /predict** - Predict one sample
- **POST /predict/batch** - Predict multiple samples

## Run Locally

```bash
# From project root
cd deployment/fastapi

# Start server
uvicorn main:app --reload
```

API at: http://localhost:8000
Docs at: http://localhost:8000/docs

## Test It

```bash
# Health check
curl http://localhost:8000/health

# Get activities
curl http://localhost:8000/activities
```

Or run the test script:
```bash
python test_api.py
```

## Test Model Performance

```bash
python test_model.py
```

This shows:
- Accuracy, precision, recall
- Confusion matrix
- Feature importance
- Saves plots as PNG files

## Docker

From project root:

```bash
# Build
docker build -f Dockerfile.fastapi -t har-api .

# Run
docker run -p 8000:8000 har-api
```

## Deploy to Cloud

### Render (Free)

1. Push to GitHub
2. Go to render.com
3. New Web Service
4. Runtime: Docker
5. Dockerfile: `Dockerfile.fastapi`
6. Deploy!

Get URL like: `https://har-api.onrender.com`

### Railway

1. Go to railway.app
2. New Project → GitHub repo
3. Auto-detects Dockerfile
4. Deploy!

## Performance

On local machine:
- Single prediction: ~10-15ms
- Batch (10 samples): ~13ms
- Throughput: ~120 requests/second

## Files

- `main.py` - FastAPI app
- `test_model.py` - Test model performance
- `test_api.py` - Test API endpoints
- `requirements.txt` - Dependencies

## Notes

Model must be trained first (`python train.py` from project root).
