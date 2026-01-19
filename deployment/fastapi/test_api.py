"""
Test FastAPI endpoints
"""

import requests
import pandas as pd
import json
import time

# API URL
BASE_URL = "http://localhost:8000"

print("=" * 60)
print("Testing FastAPI Endpoints")
print("=" * 60)

# Test 1: Root endpoint
print("\n1. Testing root endpoint...")
try:
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"Error: {e}")

# Test 2: Health check
print("\n2. Testing health endpoint...")
try:
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"Error: {e}")

# Test 3: Get activities
print("\n3. Testing activities endpoint...")
try:
    response = requests.get(f"{BASE_URL}/activities")
    print(f"Status: {response.status_code}")
    activities = response.json()
    print(f"Total activities: {activities['total']}")
    for key, value in activities['activities'].items():
        print(f"  {key}: {value}")
except Exception as e:
    print(f"Error: {e}")

# Test 4: Single prediction with real data
print("\n4. Testing single prediction...")
try:
    # Load one sample from test data
    X_test = pd.read_csv('../../data/UCI HAR Dataset/test/X_test.txt', sep=r'\s+', header=None)
    sample = X_test.iloc[0].tolist()

    payload = {"features": sample}

    start_time = time.time()
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    elapsed = time.time() - start_time

    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Prediction: {result['activity']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Response time: {elapsed*1000:.2f}ms")
except Exception as e:
    print(f"Error: {e}")

# Test 5: Batch prediction
print("\n5. Testing batch prediction...")
try:
    # Load 10 samples
    X_test = pd.read_csv('../../data/UCI HAR Dataset/test/X_test.txt', sep=r'\s+', header=None)
    samples = X_test.iloc[:10].values.tolist()

    payload = {"samples": samples}

    start_time = time.time()
    response = requests.post(f"{BASE_URL}/predict/batch", json=payload)
    elapsed = time.time() - start_time

    print(f"Status: {response.status_code}")
    results = response.json()
    print(f"Predictions: {len(results['predictions'])}")
    print(f"Response time: {elapsed*1000:.2f}ms")
    print(f"Avg time per sample: {elapsed*1000/len(samples):.2f}ms")

    # Show predictions
    print("\nPredictions:")
    for i, pred in enumerate(results['predictions'], 1):
        print(f"  {i}. {pred['activity']:20s} (confidence: {pred['confidence']:.4f})")
except Exception as e:
    print(f"Error: {e}")

# Test 6: Error handling - wrong number of features
print("\n6. Testing error handling (wrong features)...")
try:
    payload = {"features": [0.5] * 100}  # Wrong number
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print(f"Status: {response.status_code}")
    if response.status_code != 200:
        print(f"Error message: {response.json()['detail']}")
        print("✓ Error handling works correctly")
except Exception as e:
    print(f"Error: {e}")

# Test 7: Load test (measure throughput)
print("\n7. Load test (100 predictions)...")
try:
    X_test = pd.read_csv('../../data/UCI HAR Dataset/test/X_test.txt', sep=r'\s+', header=None)

    n_requests = 100
    start_time = time.time()

    for i in range(n_requests):
        sample = X_test.iloc[i % len(X_test)].tolist()
        payload = {"features": sample}
        response = requests.post(f"{BASE_URL}/predict", json=payload)

    elapsed = time.time() - start_time

    print(f"Total time: {elapsed:.2f}s")
    print(f"Avg time per request: {elapsed*1000/n_requests:.2f}ms")
    print(f"Throughput: {n_requests/elapsed:.2f} requests/second")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 60)
print("API Testing Complete!")
print("=" * 60)
