# Dataset Information

## UCI HAR Dataset

This folder contains the Human Activity Recognition dataset from UCI.

### What's in the dataset:

- **30 volunteers** aged 19-48
- **6 activities**: Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, Laying
- **Sensors**: Accelerometer and Gyroscope from Samsung Galaxy S II
- **Sampling rate**: 50Hz

### Files:

- `features.txt` - List of all 561 features
- `activity_labels.txt` - Activity names (1-6)
- `train/` - Training data (70% of volunteers)
  - `X_train.txt` - Feature values (7352 rows x 561 columns)
  - `y_train.txt` - Activity labels
  - `subject_train.txt` - Which volunteer each sample is from
- `test/` - Test data (30% of volunteers)
  - Same structure as train folder

### About the features:

The 561 features come from sensor readings that were processed with filters and calculations. They include:
- Mean, standard deviation, max, min of sensor values
- Time domain and frequency domain features
- Body and gravity acceleration components
- Jerk signals (sudden movements)
- Magnitudes and angles

All values are normalized to be between -1 and 1.

### Source:

UCI Machine Learning Repository
https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
