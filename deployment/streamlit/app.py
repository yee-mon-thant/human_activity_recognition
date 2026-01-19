import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("Human Activity Recognition")
st.write("Predict activities from smartphone sensor data")

# Try to load model
try:
    model = joblib.load('../../notebook/models/model.pkl')
    features = joblib.load('../../notebook/models/features.pkl')
    st.success("Model loaded successfully!")
except:
    st.error("Model not found. Please run train.py first")
    st.stop()

# Activity names
activities = {
    0: 'WALKING',
    1: 'WALKING_UPSTAIRS',
    2: 'WALKING_DOWNSTAIRS',
    3: 'SITTING',
    4: 'STANDING',
    5: 'LAYING'
}

st.write("---")

# File upload option
st.header("Upload Data File")
st.write("Upload a CSV file with 561 features")

uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, header=None)

    if data.shape[1] != 561:
        st.error(f"Wrong number of features. Expected 561, got {data.shape[1]}")
    else:
        data.columns = features
        st.write(f"Loaded {len(data)} samples")

        if st.button("Predict"):
            predictions = model.predict(data)

            results = pd.DataFrame({
                'Sample': range(1, len(predictions) + 1),
                'Activity': [activities[p] for p in predictions]
            })

            st.write("### Results")
            st.dataframe(results)

            # Show counts
            st.write("### Activity Counts")
            counts = pd.Series([activities[p] for p in predictions]).value_counts()
            st.bar_chart(counts)

            # Download button
            csv = results.to_csv(index=False)
            st.download_button(
                "Download Results",
                csv,
                "predictions.csv",
                "text/csv"
            )

st.write("---")

# Manual input option
st.header("Manual Input")
st.write("Enter some key values (others will be set to 0)")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Acceleration")
    acc_x = st.slider("Acc X", -1.0, 1.0, 0.0)
    acc_y = st.slider("Acc Y", -1.0, 1.0, 0.0)
    acc_z = st.slider("Acc Z", -1.0, 1.0, 0.0)

with col2:
    st.subheader("Gyroscope")
    gyro_x = st.slider("Gyro X", -1.0, 1.0, 0.0)
    gyro_y = st.slider("Gyro Y", -1.0, 1.0, 0.0)
    gyro_z = st.slider("Gyro Z", -1.0, 1.0, 0.0)

if st.button("Predict Activity"):
    # Create feature array
    input_data = np.zeros(561)
    input_data[0] = acc_x
    input_data[1] = acc_y
    input_data[2] = acc_z
    input_data[120] = gyro_x
    input_data[121] = gyro_y
    input_data[122] = gyro_z

    # Make prediction
    input_df = pd.DataFrame([input_data], columns=features)
    prediction = model.predict(input_df)[0]

    st.success(f"Predicted Activity: **{activities[prediction]}**")

st.write("---")
st.write("Built with Streamlit and XGBoost")
