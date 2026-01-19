import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Human Activity Recognition", page_icon="🏃")

st.title("🏃 Human Activity Recognition")
st.write("Predict activities from smartphone sensor data using XGBoost")

# Load model
@st.cache_resource
def load_model():
    try:
        # Try different paths
        base_paths = [
            '../../notebook/models/',  # When run from deployment/streamlit
            'notebook/models/',         # When run from project root
            './models/'                 # When deployed (Docker/Cloud)
        ]

        for base_path in base_paths:
            model_path = os.path.join(base_path, 'model.pkl')
            features_path = os.path.join(base_path, 'features.pkl')

            if os.path.exists(model_path) and os.path.exists(features_path):
                model = joblib.load(model_path)
                features = joblib.load(features_path)
                return model, features

        return None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, feature_names = load_model()

if model is None:
    st.error("⚠️ Model not found. Please run `python train.py` first.")
    st.stop()

st.success("✅ Model loaded successfully!")

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

# File upload
st.header("📤 Upload Data File")
st.write("Upload a CSV file with 561 features (one sample per row)")

uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])

if uploaded_file:
    data = pd.read_csv(uploaded_file, header=None)

    if data.shape[1] != 561:
        st.error(f"❌ Expected 561 features, got {data.shape[1]}")
    else:
        st.success(f"✅ Loaded {len(data)} samples")

        if st.button("🔮 Predict Activities", type="primary"):
            with st.spinner("Making predictions..."):
                predictions = model.predict(data.values)

                results = pd.DataFrame({
                    'Sample': range(1, len(predictions) + 1),
                    'Predicted Activity': [activities[p] for p in predictions]
                })

                st.subheader("📊 Results")
                st.dataframe(results, use_container_width=True)

                # Show counts
                st.subheader("📈 Activity Distribution")
                counts = results['Predicted Activity'].value_counts()
                st.bar_chart(counts)

                # Download
                csv = results.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv"
                )

st.write("---")

# Manual input
st.header("✍️ Manual Input")
st.write("Enter sensor values manually")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Acceleration")
    acc_x = st.slider("Acc X", -1.0, 1.0, 0.0, 0.01)
    acc_y = st.slider("Acc Y", -1.0, 1.0, 0.0, 0.01)
    acc_z = st.slider("Acc Z", -1.0, 1.0, 0.0, 0.01)

with col2:
    st.subheader("Gyroscope")
    gyro_x = st.slider("Gyro X", -1.0, 1.0, 0.0, 0.01)
    gyro_y = st.slider("Gyro Y", -1.0, 1.0, 0.0, 0.01)
    gyro_z = st.slider("Gyro Z", -1.0, 1.0, 0.0, 0.01)

if st.button("🔮 Predict Activity"):
    # Create feature array
    input_data = np.zeros(561)
    input_data[0] = acc_x
    input_data[1] = acc_y
    input_data[2] = acc_z
    input_data[120] = gyro_x
    input_data[121] = gyro_y
    input_data[122] = gyro_z

    prediction = model.predict([input_data])[0]
    proba = model.predict_proba([input_data])[0]

    st.success(f"### Predicted: **{activities[prediction]}**")
    st.write(f"Confidence: **{proba[prediction]:.2%}**")

    # Show all probabilities
    st.subheader("All Probabilities")
    prob_data = pd.DataFrame({
        'Activity': [activities[i] for i in range(6)],
        'Probability': proba
    }).sort_values('Probability', ascending=False)

    st.dataframe(prob_data, use_container_width=True)

st.write("---")
st.caption("Built with XGBoost and Streamlit | Model Accuracy: ~93%")
