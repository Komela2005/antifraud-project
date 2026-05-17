import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -----------------------------
# PAGE TITLE
# -----------------------------

st.title("Fraud Detection System")

# -----------------------------
# SIDEBAR
# -----------------------------

st.sidebar.header("Settings")

sample_size = st.sidebar.slider(
    "Sample size",
    min_value=100,
    max_value=10000,
    value=1000,
    step=100
)

selected_model = st.sidebar.selectbox(
    "Choose model",
    [
        "logistic_regression",
        "random_forest_v1",
        "random_forest_v2"
    ]
)

compare_button = st.sidebar.button("Сравнить")

# -----------------------------
# SHOW SETTINGS
# -----------------------------

st.write(f"Selected sample size: {sample_size}")
st.write(f"Selected model: {selected_model}")

# -----------------------------
# LOAD MODEL
# -----------------------------

if compare_button:

    try:

        model_path = Path("models") / f"{selected_model}.pkl"

        model = joblib.load(model_path)

        st.success(f"Model loaded: {selected_model}")

        # Example metrics
        metrics_df = pd.DataFrame({
            "Metric": [
                "Precision",
                "Recall",
                "F1"
            ],
            "Value": [
                0.91,
                0.89,
                0.90
            ]
        })

        st.subheader("Metrics Table")

        st.dataframe(metrics_df)

    except FileNotFoundError:

        st.error("Model file not found")

    except Exception as e:

        st.error(f"Error loading model: {e}")