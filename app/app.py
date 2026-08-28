import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="Cancer Risk Predictor",
    page_icon="🩺",
    layout="centered"
)

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "model_xgb_new.pkl")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")
FEATURE_NAMES_PATH = os.path.join(BASE_DIR, "feature_names.pkl")


# --- Load model and supporting files ---
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    le = joblib.load(LABEL_ENCODER_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)

    return model, le, feature_names


model, le, FEATURE_NAMES = load_artifacts()


# --- App title ---
st.title("🩺 Cancer Risk Level Predictor")

st.markdown(
    "Predict **Cancer Risk Level (Low / Medium / High)** "
    "using the trained XGBoost machine learning model."
)


# --- Prediction mode ---
option = st.radio(
    "Prediction Mode",
    ("Manual Input", "Upload CSV")
)


# --- Preprocessing ---
def preprocess_input(df):

    missing = [c for c in FEATURE_NAMES if c not in df.columns]

    if missing:
        st.warning(
            f"Missing columns detected. Filling {len(missing)} "
            f"missing columns with zeros."
        )

        for c in missing:
            df[c] = 0

    # Keep only required features
    df = df[FEATURE_NAMES].copy()

    # Convert everything to numeric
    df = df.apply(
        pd.to_numeric,
        errors="coerce"
    ).fillna(0)

    return df


# =========================================================
# MANUAL INPUT
# =========================================================

if option == "Manual Input":

    st.subheader("Enter Patient Information")

    input_data = {}

    # Create input fields
    for feature in FEATURE_NAMES:

        input_data[feature] = st.number_input(
            feature,
            value=0.0
        )

    if st.button("🔍 Predict Cancer Risk"):

        input_df = pd.DataFrame([input_data])

        X = preprocess_input(input_df)

        # Prediction
        prediction_encoded = model.predict(X)[0]

        probabilities = model.predict_proba(X)[0]

        prediction = le.inverse_transform(
            [prediction_encoded]
        )[0]

        # Result
        st.subheader("Prediction Result")

        st.success(
            f"Predicted Risk Level: **{prediction}**"
        )

        # Probability table
        probability_df = pd.DataFrame({
            "Risk Level": le.classes_,
            "Probability": probabilities
        })

        probability_df = probability_df.sort_values(
            "Probability",
            ascending=False
        ).reset_index(drop=True)

        st.subheader("Risk Probabilities")

        st.dataframe(
            probability_df,
            use_container_width=True
        )

        # High-risk probability
        high_risk_rows = probability_df[
            probability_df["Risk Level"] == "High"
        ]

        if not high_risk_rows.empty:

            high_probability = high_risk_rows[
                "Probability"
            ].iloc[0]

            st.info(
                f"Probability of High Risk: "
                f"**{high_probability:.2%}**"
            )


# =========================================================
# CSV UPLOAD
# =========================================================

else:

    st.subheader("Upload Patient Data")

    uploaded_file = st.file_uploader(
        "Upload CSV file containing patient features",
        type=["csv"]
    )

    if uploaded_file is not None:

        input_df = pd.read_csv(uploaded_file)

        st.write("Uploaded Data")

        st.dataframe(
            input_df,
            use_container_width=True
        )

        X = preprocess_input(input_df)

        # Predictions
        predictions_encoded = model.predict(X)

        probabilities = model.predict_proba(X)

        predictions = le.inverse_transform(
            predictions_encoded
        )

        # Results
        result = input_df.copy()

        result["Predicted_Risk_Level"] = predictions

        for i, class_name in enumerate(le.classes_):

            result[
                f"Probability_{class_name}"
            ] = probabilities[:, i]

        st.success("Predictions generated successfully!")

        st.subheader("Prediction Results")

        st.dataframe(
            result,
            use_container_width=True
        )

        # Download results
        csv = result.to_csv(index=False)

        st.download_button(
            label="⬇️ Download Predictions",
            data=csv,
            file_name="cancer_risk_predictions.csv",
            mime="text/csv"
        )


# --- Disclaimer ---

st.markdown("---")

st.caption(
    "⚠️ This application is an educational machine learning project "
    "and is not intended for medical diagnosis or clinical decision-making."
)

st.caption(
    "Model: XGBoost | Cancer Risk Prediction"
)
