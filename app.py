"""
Streamlit frontend for the Customer Churn Prediction model.

Two modes:
1. Single Prediction - fill a form for one customer, get an instant churn prediction.
2. Batch Prediction   - upload a CSV of many customers, get predictions for the whole file
                         with a downloadable results CSV.

Run with:
    streamlit run app.py
"""

import io
import sys
import os

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipelines.predict_pipeline import CustomData, PredictPipeline
from src.exception import CustomException

REQUIRED_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉", layout="wide")

st.title("📉 Customer Churn Prediction")
st.caption("Predict whether a telecom customer is likely to churn, using a trained ML model.")

tab1, tab2 = st.tabs(["🧍 Single Customer Prediction", "📁 Batch Prediction (CSV Upload)"])

# ----------------------------------------------------------------------
# TAB 1: Single customer form
# ----------------------------------------------------------------------
with tab1:
    st.subheader("Enter customer details")

    with st.form("single_prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Has Partner", ["No", "Yes"])
            dependents = st.selectbox("Has Dependents", ["No", "Yes"])
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
            phone_service = st.selectbox("Phone Service", ["No", "Yes"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

        with col2:
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        with col3:
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
            payment_method = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            )
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=500.0, value=70.0, step=0.5)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=15000.0, value=840.0, step=1.0)

        submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True)

    if submitted:
        try:
            custom_data = CustomData(
                gender=gender,
                SeniorCitizen=1 if senior_citizen == "Yes" else 0,
                Partner=partner,
                Dependents=dependents,
                tenure=tenure,
                PhoneService=phone_service,
                MultipleLines=multiple_lines,
                InternetService=internet_service,
                OnlineSecurity=online_security,
                OnlineBackup=online_backup,
                DeviceProtection=device_protection,
                TechSupport=tech_support,
                StreamingTV=streaming_tv,
                StreamingMovies=streaming_movies,
                Contract=contract,
                PaperlessBilling=paperless_billing,
                PaymentMethod=payment_method,
                MonthlyCharges=monthly_charges,
                TotalCharges=total_charges,
            )

            input_df = custom_data.get_data_as_data_frame()
            pipeline = PredictPipeline()
            preds, probs = pipeline.predict(input_df)

            pred = int(preds[0])
            prob = probs[0]

            st.divider()
            if pred == 1:
                st.error(f"### ⚠️ This customer is likely to CHURN")
            else:
                st.success(f"### ✅ This customer is likely to STAY")

            if prob is not None:
                st.metric("Churn probability", f"{prob * 100:.1f}%")
                st.progress(min(max(prob, 0.0), 1.0))

        except CustomException as e:
            st.error(f"Prediction failed: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

# ----------------------------------------------------------------------
# TAB 2: Batch CSV upload
# ----------------------------------------------------------------------
with tab2:
    st.subheader("Upload a CSV of customers")
    st.markdown(
        "Upload a CSV file where each row is one customer. The file **must** contain "
        "the following columns (extra columns like `customerID` or `Churn` are fine and "
        "will simply be ignored):"
    )
    st.code(", ".join(REQUIRED_COLUMNS), language=None)

    with st.expander("ℹ️ Example row format"):
        example_df = pd.DataFrame([{
            "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
            "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
            "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
            "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
            "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": 29.85,
        }])
        st.dataframe(example_df, use_container_width=True)
        st.download_button(
            "Download example CSV template",
            data=example_df.to_csv(index=False).encode("utf-8"),
            file_name="churn_input_template.csv",
            mime="text/csv",
        )

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)

            missing_cols = [c for c in REQUIRED_COLUMNS if c not in raw_df.columns]
            if missing_cols:
                st.error(f"Your CSV is missing required column(s): {', '.join(missing_cols)}")
            else:
                st.write(f"Loaded **{len(raw_df)}** rows. Preview:")
                st.dataframe(raw_df.head(10), use_container_width=True)

                if st.button("🔮 Run Batch Prediction", use_container_width=True):
                    with st.spinner("Running predictions..."):
                        input_df = raw_df[REQUIRED_COLUMNS].copy()

                        # Same cleanup the training pipeline applies
                        input_df["TotalCharges"] = pd.to_numeric(input_df["TotalCharges"], errors="coerce")
                        input_df["SeniorCitizen"] = pd.to_numeric(input_df["SeniorCitizen"], errors="coerce").fillna(0).astype(int)

                        pipeline = PredictPipeline()
                        preds, probs = pipeline.predict(input_df)

                        results_df = raw_df.copy()
                        results_df["Predicted_Churn"] = ["Yes" if p == 1 else "No" for p in preds]
                        results_df["Churn_Probability"] = [
                            round(float(p), 4) if p is not None else None for p in probs
                        ]

                    st.success(f"Done! Predicted churn for {len(results_df)} customers.")

                    churn_count = (results_df["Predicted_Churn"] == "Yes").sum()
                    col_a, col_b = st.columns(2)
                    col_a.metric("Predicted to churn", int(churn_count))
                    col_b.metric("Predicted to stay", int(len(results_df) - churn_count))

                    st.dataframe(results_df, use_container_width=True)

                    csv_buffer = io.StringIO()
                    results_df.to_csv(csv_buffer, index=False)
                    st.download_button(
                        "⬇️ Download predictions as CSV",
                        data=csv_buffer.getvalue().encode("utf-8"),
                        file_name="churn_predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

        except Exception as e:
            st.error(f"Could not process file: {e}")

st.divider()
st.caption("Model trained on the Telco Customer Churn dataset. Predictions are estimates, not guarantees.")
