import streamlit as st
import pandas as pd
import pickle
import os

# Load model
model = pickle.load(open("model/loan_model.pkl", "rb"))

st.title("🏦 Loan Approval Prediction System")

st.write("Enter applicant details to predict loan approval.")

# User Inputs
no_of_dependents = st.number_input("Number of Dependents", min_value=0)
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
income_annum = st.number_input("Annual Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.number_input("Loan Term (months)", min_value=0)
cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900)
residential_assets_value = st.number_input("Residential Assets Value", min_value=0)
commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0)
luxury_assets_value = st.number_input("Luxury Assets Value", min_value=0)
bank_asset_value = st.number_input("Bank Asset Value", min_value=0)

# Encode inputs manually
education_val = 1 if education == "Graduate" else 0
self_employed_val = 1 if self_employed == "Yes" else 0

input_data = pd.DataFrame({
    "no_of_dependents": [no_of_dependents],
    "education": [education_val],
    "self_employed": [self_employed_val],
    "income_annum": [income_annum],
    "loan_amount": [loan_amount],
    "loan_term": [loan_term],
    "cibil_score": [cibil_score],
    "residential_assets_value": [residential_assets_value],
    "commercial_assets_value": [commercial_assets_value],
    "luxury_assets_value": [luxury_assets_value],
    "bank_asset_value": [bank_asset_value]
})

if st.button("Predict Loan Status"):
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    # Feedback Loop
    feedback = st.radio(
        "Was this prediction correct?",
        ["Yes", "No"]
    )

    if feedback == "No":
        correct_label = st.selectbox(
            "What should be the correct outcome?",
            ["Approved", "Rejected"]
        )

        feedback_data = input_data.copy()
        feedback_data["correct_label"] = 1 if correct_label == "Approved" else 0

        os.makedirs("data", exist_ok=True)

        feedback_data.to_csv(
            "data/feedback_data.csv",
            mode="a",
            header=not os.path.exists("data/feedback_data.csv"),
            index=False
        )

        st.success("Feedback saved. Model can be improved using this data.")
