import streamlit as st
import pandas as pd
import pickle

# -----------------------------
# Load model
# -----------------------------
with open("my_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("💳 Loan Approval Predictor")
st.write("Enter applicant details to predict approval likelihood.")

# -----------------------------
# USER INPUTS
# -----------------------------
fico = st.slider("FICO Score", 300, 850, 650)
income = st.number_input("Monthly Gross Income", 0, 20000, 5000)
housing = st.number_input("Monthly Housing Payment", 0, 10000, 1500)
loan_amt = st.number_input("Requested Loan Amount", 1000, 100000, 20000)

employment_status = st.selectbox(
    "Employment Status",
    ["full_time", "part_time", "unemployed"]
)

lender = st.selectbox(
    "Lender",
    ["A", "B", "C"]
)

bankrupt = st.selectbox(
    "Ever Bankrupt or Foreclosure?",
    [0, 1]
)

# -----------------------------
# CREATE INPUT DATAFRAME
# -----------------------------
input_data = pd.DataFrame({
    "FICO_score": [fico],
    "Monthly_Gross_Income": [income],
    "Monthly_Housing_Payment": [housing],
    "Requested_Loan_Amount": [loan_amt],
    "Employment_Status": [employment_status],
    "Lender": [lender],
    "Ever_Bankrupt_or_Foreclose": [bankrupt]
})

# -----------------------------
# ENCODE INPUT
# -----------------------------
input_encoded = pd.get_dummies(input_data, drop_first=True)

# Align columns with training
model_features = model.feature_names_in_

for col in model_features:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

input_encoded = input_encoded[model_features]

# -----------------------------
# PREDICT
# -----------------------------
if st.button("Predict"):
    prediction = model.predict(input_encoded)[0]
    prob = model.predict_proba(input_encoded)[0][1]

    if prediction == 1:
        st.success(f"✅ Approved (Probability: {prob:.2%})")
    else:
        st.error(f"❌ Denied (Probability: {prob:.2%})")