import streamlit as st
import numpy as np
import joblib

# -------------------------------
# Load Files
# -------------------------------
model = joblib.load("logistic_regression_model.pkl")
scaler = joblib.load("scaler.pkl")

# -------------------------------
# Title
# -------------------------------
st.title("🏦 Loan Prediction App")

st.write("Fill the details below to predict Loan Status")

# -------------------------------
# User Inputs
# -------------------------------

Dependents = st.selectbox("Dependents", [0, 1, 2, 3])
ApplicantIncome = st.number_input("Applicant Income", min_value=0)
CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0)
LoanAmount = st.number_input("Loan Amount", min_value=0)
Loan_Amount_Term = st.number_input("Loan Term", min_value=0)
Credit_History = st.selectbox("Credit History", [0, 1])

# Derived feature
Total_Income = ApplicantIncome + CoapplicantIncome

# Categorical (One Hot)
Gender = st.selectbox("Gender", ["Male", "Female"])
Married = st.selectbox("Married", ["Yes", "No"])
Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# -------------------------------
# Encoding (IMPORTANT)
# -------------------------------
Gender_Female = 1 if Gender == "Female" else 0
Gender_Male = 1 if Gender == "Male" else 0

Married_Yes = 1 if Married == "Yes" else 0
Married_No = 1 if Married == "No" else 0

Education_Graduate = 1 if Education == "Graduate" else 0
Education_Not_Graduate = 1 if Education == "Not Graduate" else 0

Self_Employed_Yes = 1 if Self_Employed == "Yes" else 0
Self_Employed_No = 1 if Self_Employed == "No" else 0

Property_Area_Urban = 1 if Property_Area == "Urban" else 0
Property_Area_Semiurban = 1 if Property_Area == "Semiurban" else 0
Property_Area_Rural = 1 if Property_Area == "Rural" else 0

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("Predict"):

    input_data = np.array([[
        Dependents,
        ApplicantIncome,
        CoapplicantIncome,
        LoanAmount,
        Loan_Amount_Term,
        Credit_History,
        Total_Income,
        Gender_Female,
        Gender_Male,
        Married_No,
        Married_Yes,
        Education_Graduate,
        Education_Not_Graduate,
        Self_Employed_No,
        Self_Employed_Yes,
        Property_Area_Rural,
        Property_Area_Semiurban,
        Property_Area_Urban
    ]])

    # Scaling
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Not Approved")