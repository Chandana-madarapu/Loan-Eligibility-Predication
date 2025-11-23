import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------
# Load and preprocess data
# ---------------------------------------------
train = pd.read_csv("loan-train.csv")
test = pd.read_csv("loan-test.csv")

# Handle missing values
train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
train['Married'].fillna(train['Married'].mode()[0], inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
train['LoanAmount'].fillna(train['LoanAmount'].mean(), inplace=True)

# Encode categorical variables
train = train.replace({
    'Loan_Status': {'Y': 1, 'N': 0},
    'Gender': {'Male': 1, 'Female': 0},
    'Married': {'Yes': 1, 'No': 0},
    'Self_Employed': {'Yes': 1, 'No': 0}
})

# Select features and target variable
X = train[['Gender', 'Married', 'Dependents', 'Credit_History', 'LoanAmount']]
y = train['Loan_Status']

# Train logistic regression model
model = LogisticRegression()
model.fit(X, y)

# ---------------------------------------------
# Streamlit user interface
# ---------------------------------------------
st.title("Loan Eligibility Prediction")

st.write("This application predicts whether a loan applicant is eligible for loan approval based on basic personal and financial information.")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
marital_status = st.selectbox("Marital Status", ["Yes", "No"])
dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
credit_history = st.selectbox("Credit History", [0, 1])
loan_amount = st.number_input("Loan Amount", value=120.0)

# Convert inputs to numerical form
gender_value = 1 if gender == "Male" else 0
marital_value = 1 if marital_status == "Yes" else 0

# Prediction
if st.button("Predict Loan Status"):
    input_data = [[gender_value, marital_value, dependents, credit_history, loan_amount]]
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("Loan Approved")
    else:
        st.error("Loan Not Approved")











