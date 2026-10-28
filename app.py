import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# Load saved models / objects
@st.cache_resource
def load_assets():
    try:
        model = pickle.load(open("best_model.sav", "rb"))
        churn_list = pickle.load(open("churn_list.sav", "rb"))
        return model, churn_list
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None

model, churn_list = load_assets()

st.title("📊 Telecom Customer Churn Prediction")
st.write("Enter customer attributes below to predict whether the customer will churn.")

st.sidebar.header("Customer Details")

# Input form for customer features
def user_input_features():
    tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
    contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    payment_method = st.sidebar.selectbox(
        "Payment Method", 
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )
    monthly_charges = st.sidebar.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=70.0)
    total_charges = st.sidebar.number_input("Total Charges ($)", min_value=18.0, max_value=8500.0, value=tenure * monthly_charges)
    senior_citizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
    partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
    dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
    
    data = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'Contract': contract,
        'InternetService': internet_service,
        'PaymentMethod': payment_method
    }
    return pd.DataFrame([data])

input_df = user_input_features()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Selected Input Features")
    st.dataframe(input_df)

with col2:
    st.subheader("Prediction")
    if st.button("Predict Churn Risk"):
        if model is not None:
            # Predict probability if available, otherwise class prediction
            try:
                prediction_prob = model.predict_proba(input_df)[0][1]
                st.metric(label="Churn Probability", value=f"{prediction_prob * 100:.2f}%")
                
                if prediction_prob > 0.5:
                    st.error("⚠️ High Risk of Churn!")
                else:
                    st.success("✅ Customer is likely to stay.")
            except Exception:
                prediction = model.predict(input_df)[0]
                if prediction == 1 or prediction == "Yes":
                    st.error("⚠️ Customer will Churn!")
                else:
                    st.success("✅ Customer will Not Churn.")
        else:
            st.warning("Model file not loaded. Check best_model.sav location.")