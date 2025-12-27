import streamlit as st
import pandas as pd
import requests, json
from typing import List, Dict, Any

st.set_page_config(page_title="loan Default Prediction UI", layout="wide")

st.title("loan Default value Prediction - Streamlit UI")
st.markdown("Interactively prepare input records and call the model API `/predict`. Configure the API URL below.")

api_url = st.text_input("Model API URL", value="http://51.20.144.159/predict")

st.markdown("### Single record input")
with st.form("single_form"):
    loan_id = st.number_input('loan_id', value=1)
    age = st.number_input('age', value=24)
    annual_income = st.number_input('annual_income', value=2.0)
    employment_length = st.number_input('employment_length', value = 3.0)
    home_ownership = st.selectbox('home_ownership', options=["MORTGAGE", "OWN", "RENT"])
    purpose = st.selectbox('purpose', options=["debt_consolidation", "credit_card", "home_improvement"])
    loan_amount = st.number_input('loan_amount', value=3.5)
    term_months= st.number_input('term_months', value=0)
    interest_rate= st.number_input('interest_rate', value=3443)
    dti= st.number_input('dti', value=2.4)
    credit_score= st.number_input('credit_score', value = 647.5)
    delinquency_2yrs=st.number_input('delinquency_2yrs', value = 1)
    num_open_acc= st.number_input('num_open_acc', value =12 )
    # --- SUBMIT BUTTON MUST BE HERE ---
    submitted = st.form_submit_button("Predict Loan Default Risk")
    if submitted:
        record ={
        "loan_id": int(loan_id),
        "age": int(age),
        "annual_income": float(annual_income),
        "employment_length": float(employment_length),
        "home_ownership": home_ownership,
        "purpose": purpose,
        "loan_amount": float(loan_amount),
        "term_months": int(term_months),
        "interest_rate": float(interest_rate),
        "dti": float(dti),
        "credit_score": int(credit_score),
        "delinquency_2yrs": int(delinquency_2yrs),
        "num_open_acc": int(num_open_acc)
        }
        payload = {"records": [record]}
        try:
            resp = requests.post(api_url, json=payload, timeout=10)
            resp.raise_for_status()
            st.success("Prediction successful")
            st.json(resp.json())
        except Exception as e:
            st.error(f"Request failed: {e}")

st.markdown("---")
st.markdown("### Batch input (upload CSV)")
uploaded = st.file_uploader("Upload CSV with same columns as training (exclude target)", type=["csv"])
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        st.write("Preview of uploaded data:")
        st.dataframe(df.head())
        if st.button("Predict batch"):
            payload = {"records": df.to_dict(orient="records")}
            try:
                resp = requests.post(api_url, json=payload, timeout=30)
                resp.raise_for_status()
                st.success("Batch prediction successful")
                res = resp.json()
                preds = res.get("predictions", None)
                if preds is not None and len(preds) == len(df):
                    df['prediction'] = preds
                    st.dataframe(df)
                else:
                    st.write(res)
            except Exception as e:
                st.error(f"Batch request failed: {e}")
    except Exception as e:
        st.error(f"Could not read uploaded CSV: {e}")

st.markdown("---")
st.markdown("### Quick tips")
st.markdown("- Make sure the API URL is reachable and points to the `/predict` endpoint of your model server.")
st.markdown("- The uploaded CSV should have the same feature columns (names & dtypes) as used in training.")
