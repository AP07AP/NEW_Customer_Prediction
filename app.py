import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier

# Load the saved model
model = pickle.load(open('final_model.pkl', 'rb'))

# Streamlit user input
st.title("Customer Prediction")

# Streamlit settings
st.set_option('deprecation.showPyplotGlobalUse', False)
st.markdown('<style>body{background-color: Blue;}</style>', unsafe_allow_html=True)
st.title("Customer Segmentation")

# Cluster mapping
cluster_mapping = {
    0: "Moderate to High Value Customers with Cash Advance Usage",
    1: "High Spending Customers with Premium Credit",
    2: "Moderate Spenders with Conservative Credit Use",
    3: "Low Spenders with Moderate Cash Advance Usage"
}

# Input form for user data
with st.form("my_form"):
    balance = st.number_input(label='Balance', value=random.uniform(1000, 5000), step=0.001, format="%.6f")
    balance_frequency = st.number_input(label='Balance Frequency', value=random.uniform(0, 1), step=0.001, format="%.6f")
    purchases = st.number_input(label='Purchases', value=random.uniform(100, 1000), step=0.01, format="%.2f")
    oneoff_purchases = st.number_input(label='OneOff Purchases', value=random.uniform(50, 500), step=0.01, format="%.2f")
    installments_purchases = st.number_input(label='Installments Purchases', value=random.uniform(50, 500), step=0.01, format="%.2f")
    cash_advance = st.number_input(label='Cash Advance', value=random.uniform(0, 1000), step=0.01, format="%.6f")
    purchases_frequency = st.number_input(label='Purchases Frequency', value=random.uniform(0, 1), step=0.01, format="%.6f")
    oneoff_purchases_frequency = st.number_input(label='OneOff Purchases Frequency', value=random.uniform(0, 1), step=0.1, format="%.6f")
    purchases_installment_frequency = st.number_input(label='Purchases Installments Frequency', value=random.uniform(0, 1), step=0.1, format="%.6f")
    cash_advance_frequency = st.number_input(label='Cash Advance Frequency', value=random.uniform(0, 1), step=0.1, format="%.6f")
    cash_advance_trx = st.number_input(label='Cash Advance Trx', value=random.randint(0, 10), step=1)
    purchases_trx = st.number_input(label='Purchases TRX', value=random.randint(0, 50), step=1)
    credit_limit = st.number_input(label='Credit Limit', value=random.uniform(1000, 10000), step=0.1, format="%.1f")
    payments = st.number_input(label='Payments', value=random.uniform(0, 5000), step=0.01, format="%.6f")
    minimum_payments = st.number_input(label='Minimum Payments', value=random.uniform(0, 2000), step=0.01, format="%.6f")
    prc_full_payment = st.number_input(label='PRC Full Payment', value=random.uniform(0, 1), step=0.01, format="%.6f")
    tenure = st.number_input(label='Tenure', value=random.randint(6, 12), step=1)

    # Collect the input data into a DataFrame with the correct feature names
    feature_columns = [
        'BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE',
        'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY',
        'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT', 'TENURE'
    ]
    
    data = pd.DataFrame([[balance, balance_frequency, purchases, oneoff_purchases, installments_purchases, cash_advance, 
                          purchases_frequency, oneoff_purchases_frequency, purchases_installment_frequency, cash_advance_frequency, 
                          cash_advance_trx, purchases_trx, credit_limit, payments, minimum_payments, prc_full_payment, tenure]], 
                        columns=feature_columns)

    submitted = st.form_submit_button("Submit")

# Make predictions and display results
if submitted:
    clust = loaded_model.predict(data)[0]
    st.write(f'Data belongs to Cluster: {clust} - {cluster_mapping[clust]}')
