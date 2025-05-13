import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model with error handling
try:
    model = joblib.load("fraud_detection_model.pkl")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

st.title("Fraud Detection App")
st.markdown("Please enter the transaction details and use the predict button")
st.divider()

# Input widgets
transaction_type = st.selectbox(
    "Transaction Type",
    [
        "PAYMENT",
        "TRANSFER",
        "CASH_OUT",
        "CASH_IN",
        "DEBIT",
    ],  # Make sure these match your model's categories
)
amount = st.number_input("Amount", min_value=0.0, value=1000.0)
oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0, value=10000.0)
newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0, value=9000.0)
oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0, value=0.0)
newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0, value=0.0)

if st.button("Predict"):
    # Create DataFrame with correct feature names and order
    input_data = pd.DataFrame(
        {
            "type": [transaction_type],
            "amount": [amount],
            "oldbalanceOrg": [oldbalanceOrg],
            "newbalanceOrig": [newbalanceOrig],
            "oldbalanceDest": [oldbalanceDest],
            "newbalanceDest": [newbalanceDest],
        }
    )

    # Make prediction
    try:
        # Convert to float to avoid numpy type issues
        input_data = input_data.astype(float, errors="ignore")

        # Get prediction probabilities
        proba = model.predict_proba(input_data)[0]
        prediction = model.predict(input_data)[0]

        st.subheader("Prediction Results")

        # Display probability scores
        st.metric(label="Probability of Fraud", value=f"{proba[1]*100:.2f}%")
        st.metric(label="Probability of Legitimate", value=f"{proba[0]*100:.2f}%")

        if prediction == 1:
            st.error("⚠️ Warning: Transaction is likely fraudulent")
            st.warning(
                "Recommended action: Review transaction manually or block temporarily"
            )
        else:
            st.success("✅ Transaction appears legitimate")
            st.info("Note: Always verify unusual transactions regardless of prediction")

        # Add explanation
        with st.expander("How to interpret these results"):
            st.markdown(
                """
            - **Fraud Probability > 70%**: High risk - immediate review recommended
            - **Fraud Probability 30-70%**: Moderate risk - additional verification suggested
            - **Fraud Probability < 30%**: Low risk - likely legitimate
            """
            )

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        st.error("Please check your input values and try again")
