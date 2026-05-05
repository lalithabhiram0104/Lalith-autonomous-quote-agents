import streamlit as st
import numpy as np
import joblib
import pandas as pd

st.title("Autonomous Quote Agent System")

risk_model = joblib.load("risk_agent.pkl")
conversion_model = joblib.load("conversion_agent.pkl")
encoders = joblib.load("encoders.pkl")

st.write("Enter Quote Details")


features = st.text_input(
    "Enter numeric features separated by commas"
)

if st.button("Run Quote Agents"):

    if features == "":
        st.warning("Please enter feature values")
        st.stop()

    try:
        row = np.array(list(map(float, features.split(","))))
    except:
        st.error("Invalid input format")
        st.stop()

    expected = risk_model.n_features_in_

    if len(row) != expected:
        st.error(f"Please enter {expected} feature values")
        st.stop()


    risk_prob = risk_model.predict_proba([row])[0][1]

    if risk_prob < 0.4:
        risk_tier = "Low"
    elif risk_prob < 0.7:
        risk_tier = "Medium"
    else:
        risk_tier = "High"


    bind_prob = conversion_model.predict_proba([row])[0][1]


    premium = row[-1]

    if bind_prob < 0.3:
        premium_flag = 1
        premium_advice = "Premium Too High – Suggest Discount"
    else:
        premium_flag = 0
        premium_advice = "Premium Acceptable"


    if bind_prob > 0.75 and premium_flag == 0:
        decision = "Auto Approve"
    elif bind_prob > 0.4:
        decision = "Agent Follow-Up"
    else:
        decision = "Escalate to Underwriter"


    st.subheader("Agent Outputs")

    st.write("Risk Score:", risk_prob)
    st.write("Risk Tier:", risk_tier)
    st.write("Bind Probability:", bind_prob)
    st.write("Premium Advice:", premium_advice)
    st.write("Final Decision:", decision)
