import streamlit as st
import numpy as np
import pickle

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")

# -------------------------------
# TITLE & HEADER
# -------------------------------
st.title("🏦 Loan Risk Evaluation Dashboard")
st.caption("Model: Random Forest | Dataset: UCI Credit Card Default")

st.markdown("""
Evaluate customer creditworthiness using behavioral and financial indicators.  
This tool helps in **data-driven lending decisions**.
""")

# -------------------------------
# LOAD MODEL
# -------------------------------
try:
    model = pickle.load(open('model.pkl', 'rb'))
    st.success("✅ Model loaded successfully")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# -------------------------------
# LAYOUT (2 COLUMNS)
# -------------------------------
col1, col2 = st.columns(2)

# -------------------------------
# INPUT SECTION
# -------------------------------
with col1:
    st.subheader("📥 Customer Inputs")

    limit_bal = st.number_input("💰 Credit Limit", min_value=1000, value=50000)
    age = st.number_input("🎂 Age", min_value=18, value=30)

    pay_0 = st.selectbox(
        "📊 Recent Repayment Status (PAY_0)",
        [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8],
        help="-1 = paid on time, higher values = more delay"
    )

    bill_amt1 = st.number_input("🧾 Last Bill Amount", min_value=0, value=20000)

    predict_btn = st.button("🔍 Predict Risk")

# -------------------------------
# OUTPUT SECTION
# -------------------------------
with col2:
    st.subheader("📊 Prediction Result")

    if predict_btn:
        try:
            data = np.array([[limit_bal, age, pay_0, bill_amt1]])
            prob = model.predict_proba(data)[0][1]

            st.metric("Default Probability", f"{prob:.2f}")

            if prob < 0.3:
                st.success("🟢 Low Risk → Loan Approved")
            elif prob < 0.6:
                st.warning("🟡 Medium Risk → Manual Review Required")
            else:
                st.error("🔴 High Risk → Loan Rejected")

            # -------------------------------
            # INSIGHT
            # -------------------------------
            st.markdown("### 📊 Key Insight")
            st.info(
                "Customers with higher repayment delays (PAY_0) tend to have a significantly higher probability of default."
            )

        except Exception as e:
            st.error(f"❌ Prediction error: {e}")

# -------------------------------
# FOOTER / DISCLAIMER
# -------------------------------
st.markdown("---")
st.info(
    "⚠️ Note: This is a decision-support tool. Final lending decisions should include additional financial and behavioral assessments."
)