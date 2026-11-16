import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go

# ==========================================
# 1. Page Configuration (Wide layout)
# ==========================================
st.set_page_config(
    page_title="HTG-AP Severity Predictor",
    page_icon="🏥",
    layout="wide"
)

# ==========================================
# 2. Cache Model
# ==========================================
@st.cache_resource
def load_model():
    try:
        model = joblib.load('sap_rf_model.pkl')
        scaler = joblib.load('sap_scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please ensure 'sap_rf_model.pkl' and 'sap_scaler.pkl' are in the same directory.")
        st.stop()

rf_model, scaler = load_model()

# The internal cut-off is kept for logic, but hidden from text display
CUTOFF_RISK = 0.235 

# ==========================================
# 3. Main UI Layout (Left: Inputs, Right: Results)
# ==========================================
st.title("🏥 Hypertriglyceridemic Acute Pancreatitis (HTG-AP) Severity Predictor")
st.markdown("An early clinical risk stratification tool designed to predict the development of severe acute pancreatitis in patients with hypertriglyceridemia.")
st.divider()

col_input, col_result = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("📋 Patient Baseline Characteristics")
    
    in_col1, in_col2 = st.columns(2)
    with in_col1:
        cr = st.number_input("Creatinine (μmol/L)", min_value=0.0, max_value=2000.0, value=60.0, step=1.0)
        bun = st.number_input("BUN (mmol/L)", min_value=0.0, max_value=100.0, value=4.5, step=0.1)
        ldh = st.number_input("LDH (U/L)", min_value=0.0, max_value=5000.0, value=250.0, step=1.0)
        hdl = st.number_input("HDL-C (mmol/L)", min_value=0.0, max_value=10.0, value=1.0, step=0.01)
    
    with in_col2:
        ast = st.number_input("AST (U/L)", min_value=0.0, max_value=5000.0, value=25.0, step=1.0)
        ascites_input = st.selectbox("Ascites", ["No", "Yes"])
        npar_tyg = st.number_input("NPAR×TyG Index", min_value=0.0, max_value=100.0, value=8.0, step=0.1)
    
    ascites_val = 1 if ascites_input == "Yes" else 0
    
    st.write("") 
    predict_btn = st.button("🚀 Predict Severity Risk", use_container_width=True, type="primary")

with col_result:
    st.subheader("📊 Risk Assessment Result")
    
    if predict_btn:
        features = np.array([[cr, bun, ldh, hdl, ast, ascites_val, npar_tyg]])
        features_scaled = scaler.transform(features)
        
        risk_prob = rf_model.predict_proba(features_scaled)[0][1]
        is_high_risk = risk_prob >= CUTOFF_RISK
        
        if is_high_risk:
            st.error("**High Risk**: The patient has a high probability of developing severe hypertriglyceridemic acute pancreatitis. Close clinical monitoring or ICU admission is highly recommended.")
        else:
            st.success("**Low Risk**: The patient currently has a low risk of progressing to severe hypertriglyceridemic acute pancreatitis.")

        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_prob * 100,
            number = {'suffix': "%", 'valueformat': ".1f"},
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Predicted Probability of Severe Disease", 'font': {'size': 18}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "#E9392B" if is_high_risk else "#75A025"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, CUTOFF_RISK * 100], 'color': 'rgba(117, 160, 37, 0.15)'}, 
                    {'range': [CUTOFF_RISK * 100, 100], 'color': 'rgba(233, 57, 43, 0.15)'}],
                'threshold': {
                    'line': {'color': "black", 'width': 3},
                    'thickness': 0.75,
                    'value': CUTOFF_RISK * 100}
            }
        ))
        
        fig.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👈 Please enter patient baseline characteristics on the left and click 'Predict Severity Risk' to view the results.")
