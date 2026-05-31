import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import plotly.graph_objects as go
import plotly.express as px

# Konfigurasi halaman
st.set_page_config(
    page_title="Battery Health Predictor",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
    .soh-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 20px;
        text-align: center;
        color: white;
    }
    .soh-critical {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
    }
    .soh-warning {
        background: linear-gradient(135deg, #f2994a 0%, #f2c94c 100%;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🔋 Battery Health Predictor</h1>
    <p>Electric Vehicle State of Health (SOH) Estimation using BiLSTM</p>
    <p style="font-size: 0.8rem;">Project SC 2026 | ANN-based Prediction</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/battery-charging.png", width=80)
    st.markdown("### About")
    st.info("""
    **Metode:** BiLSTM (Bidirectional LSTM)
    
    **Dataset:** CNR Italy EIS Dataset (Cell 08)
    
    **Fitur:** Aging cycle, SOC, R_int, OCV
    
    **Output:** State of Health (SOH) & Recommendation
    """)
    st.markdown("---")
    st.caption("© 2026 | Built for Academic Project")

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("bilstm_soh_model.h5")
    scaler_X = joblib.load("scaler_X.pkl")
    scaler_y = joblib.load("scaler_y.pkl")
    return model, scaler_X, scaler_y

# Input form
st.subheader("📝 Battery Parameters Input")

col1, col2 = st.columns(2)

with col1:
    cycle = st.number_input("Aging Cycle", min_value=0, max_value=2000, value=100, step=10)
    soc = st.number_input("State of Charge SOC (%)", min_value=0, max_value=100, value=80, step=5)

with col2:
    r_int = st.number_input("Internal Resistance R_int (%)", min_value=0, max_value=200, value=10, step=5)
    ocv = st.number_input("Open Circuit Voltage OCV (V)", min_value=3.0, max_value=4.5, value=4.1, step=0.05)

# Prediksi
if st.button("🔮 Predict Battery Health", type="primary", use_container_width=True):
    try:
        model, scaler_X, scaler_y = load_model()
        
        # Buat input sequence (perlu 10 cycle berurutan)
        # Untuk demo, kita asumsikan cycle sebelumnya adalah cycle-9 sampai cycle-1
        prev_cycles = np.linspace(cycle-9, cycle, 10).reshape(-1, 1)
        input_data = np.column_stack([prev_cycles, 
                                       np.full(10, soc), 
                                       np.full(10, r_int), 
                                       np.full(10, ocv)])
        
        input_scaled = scaler_X.transform(input_data)
        input_seq = input_scaled.reshape(1, 10, -1)
        
        pred_scaled = model.predict(input_seq, verbose=0)
        soh = scaler_y.inverse_transform(pred_scaled)[0][0]
        
        # Tentukan status
        if soh >= 90:
            status = "✅ SEHAT"
            status_desc = "Battery in excellent condition"
            recommendation = "Continue normal usage. Next service in 6 months."
            color_class = "soh-card"
        elif soh >= 70:
            status = "⚠️ WASPADA"
            status_desc = "Battery showing degradation"
            recommendation = "Schedule battery inspection and balancing service soon."
            color_class = "soh-card soh-warning"
        else:
            status = "🔴 KRITIS"
            status_desc = "Battery health critical"
            recommendation = "Battery replacement strongly recommended for safety."
            color_class = "soh-card soh-critical"
        
        # Tampilkan hasil
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="{color_class}">
                <h3>State of Health</h3>
                <h1 style="font-size: 3rem;">{soh:.1f}%</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="soh-card">
                <h3>Status</h3>
                <h2>{status}</h2>
                <p>{status_desc}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="soh-card">
                <h3>Recommendation</h3>
                <p>{recommendation}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = soh,
            title = {'text': "SOH Meter"},
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "#667eea"},
                'steps': [
                    {'range': [0, 70], 'color': "#eb3349"},
                    {'range': [70, 90], 'color': "#f2994a"},
                    {'range': [90, 100], 'color': "#11998e"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': soh
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error: {e}")
        st.info("Pastikan file model (bilstm_soh_model.h5, scaler_X.pkl, scaler_y.pkl) sudah ada di direktori yang sama.")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>Dataset: CNR Italy EIS Dataset | Model: BiLSTM | Project SC 2026</p>
</div>
""", unsafe_allow_html=True)