import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Battery Health Predictor",
    page_icon="🔋",
    layout="wide"
)

# ==================== HEADER ====================
st.title("🔋 Battery Health Predictor")
st.markdown("### Prediksi State of Health (SOH) Baterai Mobil Listrik")
st.markdown("Menggunakan **Artificial Neural Network (ANN) - MLPRegressor**")
st.markdown("---")

# ==================== SIDEBAR INFO ====================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/battery-charging.png", width=80)
    st.markdown("### Tentang Aplikasi")
    st.info("""
    **Metode:** ANN (MLPRegressor)
    
    **Dataset:** CNR Italy EIS Dataset (Cell 08, 2026)
    
    **Fitur:** Aging cycle, SOC, R_int, OCV
    
    **Output:** State of Health (SOH) & Rekomendasi
    """)
    st.markdown("---")
    st.caption("Project SC 2026 | ANN-based Prediction")

# ==================== LOAD MODEL ====================
@st.cache_resource
def load_model():
    model = joblib.load('model_soh.pkl')
    scaler_X = joblib.load('scaler_X.pkl')
    scaler_y = joblib.load('scaler_y.pkl')
    return model, scaler_X, scaler_y

try:
    model, scaler_X, scaler_y = load_model()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.info("Pastikan file model_soh.pkl, scaler_X.pkl, scaler_y.pkl ada di direktori yang sama.")
    st.stop()

# ==================== INPUT FORM ====================
st.subheader("📝 Battery Parameters Input")

col1, col2 = st.columns(2)

with col1:
    cycle = st.number_input(
        "🔄 Aging Cycle", 
        min_value=0, 
        max_value=2000, 
        value=100, 
        step=10,
        help="Jumlah siklus charge-discharge yang sudah dilalui baterai"
    )
    
    soc = st.number_input(
        "🔋 SOC (%)", 
        min_value=0, 
        max_value=100, 
        value=80, 
        step=5,
        help="State of Charge - Level persentase pengisian baterai saat ini"
    )

with col2:
    r_int = st.number_input(
        "⚡ R_int (%)", 
        min_value=0, 
        max_value=200, 
        value=100, 
        step=5,
        help="Internal Resistance - Resistansi internal baterai (100% = normal)"
    )
    
    ocv = st.number_input(
        "🔌 OCV (V)", 
        min_value=3.0, 
        max_value=4.5, 
        value=4.15, 
        step=0.05,
        help="Open Circuit Voltage - Tegangan baterai saat tidak dibebani"
    )

# ==================== PREDICT BUTTON ====================
if st.button("🔮 Predict Battery Health", type="primary", use_container_width=True):
    # Prediksi
    input_data = np.array([[cycle, soc, r_int, ocv]])
    input_scaled = scaler_X.transform(input_data)
    pred_scaled = model.predict(input_scaled)
    soh = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
    
    # Tentukan status
    if soh >= 90:
        status = "✅ SEHAT"
        status_desc = "Battery in excellent condition"
        recommendation = "Continue normal usage. Next service in 6 months."
        color_class = "healthy"
    elif soh >= 70:
        status = "⚠️ WASPADA"
        status_desc = "Battery showing degradation"
        recommendation = "Schedule battery inspection and balancing service soon."
        color_class = "warning"
    else:
        status = "🔴 KRITIS"
        status_desc = "Battery health critical"
        recommendation = "Battery replacement strongly recommended."
        color_class = "critical"
    
    # Tampilkan hasil
    st.markdown("---")
    st.subheader("📊 Prediction Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("State of Health (SOH)", f"{soh:.1f}%")
    
    with col2:
        st.metric("Status", status)
    
    with col3:
        st.metric("Recommendation", recommendation[:50] + "...")
    
    # Gauge Chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=soh,
        title={"text": "SOH Meter"},
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#667eea"},
            "steps": [
                {"range": [0, 70], "color": "#eb3349"},
                {"range": [70, 90], "color": "#f2994a"},
                {"range": [90, 100], "color": "#11998e"}
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": soh
            }
        }
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    # Detail rekomendasi
    st.markdown(f"""
    <div style="background-color: #f0f2f6; padding: 1rem; border-radius: 10px; margin-top: 1rem;">
        <h4>💡 Rekomendasi Detail</h4>
        <p>{recommendation}</p>
        <p><strong>SOH:</strong> {soh:.1f}% - {status_desc}</p>
    </div>
    """, unsafe_allow_html=True)

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>Dataset: CNR Italy EIS Dataset (Cell 08) | Model: ANN (MLPRegressor) | Project SC 2026</p>
    <p>Input parameters: Aging Cycle, SOC, R_int, OCV → Output: State of Health (SOH)</p>
</div>
""", unsafe_allow_html=True)
