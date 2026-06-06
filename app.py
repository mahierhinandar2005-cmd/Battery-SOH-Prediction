import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go
import pandas as pd

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Battery Health Predictor",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== CUSTOM CSS PREMIUM ====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,300;14..32,400;14..32,500;14..32,600;14..32,700;14..32,800;14..32,900&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    min-height: 100vh;
}

/* Hide default Streamlit elements */
#MainMenu, footer, header {
    visibility: hidden;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* ========== HEADER ========== */
.app-title {
    text-align: center;
    background: linear-gradient(135deg, #10b981, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.5rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin-bottom: 0.5rem;
}

.app-sub {
    text-align: center;
    color: #94a3b8;
    font-size: 1rem;
    font-weight: 500;
    margin-bottom: 2rem;
}

/* ========== INFO CARD ========== */
.info-card {
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.3);
    border-radius: 16px;
    padding: 1rem 1.5rem;
    margin-bottom: 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 1rem;
}

.info-badge {
    background: rgba(16, 185, 129, 0.2);
    border-radius: 99px;
    padding: 0.25rem 1rem;
    font-size: 0.75rem;
    font-weight: 600;
    color: #10b981;
    display: inline-block;
}

/* ========== INPUT CARD ========== */
.input-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(16, 185, 129, 0.2);
    border-radius: 24px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(10px);
}

.input-label {
    color: #94a3b8;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 0.25rem;
}

.input-hint {
    color: #475569;
    font-size: 0.7rem;
    margin-top: 0.25rem;
}

/* ========== RESULT CARD ========== */
.result-card {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(6, 182, 212, 0.1));
    border: 2px solid #10b981;
    border-radius: 24px;
    padding: 1.75rem;
    text-align: center;
    margin: 1rem 0;
    box-shadow: 0 8px 32px rgba(16, 185, 129, 0.2);
}

.result-percent {
    font-size: 4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #10b981, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
}

.result-label {
    color: #94a3b8;
    font-size: 0.9rem;
    font-weight: 500;
    margin-top: 0.5rem;
}

/* ========== METRIC GRID ========== */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin: 1.5rem 0;
}

.metric-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(16, 185, 129, 0.2);
    border-radius: 16px;
    padding: 1rem;
    text-align: center;
}

.metric-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #10b981;
}

.metric-label {
    font-size: 0.7rem;
    color: #64748b;
    margin-top: 0.25rem;
}

/* ========== BUTTON ========== */
.stButton > button {
    background: linear-gradient(135deg, #10b981, #06b6d4) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 1.5rem !important;
    width: 100% !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 14px rgba(16, 185, 129, 0.3) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(16, 185, 129, 0.4) !important;
}

/* ========== DIVIDER ========== */
.divider {
    border: none;
    border-top: 1px solid rgba(16, 185, 129, 0.15);
    margin: 1.5rem 0;
}

/* ========== TIP BOX ========== */
.tip-box {
    background: rgba(6, 182, 212, 0.1);
    border-left: 3px solid #06b6d4;
    border-radius: 0 12px 12px 0;
    padding: 0.75rem 1rem;
    margin: 0.75rem 0;
    font-size: 0.85rem;
    color: #94a3b8;
}

/* ========== STREAMLIT INPUT OVERRIDES ========== */
.stNumberInput label, .stSlider label {
    color: #cbd5e1 !important;
    font-weight: 600 !important;
}

.stNumberInput input, .stSlider div {
    background: rgba(30, 41, 59, 0.8) !important;
    color: white !important;
    border: 1px solid rgba(16, 185, 129, 0.3) !important;
    border-radius: 10px !important;
}

.stSlider > div > div > div > div {
    background: #10b981 !important;
}

/* ========== RECOMMENDATION CARD ========== */
.rec-card {
    background: rgba(16, 185, 129, 0.05);
    border-radius: 16px;
    padding: 1rem;
    margin-top: 1rem;
}

.rec-title {
    font-weight: 700;
    color: #10b981;
    margin-bottom: 0.5rem;
}

.rec-text {
    color: #94a3b8;
    font-size: 0.85rem;
    line-height: 1.5;
}

/* ========== FOOTER ========== */
.footer {
    text-align: center;
    color: #475569;
    font-size: 0.7rem;
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid rgba(16, 185, 129, 0.15);
}
</style>
""", unsafe_allow_html=True)

# ==================== HEADER ====================
st.markdown('<div class="app-title">🔋 Battery Health Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="app-sub">Sistem Prediksi State of Health (SOH) Baterai Mobil Listrik menggunakan ANN</div>', unsafe_allow_html=True)

# ==================== INFO CARD ====================
st.markdown(f"""
<div class="info-card">
    <div>
        <span class="info-badge">📊 AN</span>
        <span style="color: #94a3b8; font-size: 0.75rem; margin-left: 0.5rem;">Multi-Layer Perceptron (128-64-32)</span>
    </div>
    <div>
        <span class="info-badge">📁 CNR Italy</span>
        <span style="color: #94a3b8; font-size: 0.75rem; margin-left: 0.5rem;">EIS Dataset 2026</span>
    </div>
    <div>
        <span class="info-badge">🎯 Regresi</span>
        <span style="color: #94a3b8; font-size: 0.75rem; margin-left: 0.5rem;">MAPE ≤ 10%</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ==================== LOAD MODEL ====================
@st.cache_resource
def load_model():
    try:
        model = joblib.load('model_soh.pkl')
        scaler_X = joblib.load('scaler_X.pkl')
        scaler_y = joblib.load('scaler_y.pkl')
        return model, scaler_X, scaler_y
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()

model, scaler_X, scaler_y = load_model()

# ==================== INPUT SECTION ====================
st.markdown('<div class="input-card">', unsafe_allow_html=True)
st.markdown('<div style="margin-bottom: 1rem;"><span class="info-badge">📝 INPUT</span><span style="color: #94a3b8; font-size: 0.7rem; margin-left: 0.5rem;">Masukkan parameter baterai</span></div>', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="input-label">🔄 AGING CYCLE</div>', unsafe_allow_html=True)
    cycle = st.number_input("", min_value=0, max_value=2000, value=100, step=10, key="cycle", label_visibility="collapsed")
    st.markdown('<div class="input-hint">Jumlah siklus charge-discharge (0-2000 cycle)</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="input-label" style="margin-top: 1rem;">🔋 SOC (%)</div>', unsafe_allow_html=True)
    soc = st.number_input("", min_value=0, max_value=100, value=80, step=5, key="soc", label_visibility="collapsed")
    st.markdown('<div class="input-hint">State of Charge — level pengisian baterai</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="input-label">⚡ R_int (%)</div>', unsafe_allow_html=True)
    r_int = st.number_input("", min_value=0, max_value=200, value=100, step=5, key="rint", label_visibility="collapsed")
    st.markdown('<div class="input-hint">Internal Resistance — 100% = normal</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="input-label" style="margin-top: 1rem;">🔌 OCV (V)</div>', unsafe_allow_html=True)
    ocv = st.number_input("", min_value=3.0, max_value=4.5, value=4.15, step=0.05, key="ocv", label_visibility="collapsed")
    st.markdown('<div class="input-hint">Open Circuit Voltage — tegangan saat diam</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ==================== PREDICT BUTTON ====================
if st.button("🔮 PREDIKSI KESEHATAN BATERAI"):
    # Prediksi
    input_data = np.array([[cycle, soc, r_int, ocv]])
    input_scaled = scaler_X.transform(input_data)
    pred_scaled = model.predict(input_scaled)
    soh = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
    
    # Tentukan status
    if soh >= 90:
        status = "✅ SEHAT"
        status_desc = "Baterai dalam kondisi sangat baik"
        recommendation = "Lanjutkan penggunaan normal. Service rutin setiap 6 bulan."
        color = "#10b981"
    elif soh >= 70:
        status = "⚠️ WASPADA"
        status_desc = "Baterai mulai menunjukkan degradasi"
        recommendation = "Segera lakukan inspeksi dan balancing cell."
        color = "#f59e0b"
    else:
        status = "🔴 KRITIS"
        status_desc = "Kesehatan baterai kritis"
        recommendation = "Ganti baterai segera untuk keselamatan dan performa."
        color = "#ef4444"
    
    # ==================== RESULT SECTION ====================
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown('<div style="margin-bottom: 1rem;"><span class="info-badge">📊 HASIL PREDIKSI</span><span style="color: #94a3b8; font-size: 0.7rem; margin-left: 0.5rem;">Berdasarkan model ANN</span></div>', unsafe_allow_html=True)
    
    # Result card
    st.markdown(f"""
    <div class="result-card">
        <div class="result-percent">{soh:.1f}%</div>
        <div class="result-label">State of Health (SOH)</div>
        <div style="margin-top: 12px;">
            <span style="display: inline-block; background: {color}20; color: {color}; padding: 0.25rem 1rem; border-radius: 99px; font-weight: 700;">
                {status}
            </span>
        </div>
        <div style="margin-top: 8px; color: #94a3b8; font-size: 0.8rem;">
            {status_desc}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ==================== GAUGE CHART ====================
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=soh,
        title={"text": "SOH Meter", "font": {"color": "#cbd5e1"}},
        domain={"x": [0, 1], "y": [0, 1]},
        number={"font": {"color": "#10b981", "size": 40}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#64748b"},
            "bar": {"color": "#10b981"},
            "bgcolor": "#1e293b",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 70], "color": "#7f1a1a"},
                {"range": [70, 90], "color": "#854d0e"},
                {"range": [90, 100], "color": "#14532d"}
            ],
            "threshold": {
                "line": {"color": "#fbbf24", "width": 4},
                "thickness": 0.75,
                "value": soh
            }
        }
    ))
    fig.update_layout(
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#cbd5e1"}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # ==================== METRICS ====================
    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-value">{cycle}</div>
            <div class="metric-label">Aging Cycle</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{soc}%</div>
            <div class="metric-label">SOC</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{r_int}%</div>
            <div class="metric-label">R_int</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{ocv} V</div>
            <div class="metric-label">OCV</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ==================== RECOMMENDATION ====================
    st.markdown(f"""
    <div class="rec-card">
        <div class="rec-title">💡 REKOMENDASI</div>
        <div class="rec-text">{recommendation}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== FOOTER ====================
st.markdown("""
<div class="footer">
    🔋 Dataset: CNR Italy EIS Dataset (2026) | 🧠 Model: ANN (MLPRegressor) | 🎓 Project SC 2026
</div>
""", unsafe_allow_html=True)
