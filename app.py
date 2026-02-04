
import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="R-744 HVAC Intelligent Control",
    page_icon="❄️",
    layout="wide"
)

# -------------------------------------------------
# Custom CSS (Neon + Professional)
# -------------------------------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(120deg, #0f2027, #203a43, #2c5364);
}

.main {
    background: transparent;
}

h1, h2, h3 {
    font-family: 'Computer Modern', serif;
    color: #e0f7fa;
}

label {
    color: #e0f7fa !important;
    font-weight: 600;
}

div[data-testid="stNumberInput"] input {
    background-color: #0b1d26;
    color: #00fff5;
    border: 1px solid #00fff5;
    border-radius: 8px;
}

.stButton button {
    background: linear-gradient(90deg, #00f260, #0575e6);
    color: black;
    font-weight: bold;
    border-radius: 10px;
    padding: 10px 20px;
}

.stButton button:hover {
    background: linear-gradient(90deg, #0575e6, #00f260);
    color: white;
}

.neon-box {
    border: 1px solid #00fff5;
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 0 15px #00fff5;
    background-color: rgba(0,0,0,0.25);
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Load Model & Scaler
# -------------------------------------------------
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

# -------------------------------------------------
# HEADER SECTION
# -------------------------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("## ❄️ R-744 HVAC Intelligent Supervisory Control")
    st.markdown("""
    **AI-based surrogate modeling for optimal HVAC operation**  
    _Data-driven prediction of gas cooler operating conditions_
    """)

with col2:
    img = Image.open("ac.png")
    st.image(img, use_column_width=True)

st.markdown("---")

# -------------------------------------------------
# INPUT SECTION
# -------------------------------------------------
st.markdown("### ⚡ System Input Conditions")

with st.container():
    st.markdown('<div class="neon-box">', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        dbt = st.number_input("Dry Bulb Temperature (°C)", value=35.0)
        wbt = st.number_input("Wet Bulb Temperature (°C)", value=25.0)

    with c2:
        building_load = st.number_input(
            "Net Building Thermal Load (kW)",
            help="+ Cooling demand | − Heating demand",
            value=10.0
        )

    with c3:
        rsh = st.number_input("Room Sensible Heat Load (kW)", value=5.0)
        rsc = st.number_input("Room Sensible Cooling Load (kW)", value=8.0)

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------
# MODEL SELECTION
# -------------------------------------------------
st.markdown("### 🧠 Surrogate Model Selection")


# -------------------------------------------------
# PREDICTION
# -------------------------------------------------
if st.button("🚀 Predict Optimal Operating Pressure"):

    input_df = pd.DataFrame([{
        "DBT": dbt,
        "WBT": wbt,
        "Build. Load": building_load,
        "RSH": rsh,
        "RSC": rsc
    }])

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]

    st.markdown("### ✅ Prediction Result")
    st.success(
        f"**Optimal Gas Cooler Operating Pressure:**  \n"
        f"### {prediction:.2f} bar"
    )

    if dbt < wbt:
        st.warning("Dry bulb temperature should not be lower than wet bulb temperature.")
