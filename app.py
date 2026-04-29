import streamlit as st
import requests
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# --------------------------------------------------
# Custom CSS for better UI
# --------------------------------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 40px;
        font-weight: 700;
        color: #1f2937;
    }
    .sub-title {
        font-size: 18px;
        color: #6b7280;
        margin-bottom: 25px;
    }
    .card {
        padding: 20px;
        border-radius: 15px;
        background-color: #f9fafb;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    .badge {
        display: inline-block;
        padding: 6px 14px;
        background-color: #e0f2fe;
        color: #0369a1;
        border-radius: 20px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# Title Section
# --------------------------------------------------
st.markdown('<div class="main-title">🚗 Used Car Price Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Get an estimated resale value of your vehicle</div>', unsafe_allow_html=True)

# --------------------------------------------------
# Load Dataset (✅ FIXED PATH)
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent   # This is D:\Used Car Price
DATA_PATH = BASE_DIR / "data" / "used_cars.csv"

if not DATA_PATH.exists():
    st.error(f"❌ Dataset not found at: {DATA_PATH}")
    st.stop()

df = pd.read_csv(DATA_PATH)

# Drop unwanted column if present
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

# --------------------------------------------------
# Sidebar – Inputs
# --------------------------------------------------
st.sidebar.header("🔧 Car Details")

year = st.sidebar.number_input("Manufacturing Year", 2000, 2025, 2022)
km = st.sidebar.number_input("Kilometers Driven", min_value=0)

# ✅ Safe check if required columns exist
required_cols = [
    "brand", "model", "fuel_type", "transmission_type",
    "city", "number_of_owners", "bodytype"
]

missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    st.error(f"❌ Missing columns in dataset: {missing_cols}")
    st.stop()

brands = sorted(df["brand"].dropna().astype(str).unique())
brand = st.sidebar.selectbox("Brand", brands)

models = sorted(
    df[df["brand"] == brand]["model"]
    .dropna()
    .astype(str)
    .unique()
)
model = st.sidebar.selectbox("Model", models)

fuel = st.sidebar.selectbox(
    "Fuel Type",
    sorted(df["fuel_type"].dropna().astype(str).unique())
)

trans = st.sidebar.selectbox(
    "Transmission",
    sorted(df["transmission_type"].dropna().astype(str).unique())
)

city = st.sidebar.selectbox(
    "City",
    sorted(df["city"].dropna().astype(str).unique())
)

owners = st.sidebar.selectbox(
    "Number of Owners",
    sorted(df["number_of_owners"].dropna().astype(int).unique())
)

# --------------------------------------------------
# Auto Body Type
# --------------------------------------------------
bodytype_series = df[
    (df["brand"] == brand) & (df["model"] == model)
]["bodytype"].dropna()

if len(bodytype_series) > 0:
    body = str(bodytype_series.mode()[0])
else:
    body = "Unknown"

# --------------------------------------------------
# Main Content Layout
# --------------------------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📋 Selected Car Summary")
    st.markdown(
        f"""
        <div class="card">
            <b>Brand:</b> {brand}<br>
            <b>Model:</b> {model}<br>
            <b>Year:</b> {year}<br>
            <b>Kilometers Driven:</b> {km:,} km<br>
            <b>Fuel Type:</b> {fuel}<br>
            <b>Transmission:</b> {trans}<br>
            <b>City:</b> {city}<br>
            <b>Owners:</b> {owners}<br><br>
            <span class="badge">{body}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown("### 💰 Price Estimation")

    if st.button("🚀 Predict Price", use_container_width=True):
        payload = {
            "manufacturing_year": int(year),
            "km_driven": int(km),
            "brand": brand,
            "model": model,
            "fuel_type": fuel,
            "transmission_type": trans,
            "city": city,
            "bodytype": body,
            "number_of_owners": int(owners)
        }

        try:
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json=payload,
                timeout=10
            )

            if response.status_code == 200:
                price = response.json().get("predicted_price", None)
                if price is not None:
                    st.success(f"Estimated Price: ₹ {int(price):,}")
                else:
                    st.error("❌ API response missing 'predicted_price'")
            else:
                st.error(f"❌ Prediction failed (Status: {response.status_code})")

        except Exception as e:
            st.error(f"❌ API Error: {e}")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.markdown(
    "<center>Developed by <b>Team Plan B</b> | Machine Learning Project</center>",
    unsafe_allow_html=True
)
