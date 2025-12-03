import streamlit as st
import numpy as np
import pandas as pd
import pickle
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler

st.title("🤖 Model & Predictions")

# ------------------------------
# TRAIN MODEL IF NOT EXISTS
# ------------------------------
import os

if not os.path.exists("models/model.pkl"):

    st.warning("Training model for first time. This may take a moment...")

    df = pd.read_csv("data/walmart.csv")

    df = df.dropna()

    features = ["Store", "Dept", "Temperature", "Fuel_Price", "Unemployment", "IsHoliday"]
    target = "Weekly_Sales"

    X = df[features]
    y = df[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LGBMRegressor()
    model.fit(X_scaled, y)

    os.makedirs("models", exist_ok=True)
    pickle.dump(model, open("models/model.pkl", "wb"))
    pickle.dump(scaler, open("models/scaler.pkl", "wb"))

    st.success("Model trained and saved!")

# ------------------------------
# LOAD MODEL
# ------------------------------
model = pickle.load(open("models/model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

# ------------------------------
# SINGLE PREDICTION
# ------------------------------
st.subheader("🎯 Predict Single Week Sales")

store = st.number_input("Store", 1)
dept = st.number_input("Department", 1)
temp = st.slider("Temperature", -20.0, 120.0, 70.0)
fuel = st.slider("Fuel Price", 1.0, 6.0, 2.5)
unemp = st.slider("Unemployment", 0.0, 20.0, 6.0)
holiday = st.selectbox("Holiday Week?", ["Yes", "No"])

holiday = 1 if holiday == "Yes" else 0

if st.button("Predict"):

    x = np.array([[store, dept, temp, fuel, unemp, holiday]])
    x_scaled = scaler.transform(x)
    pred = model.predict(x_scaled)[0]

    st.success(f"💰 Predicted Weekly Sales: **${pred:,.2f}**")


# ------------------------------
# BATCH PREDICTION
# ------------------------------
st.subheader("📥 Upload Data for Batch Prediction")

batch = st.file_uploader("Upload CSV", type=["csv"], key="batch")

if batch:
    df = pd.read_csv(batch)

    required = ["Store", "Dept", "Temperature", "Fuel_Price", "Unemployment", "IsHoliday"]
    missing = [c for c in required if c not in df]

    if missing:
        st.error(f"Missing columns: {missing}")

    else:
        X = df[required]
        X_scaled = scaler.transform(X)
        df["Predicted_Sales"] = model.predict(X_scaled)

        st.write(df.head())
        st.download_button("Download Predictions", df.to_csv(index=False), "predictions.csv")
