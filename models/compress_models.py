import joblib

print("Compressing models...")

# compress RandomForest
model = joblib.load("models/rf_model.pkl")
joblib.dump(model, "models/rf_model_compressed.pkl", compress=3)
print("rf_model compressed!")

# compress XGBoost
model = joblib.load("models/xgb_model.pkl")
joblib.dump(model, "models/xgb_model_compressed.pkl", compress=3)
print("xgb_model compressed!")

# compress scaler
scaler = joblib.load("models/scaler.pkl")
joblib.dump(scaler, "models/scaler_compressed.pkl", compress=3)
print("scaler compressed!")

print("\nAll models compressed successfully!")
