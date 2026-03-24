import os
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

MODEL_DIR = "models"

# Load best model and scaler
with open(os.path.join(MODEL_DIR, "best_model.txt"), "r") as f:
    best_name = f.read().strip()

model_path = os.path.join(MODEL_DIR, f"{best_name}.pkl")
scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Input feature list (must match training)
FEATURES = [
    "amt", "zip", "lat", "long", "city_pop", "unix_time", "merch_lat", "merch_long"
]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Build feature vector in same order as training
        x = []
        for col in FEATURES:
            val = float(data.get(col, 0.0))
            x.append(val)

        x = np.array(x).reshape(1, -1)
        x_scaled = scaler.transform(x)
        pred = model.predict(x_scaled)[0]
        proba = model.predict_proba(x_scaled)[0].tolist()

        result = {
            "is_fraud": bool(pred),
            "probability": {
                "legitimate": round(proba[0], 4),
                "fraud": round(proba[1], 4)
            }
        }

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
