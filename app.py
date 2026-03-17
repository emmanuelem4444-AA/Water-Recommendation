"""
app.py
======
Flask backend for the Plant Water Requirement Predictor.

Run:
    python app.py

Then visit:
    http://127.0.0.1:5000
"""

import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template

# ── App setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── Load the model once at startup ───────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "water_requirement_pipeline.joblib")

print(f"[INFO] Loading model from: {MODEL_PATH}")
try:
    pipeline = joblib.load(MODEL_PATH)
    print("[INFO] Model loaded successfully.")
except Exception as e:
    pipeline = None
    print(f"[ERROR] Failed to load model: {e}")


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    """Serve the main HTML page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Accept JSON body with plant features and return predicted water requirement.

    Expected JSON fields:
        - Plant         (str)
        - Area_m2       (float)
        - Soil_Type     (str)
        - Climate       (str)
        - Temperature_C (float)

    Returns:
        JSON: { "water_L_per_day": <float> }
              or { "error": <message> } on failure
    """
    if pipeline is None:
        return jsonify({"error": "Model not loaded. Please check server logs."}), 500

    try:
        data = request.get_json(force=True)

        # Validate required fields
        required_fields = ["Plant", "Area_m2", "Soil_Type", "Climate", "Temperature_C"]
        missing = [f for f in required_fields if f not in data or data[f] == "" or data[f] is None]
        if missing:
            return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400

        # Parse and validate numeric fields
        try:
            area = float(data["Area_m2"])
            temp = float(data["Temperature_C"])
        except (ValueError, TypeError):
            return jsonify({"error": "Area and Temperature must be valid numbers."}), 400

        if area <= 0:
            return jsonify({"error": "Area must be greater than 0."}), 400

        if temp < -50 or temp > 70:
            return jsonify({"error": "Temperature must be between -50°C and 70°C."}), 400

        # Build DataFrame for prediction
        df_input = pd.DataFrame([{
            "Plant":         str(data["Plant"]).strip(),
            "Area_m2":       area,
            "Soil_Type":     str(data["Soil_Type"]).strip(),
            "Climate":       str(data["Climate"]).strip(),
            "Temperature_C": temp,
        }])

        # Run prediction
        prediction = pipeline.predict(df_input)
        water_req = round(float(prediction[0]), 2)

        return jsonify({"water_L_per_day": water_req})

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
