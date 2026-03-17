"""
predict.py
==========
Example prediction script for the Plant Water Requirement model.

This script loads the trained pipeline from 'water_requirement_pipeline.joblib'
and demonstrates how to use predict_water_requirement() with new plant data.

Run AFTER train_model.py has been executed at least once.

Usage
-----
    python predict.py

Integration into Streamlit / FastAPI
--------------------------------------
  Streamlit:
    prediction = predict_water_requirement({"Plant": ..., ...})
    st.metric("Water Required", f"{prediction[0]:.2f} L/day")

  FastAPI:
    @app.post("/predict")
    def predict(data: PlantInput):
        result = predict_water_requirement(data.dict())
        return {"water_L_per_day": round(float(result[0]), 2)}
"""

import os
import joblib
import pandas as pd
import numpy as np

# ── Path to the saved pipeline ──────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "water_requirement_pipeline.joblib")


# ── Core prediction function ─────────────────────────────────────────────────
def predict_water_requirement(
    data,
    model_path: str = MODEL_PATH,
    _pipeline=None,          # internal cache; set by load_pipeline()
) -> np.ndarray:
    """
    Predict daily water requirement (L/day) for one or more plant samples.

    Parameters
    ----------
    data : dict | list[dict] | pd.DataFrame
        Plant feature data. Required fields:
          - Plant         (str)   e.g. "Hibiscus"
          - Area_m2       (float) e.g. 5.0
          - Soil_Type     (str)   e.g. "Loamy"
          - Climate       (str)   e.g. "Tropical"
          - Temperature_C (float) e.g. 30.0

    model_path : str
        Path to water_requirement_pipeline.joblib

    Returns
    -------
    np.ndarray  — predicted water requirements in litres per day
    """
    pipeline = _pipeline if _pipeline is not None else joblib.load(model_path)

    if isinstance(data, dict):
        df_input = pd.DataFrame([data])
    elif isinstance(data, list):
        df_input = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        df_input = data.copy()
    else:
        raise TypeError(f"Unsupported input type: {type(data)}")

    return pipeline.predict(df_input)


# ── Load pipeline once (avoids repeated disk I/O in apps) ────────────────────
def load_pipeline(model_path: str = MODEL_PATH):
    """Load and return the trained pipeline from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Please run 'python train_model.py' first."
        )
    print(f"[INFO] Loading pipeline from: {model_path}")
    return joblib.load(model_path)


# ── Example predictions ──────────────────────────────────────────────────────
if __name__ == "__main__":

    pipeline = load_pipeline()

    # ── Sample 1 : Single plant as a dict ──
    sample_1 = {
        "Plant":         "Hibiscus",
        "Area_m2":       5.0,
        "Soil_Type":     "Loamy",
        "Climate":       "Tropical",
        "Temperature_C": 30.0,
    }

    # ── Sample 2 : Another plant ──
    sample_2 = {
        "Plant":         "Bird of Paradise",
        "Area_m2":       12.5,
        "Soil_Type":     "Sandy",
        "Climate":       "Arid",
        "Temperature_C": 38.0,
    }

    # ── Batch prediction using a list of dicts ──
    batch_samples = [sample_1, sample_2]

    pred_1     = predict_water_requirement(sample_1,     _pipeline=pipeline)
    pred_2     = predict_water_requirement(sample_2,     _pipeline=pipeline)
    pred_batch = predict_water_requirement(batch_samples, _pipeline=pipeline)

    print("\n" + "="*55)
    print("  PLANT WATER REQUIREMENT — EXAMPLE PREDICTIONS")
    print("="*55)

    for i, (s, p) in enumerate(zip(batch_samples, pred_batch), start=1):
        print(f"\n  Plant {i}: {s['Plant']}")
        print(f"    Area          : {s['Area_m2']} m²")
        print(f"    Soil Type     : {s['Soil_Type']}")
        print(f"    Climate       : {s['Climate']}")
        print(f"    Temperature   : {s['Temperature_C']} °C")
        print(f"    ➜  Predicted Water Requirement : {p:.2f} L/day")

    print("\n" + "="*55)

    # ── DataFrame input example ──
    df_input = pd.DataFrame([
        {
            "Plant":         "Peace Lily",
            "Area_m2":       3.0,
            "Soil_Type":     "Clay",
            "Climate":       "Humid",
            "Temperature_C": 26.0,
        }
    ])
    pred_df = predict_water_requirement(df_input, _pipeline=pipeline)
    print(f"\n  DataFrame input — Peace Lily: {pred_df[0]:.2f} L/day")
    print("\n[DONE] All predictions complete.")
