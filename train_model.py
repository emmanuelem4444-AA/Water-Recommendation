"""
train_model.py
==============
End-to-end ML pipeline to predict daily plant water requirement (Water_L_per_day).

Dataset   : tropical_plants_2000_dataset.xlsx
Algorithm : RandomForestRegressor wrapped in a scikit-learn Pipeline
Output    : water_requirement_pipeline.joblib  (pipeline + model in one file)

Usage
-----
    python train_model.py

The saved .joblib file can be loaded in any Streamlit / FastAPI application via:
    import joblib
    pipeline = joblib.load("water_requirement_pipeline.joblib")
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ─────────────────────────────────────────
# 1. Configuration
# ─────────────────────────────────────────
DATASET_PATH   = os.path.join(os.path.dirname(__file__), "tropical_plants_2000_dataset.xlsx")
MODEL_PATH     = os.path.join(os.path.dirname(__file__), "water_requirement_pipeline.joblib")

CATEGORICAL_FEATURES = ["Plant", "Soil_Type", "Climate"]
NUMERICAL_FEATURES   = ["Area_m2", "Temperature_C"]
TARGET               = "Water_L_per_day"

TEST_SIZE    = 0.20
RANDOM_STATE = 42

# ─────────────────────────────────────────
# 2. Load & validate data
# ─────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    """Load the Excel dataset and perform basic validation."""
    print(f"[INFO] Loading dataset from: {path}")
    df = pd.read_excel(path)
    print(f"[INFO] Shape: {df.shape}")
    print(f"[INFO] Columns: {df.columns.tolist()}")
    required = CATEGORICAL_FEATURES + NUMERICAL_FEATURES + [TARGET]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")
    # Drop rows with NaN in required columns
    before = len(df)
    df.dropna(subset=required, inplace=True)
    after  = len(df)
    if before != after:
        print(f"[WARN] Dropped {before - after} rows with missing values.")
    return df


# ─────────────────────────────────────────
# 3. Build preprocessing + model pipeline
# ─────────────────────────────────────────
def build_pipeline() -> Pipeline:
    """
    Returns a scikit-learn Pipeline that bundles:
      - ColumnTransformer (OHE for categoricals, StandardScaler for numericals)
      - RandomForestRegressor
    """
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    numerical_transformer   = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
            ("num", numerical_transformer,   NUMERICAL_FEATURES),
        ],
        remainder="drop",
    )

    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor",    rf),
    ])

    return pipeline


# ─────────────────────────────────────────
# 4. Train & evaluate
# ─────────────────────────────────────────
def train_and_evaluate(df: pd.DataFrame) -> Pipeline:
    X = df[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"\n[INFO] Train samples : {len(X_train)}")
    print(f"[INFO] Test  samples : {len(X_test)}")

    pipeline = build_pipeline()
    print("\n[INFO] Training RandomForestRegressor (n_estimators=500) …")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
    r2     = r2_score(y_test, y_pred)

    print("\n" + "="*45)
    print("  MODEL EVALUATION (Test Set)")
    print("="*45)
    print(f"  RMSE : {rmse:.4f} L/day")
    print(f"  R²   : {r2:.4f}")
    print("="*45)

    return pipeline


# ─────────────────────────────────────────
# 5. Save pipeline
# ─────────────────────────────────────────
def save_pipeline(pipeline: Pipeline, path: str) -> None:
    joblib.dump(pipeline, path)
    print(f"\n[INFO] Pipeline saved to: {path}")


# ─────────────────────────────────────────
# 6. Inference helper  (reusable in apps)
# ─────────────────────────────────────────
def predict_water_requirement(
    data,                          # dict or pd.DataFrame or list of dicts
    model_path: str = MODEL_PATH,
    pipeline: Pipeline = None,
) -> np.ndarray:
    """
    Predict daily water requirement for one or more plant samples.

    Parameters
    ----------
    data : dict | list[dict] | pd.DataFrame
        Input features. Required keys / columns:
            'Plant', 'Area_m2', 'Soil_Type', 'Climate', 'Temperature_C'
    model_path : str
        Path to the saved .joblib pipeline (used only if `pipeline` is None).
    pipeline : sklearn.pipeline.Pipeline, optional
        Pass a pre-loaded pipeline to avoid disk I/O on every call.

    Returns
    -------
    np.ndarray
        Predicted water requirements in litres per day.

    Example
    -------
    >>> prediction = predict_water_requirement({
    ...     "Plant": "Hibiscus",
    ...     "Area_m2": 5.0,
    ...     "Soil_Type": "Loamy",
    ...     "Climate": "Tropical",
    ...     "Temperature_C": 30.0,
    ... })
    >>> print(f"Predicted: {prediction[0]:.2f} L/day")
    """
    if pipeline is None:
        pipeline = joblib.load(model_path)

    if isinstance(data, dict):
        df_input = pd.DataFrame([data])
    elif isinstance(data, list):
        df_input = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        df_input = data.copy()
    else:
        raise TypeError(f"Unsupported input type: {type(data)}")

    return pipeline.predict(df_input)


# ─────────────────────────────────────────
# 7. Entry point
# ─────────────────────────────────────────
if __name__ == "__main__":
    df       = load_data(DATASET_PATH)
    pipeline = train_and_evaluate(df)
    save_pipeline(pipeline, MODEL_PATH)

    # Quick smoke-test with first row of the dataset
    sample   = df[CATEGORICAL_FEATURES + NUMERICAL_FEATURES].iloc[[0]]
    pred     = predict_water_requirement(sample, pipeline=pipeline)
    actual   = df[TARGET].iloc[0]
    print(f"\n[INFO] Smoke-test — Actual: {actual:.2f} L/day | Predicted: {pred[0]:.2f} L/day")
    print("\n[DONE] Training complete.")
