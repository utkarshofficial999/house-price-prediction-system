import os
import json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional

# Initialize FastAPI App
app = FastAPI(
    title="House Price Prediction API",
    description="Dual Machine Learning Service (Linear Regression & Decision Tree)",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Models & Metadata
LR_PATH = "lr_model.joblib"
DT_PATH = "dt_model.joblib"
METADATA_PATH = "model_metadata.json"

if not os.path.exists(LR_PATH) or not os.path.exists(DT_PATH) or not os.path.exists(METADATA_PATH):
    raise RuntimeError("Model files not found! Please run train_model.py first.")

lr_model = joblib.load(LR_PATH)
dt_model = joblib.load(DT_PATH)

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

features_list = metadata["features"]
feature_stats = metadata["feature_stats"]
lr_meta = metadata["models"]["linear_regression"]
dt_meta = metadata["models"]["decision_tree"]


class PropertyInput(BaseModel):
    Area_sqft: float = Field(..., ge=300, le=10000, description="Living area in square feet")
    Bedrooms: int = Field(..., ge=1, le=10, description="Number of bedrooms")
    Bathrooms: int = Field(..., ge=1, le=10, description="Number of bathrooms")
    Floors: int = Field(..., ge=1, le=10, description="Number of floors")
    Age_years: float = Field(..., ge=0, le=100, description="Age of the house in years")
    Parking_spaces: int = Field(..., ge=0, le=10, description="Number of parking spaces")
    Distance_school_km: float = Field(..., ge=0.0, le=50.0, description="Distance to nearest school (km)")
    Distance_hospital_km: float = Field(..., ge=0.0, le=50.0, description="Distance to nearest hospital (km)")
    Distance_metro_km: float = Field(..., ge=0.0, le=50.0, description="Distance to nearest metro station (km)")
    Crime_rate: float = Field(..., ge=0.0, le=50.0, description="Crime rate score in neighborhood")
    Property_tax: float = Field(..., ge=0.0, le=500000.0, description="Annual property tax in INR")
    Furnished: int = Field(..., ge=0, le=1, description="Furnished status (0 = Unfurnished, 1 = Furnished)")
    Location_score: float = Field(..., ge=1.0, le=10.0, description="Location rating score (1 to 10)")
    model_choice: Optional[str] = Field("both", description="linear_regression, decision_tree, or both")


def format_currency_inr(amount: float) -> str:
    """Format numeric INR amount into Crores (Cr) or Lakhs (L)."""
    abs_amt = abs(amount)
    if abs_amt >= 10000000:
        cr = amount / 10000000
        return f"₹{cr:.2f} Cr"
    elif abs_amt >= 100000:
        lakh = amount / 100000
        return f"₹{lakh:.2f} Lakh"
    else:
        return f"₹{amount:,.0f}"


@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "models_loaded": ["LinearRegression", "DecisionTreeRegressor"],
        "algorithms": {
            "linear_regression": lr_meta["algorithm"],
            "decision_tree": dt_meta["algorithm"]
        }
    }


@app.get("/api/model-info")
def get_model_info():
    return metadata


@app.post("/api/predict")
def predict_price(prop: PropertyInput):
    try:
        # Build DataFrame with exact feature ordering
        data_dict = prop.model_dump()
        input_df = pd.DataFrame([data_dict])[features_list]

        # 1. Linear Regression Inference
        raw_lr = float(lr_model.predict(input_df)[0])
        pred_lr = max(raw_lr, 500000.0)

        # 2. Decision Tree Inference
        raw_dt = float(dt_model.predict(input_df)[0])
        pred_dt = max(raw_dt, 500000.0)

        # 3. Consensus Ensemble
        pred_consensus = (pred_lr + pred_dt) / 2.0
        delta_inr = abs(pred_lr - pred_dt)

        # Feature contributions for Linear Regression
        lr_contributions = {}
        for feat in features_list:
            val = float(data_dict[feat])
            coef = float(lr_meta["coefficients"].get(feat, 0.0))
            lr_contributions[feat] = {
                "value": val,
                "coefficient": coef,
                "impact_inr": round(val * coef, 2),
                "impact_formatted": format_currency_inr(val * coef)
            }

        # Decision Tree feature importances
        dt_importances = dt_meta["feature_importances"]

        # Confidence bounds based on selected model RMSE
        lr_rmse = lr_meta["rmse_test"]
        dt_rmse = dt_meta["rmse_test"]

        return {
            "linear_regression": {
                "predicted_price": round(pred_lr, 2),
                "predicted_price_formatted": format_currency_inr(pred_lr),
                "price_per_sqft": round(pred_lr / prop.Area_sqft, 2),
                "price_per_sqft_formatted": f"₹{pred_lr / prop.Area_sqft:,.0f}/sqft",
                "range": {
                    "low": round(max(pred_lr - lr_rmse, 300000.0), 2),
                    "high": round(pred_lr + lr_rmse, 2),
                    "low_formatted": format_currency_inr(max(pred_lr - lr_rmse, 300000.0)),
                    "high_formatted": format_currency_inr(pred_lr + lr_rmse),
                },
                "metrics": {
                    "r2_test": lr_meta["r2_test"],
                    "rmse_inr": lr_rmse,
                    "mape_percent": lr_meta["mape_test"]
                },
                "contributions": lr_contributions
            },
            "decision_tree": {
                "predicted_price": round(pred_dt, 2),
                "predicted_price_formatted": format_currency_inr(pred_dt),
                "price_per_sqft": round(pred_dt / prop.Area_sqft, 2),
                "price_per_sqft_formatted": f"₹{pred_dt / prop.Area_sqft:,.0f}/sqft",
                "range": {
                    "low": round(max(pred_dt - dt_rmse, 300000.0), 2),
                    "high": round(pred_dt + dt_rmse, 2),
                    "low_formatted": format_currency_inr(max(pred_dt - dt_rmse, 300000.0)),
                    "high_formatted": format_currency_inr(pred_dt + dt_rmse),
                },
                "metrics": {
                    "r2_test": dt_meta["r2_test"],
                    "rmse_inr": dt_rmse,
                    "mape_percent": dt_meta["mape_test"]
                },
                "importances": dt_importances
            },
            "consensus": {
                "predicted_price": round(pred_consensus, 2),
                "predicted_price_formatted": format_currency_inr(pred_consensus),
                "price_per_sqft_formatted": f"₹{pred_consensus / prop.Area_sqft:,.0f}/sqft",
                "delta_between_models": round(delta_inr, 2),
                "delta_formatted": format_currency_inr(delta_inr),
                "agreement_pct": round(max(0, 100 - (delta_inr / pred_consensus * 100)), 2)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Serve static web frontend
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def serve_index():
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))
