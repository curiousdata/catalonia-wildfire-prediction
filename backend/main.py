from fastapi import FastAPI, Query
import pandas as pd
import xgboost as xgb
import xarray as xr
from datetime import date
from fastapi.responses import JSONResponse

# Initialize FastAPI app
app = FastAPI()

# Load the trained model from local storage
model_path = "model/IberFire_demo_model.ubj"
model = xgb.XGBClassifier()
model.load_model(model_path)

# Load demo dataset (lazily)
dataset_path = "data/IberFire_demo.nc"
data = xr.open_dataset(dataset_path)

@app.get("/predict")
async def predict(
    year: int = Query(..., description="Year of the prediction"),
    month: int = Query(..., description="Month of the prediction"),
    day: int = Query(..., description="Day of the prediction")
):
    """
    Predict wildfire risks for a given date.
    """

    # Build date and select nearest time slice
    query_date = date(year, month, day).isoformat()
    date_entry = data.sel(time=query_date, method="nearest")

    # Convert to DataFrame and select features
    input_data = date_entry.to_dataframe().reset_index()

    selected_features = [
        "x", "y", "is_near_fire",
        "t2m_mean", "RH_mean", "FWI",
        "NDVI", "LAI",
        "elevation_mean", "slope_mean",
        "time"
    ]
    input_data = input_data[selected_features]

    # Rename columns to match model training
    input_data.rename(columns={
        "t2m_mean": "temperature_mean",
        "RH_mean": "relative_humidity_mean",
        "FWI": "fire_weather_index",
        "NDVI": "normalized_difference_vegetation_index",
        "LAI": "leaf_area_index",
        "elevation_mean": "mean_elevation",
        "slope_mean": "mean_slope"
    }, inplace=True)

    # Extract time components
    if "time" in input_data.columns:
        input_data["year"] = input_data["time"].dt.year
        input_data["month"] = input_data["time"].dt.month
        input_data["day"] = input_data["time"].dt.day
        input_data.drop(columns=["time"], inplace=True)

    # Reorder columns to match training
    expected_order = model.get_booster().feature_names
    input_data = input_data[expected_order]

    # Predict
    predictions = model.predict_proba(input_data)

    # Build output
    result_df = pd.DataFrame({
        "x": input_data["x"],
        "y": input_data["y"],
        "risk": predictions[:, 1]
    })

    return JSONResponse(result_df.to_dict(orient="records"))