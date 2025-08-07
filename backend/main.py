from fastapi import FastAPI, Query
import pandas as pd
import mlflow.pyfunc
from typing import List
import xgboost as xgb
import xarray as xr
from datetime import datetime, time, timedelta

# Initialize FastAPI app
app = FastAPI()

# Load the trained model from local storage
model_path = "model/IberFire_demo_model.ubj"
model = xgb.XGBClassifier()
model.load_model(model_path)

# Load demo dataset
dataset_path = "data/IberFire_demo.nc"
data = xr.open_dataset(dataset_path)

@app.get("/predict")
async def predict(year= Query(..., description="Year of the prediction"),
                 month= Query(..., description="Month of the prediction"),
                 day= Query(..., description="Day of the prediction"),
                 latitudes: List[float] = Query(..., description="List of latitudes"),
                 longitudes: List[float] = Query(..., description="List of longitudes")):
    """
    Predict wildfire risks for a given date and coordinates.
    """

    # Find entry with the same date in columns "year", "month", "day"
    query_date = date(year, month, day).isoformat()
    date_entry = data.sel(time=query_date)    
    
    # Prepare input data
    input_data = date_entry.to_dataframe().reset_index()

    predictions = model.predict_proba(input_data)

    # Return predictions as a JSON response
    return {"predictions": predictions.tolist()}