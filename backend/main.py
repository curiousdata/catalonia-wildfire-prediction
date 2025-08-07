from fastapi import FastAPI, Query
import pandas as pd
import mlflow.pyfunc
from typing import List
import xgboost as xgb

# Initialize FastAPI app
app = FastAPI()

# Load the trained model from local storage
model_path = "model/IberFire_demo_model.ubj"
model = xgb.XGBClassifier()
model.load_model(model_path)

# Load demo dataset
dataset_path = "data/IberFire_demo.parquet"
data = pd.read_parquet(dataset_path)

@app.get("/predict")
async def predict(year= Query(..., description="Year of the prediction"),
                 month= Query(..., description="Month of the prediction"),
                 day= Query(..., description="Day of the prediction"),
                 latitudes: List[float] = Query(..., description="List of latitudes"),
                 longitudes: List[float] = Query(..., description="List of longitudes")):
    """
    Predict wildfire risks for a given date and coordinates.
    """

    # Find entry with the same date
    if year in data["year"].values and month in data["month"].values and day in data["day"].values:
        input_data = data[(data["year"] == year) & (data["month"] == month) & (data["day"] == day)]
    else:
        year = 2022  # Default to a known year if the date is not found
        input_data = data[(data["year"] == year) & (data["month"] == month) & (data["day"] == day)]

    predictions = model.predict_proba(input_data)

    # Return predictions as a JSON response
    return {"predictions": predictions.tolist()}