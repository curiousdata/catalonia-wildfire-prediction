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
async def predict(date: str, latitudes: List[float] = Query(...), longitudes: List[float] = Query(...)):
    """
    Predict wildfire risks for a given date and coordinates.
    """
    # TODO: Parse the data string; find the data in the dataset, and infer dataset entry 
    # based on the date provided through the model.
    # Create a DataFrame for input features
    input_data = pd.DataFrame({
        "latitude": latitudes,
        "longitude": longitudes,
        "date": [date] * len(latitudes)
    })

    # Add any necessary feature engineering here (e.g., extracting day of year)
    input_data["day_of_year"] = pd.to_datetime(input_data["date"]).dt.dayofyear

    # Make predictions
    predictions = model.predict(input_data)

    # Return predictions as a JSON response
    return {"date": date, "predictions": predictions.tolist()}