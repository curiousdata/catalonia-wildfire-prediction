from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from ..models.loader import load_model
from ..inference.predict import make_prediction

router = APIRouter()

class PredictionRequest(BaseModel):
    data: List[float]

class PredictionResponse(BaseModel):
    prediction: float

model = load_model("/Users/vladimir/catalonia-wildfire-prediction/models/resnet34_v8.pth")

@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        prediction = make_prediction(model, request.data)
        return PredictionResponse(prediction=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))