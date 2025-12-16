from pydantic import BaseModel
from typing import List, Optional

class PredictionRequest(BaseModel):
    image_path: str

class PredictionResponse(BaseModel):
    success: bool
    predictions: List[float]
    error: Optional[str] = None