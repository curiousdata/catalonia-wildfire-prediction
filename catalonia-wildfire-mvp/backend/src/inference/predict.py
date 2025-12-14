from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import torch
import torchvision.transforms as transforms
from PIL import Image

router = APIRouter()

# Load the model
model_path = "/Users/vladimir/catalonia-wildfire-prediction/models/resnet34_v8.pth"
model = torch.load(model_path)
model.eval()

# Define request and response schemas
class PredictionRequest(BaseModel):
    image_path: str

class PredictionResponse(BaseModel):
    class_label: str
    confidence: float

# Define the prediction endpoint
@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Load and preprocess the image
        image = Image.open(request.image_path)
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

        # Make prediction
        with torch.no_grad():
            output = model(input_batch)
        
        # Get the predicted class and confidence
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        class_index = probabilities.argmax().item()
        confidence = probabilities[class_index].item()

        return PredictionResponse(class_label=str(class_index), confidence=confidence)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))