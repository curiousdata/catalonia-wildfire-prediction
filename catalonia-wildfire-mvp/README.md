# README.md

# Catalonia Wildfire Prediction MVP

This project is a Minimum Viable Product (MVP) for predicting wildfire risks in Catalonia using a pre-trained ResNet model. It consists of a backend service that serves the model and a frontend application that provides a user interface for making predictions.

## Project Structure

```
catalonia-wildfire-mvp
├── backend
│   ├── src
│   │   ├── main.py          # Entry point for the backend application
│   │   ├── api
│   │   │   └── routes.py    # API routes for model inference
│   │   ├── models
│   │   │   └── loader.py     # Functions to load the pre-trained model
│   │   ├── inference
│   │   │   └── predict.py    # Logic for making predictions
│   │   └── types
│   │       └── schema.py     # Data schemas for validation
│   ├── requirements.txt      # Python dependencies for the backend
│   ├── Dockerfile             # Docker instructions for the backend
│   └── README.md             # Documentation for the backend service
├── frontend
│   ├── src
│   │   └── app.py           # Entry point for the frontend application
│   ├── requirements.txt      # Python dependencies for the frontend
│   ├── Dockerfile             # Docker instructions for the frontend
│   └── README.md             # Documentation for the frontend service
├── models
│   ├── resnet34_v8.pth       # Pre-trained model for inference
│   └── README.md             # Documentation for the model
├── docker-compose.yml         # Docker Compose configuration for services
└── README.md                 # Overall documentation for the project
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd catalonia-wildfire-mvp
   ```

2. **Backend Setup:**
   - Navigate to the `backend` directory.
   - Install dependencies:
     ```
     pip install -r requirements.txt
     ```
   - Run the backend service:
     ```
     uvicorn src.main:app --host 0.0.0.0 --port 8000
     ```

3. **Frontend Setup:**
   - Navigate to the `frontend` directory.
   - Install dependencies:
     ```
     pip install -r requirements.txt
     ```
   - Run the frontend application:
     ```
     streamlit run src/app.py
     ```

4. **Using Docker:**
   - Build and run the services using Docker Compose:
     ```
     docker-compose up --build
     ```

## Usage

- Access the frontend application at `http://localhost:8501`.
- Use the interface to input the date for which you want to predict wildfire risks.
- The backend will process the request and return the predictions.

## License

This project is licensed under the MIT License.