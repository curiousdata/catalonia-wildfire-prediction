# backend/README.md

# Catalonia Wildfire MVP Backend

This is the backend service for the Catalonia Wildfire MVP project. It serves a pre-trained model for predicting wildfire risks based on input data.

## Project Structure

- `src/main.py`: Entry point of the backend application, initializes the FastAPI app and includes API routes.
- `src/api/routes.py`: Defines the API routes for model inference.
- `src/models/loader.py`: Contains functions to load the pre-trained model from the specified path.
- `src/inference/predict.py`: Logic for making predictions using the loaded model.
- `src/types/schema.py`: Defines data schemas for request and response validation.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd catalonia-wildfire-mvp/backend
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   uvicorn src.main:app --host 0.0.0.0 --port 8000
   ```

## Docker Instructions

To build and run the backend service using Docker:

1. Build the Docker image:
   ```
   docker build -t catalonia-wildfire-backend .
   ```

2. Run the Docker container:
   ```
   docker run -p 8000:8000 catalonia-wildfire-backend
   ```

## Usage

Once the backend is running, you can access the API at `http://localhost:8000`. Use the defined endpoints for model inference.

## License

This project is licensed under the MIT License.