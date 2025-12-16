# /catalonia-wildfire-mvp/catalonia-wildfire-mvp/frontend/README.md

# Catalonia Wildfire Prediction Frontend

This is the frontend component of the Catalonia Wildfire Prediction project. It provides a user interface for interacting with the backend model inference service.

## Overview

The frontend is built using Streamlit, a powerful framework for creating web applications for machine learning and data science projects. This application allows users to input parameters and visualize wildfire risk predictions.

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd catalonia-wildfire-mvp/frontend
   ```

2. **Install Dependencies**
   Make sure you have Python 3.7 or higher installed. Then, install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**
   Start the Streamlit application:
   ```bash
   streamlit run src/app.py
   ```

4. **Access the Application**
   Open your web browser and go to `http://localhost:8501` to access the application.

## Docker Instructions

To run the frontend in a Docker container, follow these steps:

1. **Build the Docker Image**
   ```bash
   docker build -t catalonia-wildfire-frontend .
   ```

2. **Run the Docker Container**
   ```bash
   docker run -p 8501:8501 catalonia-wildfire-frontend
   ```

3. **Access the Application**
   Open your web browser and go to `http://localhost:8501`.

## Features

- Input parameters for date selection to predict wildfire risks.
- Visualize predictions using heatmaps.
- User-friendly interface for easy interaction.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for details.