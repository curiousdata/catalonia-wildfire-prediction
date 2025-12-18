# Catalonia Wildfire Prediction

A machine learning project for predicting wildfire risks in Catalonia using deep learning models trained on the **IberFire dataset**, a comprehensive datacube containing environmental, meteorological, and geographical data for Spain.

## Overview

This project uses convolutional neural networks (CNNs) to perform spatial segmentation and predict wildfire risks in Catalonia. The system includes:
- Data processing pipelines to convert NetCDF data to Zarr format
- CNN-based models (U-Net with various encoders) for wildfire prediction
- A web-based MVP application for visualizing predictions
- MLflow integration for experiment tracking

## Project Structure

```
catalonia-wildfire-prediction/
├── catalonia-wildfire-mvp/      # Web application (FastAPI + Streamlit)
│   ├── backend/                 # API service for model inference
│   └── frontend/                # Streamlit UI for predictions
├── data/                        # Datasets (NetCDF, Zarr formats)
├── notebooks/                   # Jupyter notebooks for EDA and experiments
├── scripts/                     # Training and data processing scripts
│   ├── train.py                 # CNN model training
│   ├── netcdf_to_zarr.py       # Data format conversion
│   └── coarsen.py              # Spatial resolution adjustment
├── src/                         # Core modules
│   ├── data/                    # Dataset classes
│   └── models/                  # Model architectures
├── docker-compose.yml           # Monolith application configuration
└── requirements.txt             # Python dependencies
```

## Quick Start

### Prerequisites
- Python 3.13+
- Docker and Docker Compose (for the monolith application)
- CUDA-compatible GPU (recommended for training)

### Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start MLflow server (optional, for experiment tracking):**
   ```bash
   mlflow server --host 0.0.0.0 --port 5001
   ```
   Access MLflow UI at [http://localhost:5001](http://localhost:5001)

### Training a Model

Train a CNN model for wildfire prediction:
```bash
python scripts/train.py --model_name resnet34_v8 --epochs 50
```

The training script uses the IberFire dataset in Zarr format and logs metrics to MLflow.

### Running the Web Application

The monolith application can be started using Docker Compose from the project root:
```bash
docker-compose up --build
```

Access the application at [http://localhost:8501](http://localhost:8501)

## Data Processing

The project includes scripts for processing the IberFire dataset:
- `netcdf_to_zarr.py`: Convert NetCDF to Zarr format for efficient data access
- `coarsen.py`: Reduce spatial resolution for faster experimentation
- `rechunk.py`: Optimize data chunking for processing

## Acknowledgments

This project builds upon the work of several researchers and open-source projects:

### Dataset
- **IberFire Dataset**: Provided by **Julen Ercibengoa Calvo** (julen.ercibengoa@gmail.com, julen.ercibengoa@tekniker.es). The IberFire datacube contains environmental and meteorological data for Spain with 1km spatial resolution and daily temporal resolution.

### Libraries and Frameworks
- **PyTorch**: Deep learning framework
- **segmentation_models_pytorch**: Pre-trained segmentation models
- **MLflow**: Experiment tracking and model management
- **xarray & Zarr**: Multi-dimensional array processing and storage
- **FastAPI & Streamlit**: Web application framework

### Research Papers
- **Long-Tailed Classification via Logit Adjustment**: Menon, A. K., Jayasumana, S., Rawat, A. S., Jain, H., Veit, A., & Kumar, S. (2020). Long-tail learning via logit adjustment. arXiv preprint arXiv:2007.07314. https://arxiv.org/abs/2007.07314
  - Used for addressing class imbalance in the wildfire prediction model through logit-adjusted loss

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

