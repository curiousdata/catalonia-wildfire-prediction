# Catalonia Wildfire Prediction

A machine learning project for predicting wildfire risks in Catalonia using deep learning models trained on the **IberFire dataset**, a comprehensive datacube containing environmental, meteorological, and geographical data for Spain (the initial plan was for Catalonia only, hence the name).

## Overview

This project uses convolutional neural networks (CNNs), specifically **U-Net architecture with various encoders**, to perform spatial segmentation and predict wildfire risks in Catalonia. The system includes:
- Data processing pipelines to convert NetCDF data to Zarr format
- U-Net based models with different encoders (ResNet34, etc.) for wildfire prediction
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
│   ├── train.py                 # U-Net model training
│   ├── netcdf_to_zarr.py       # Data format conversion
│   └── coarsen.py              # Spatial resolution adjustment
├── src/                         # Core modules
│   ├── data/                    # Dataset classes
│   └── models/                  # Model architectures
├── docker-compose.yml           # Monolith application configuration
└── requirements.txt             # Python dependencies
```

## Quick Start

This section is divided into three parts based on your use case:
1. **Running the Application** - For users who just want to try the wildfire prediction app
2. **Training the Model** - For users who want to train the model on the existing dataset
3. **Full Experimentation** - For users who want to experiment with data processing, feature engineering, and model architecture

### Storage Requirements
**Important:** The gold dataset and latest model are managed via Git LFS. Ensure you have enough storage: the dataset and model are approximately **1 GB in total**.

---

### Part 1: Running the Application

If you just want to start the wildfire prediction application:

**Prerequisites:**
- Docker and Docker Compose
- Git LFS (for downloading the dataset and model)

**Steps:**
1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/catalonia-wildfire-prediction.git
   cd catalonia-wildfire-prediction
   ```
3. Start the application:
   ```bash
   docker-compose up --build
   ```
4. Access the application at [http://localhost:8501](http://localhost:8501)

The gold dataset (`data/gold/IberFire_coarse32_time1.zarr`) and the latest model (`models/resnet34_v9.pth`) are managed by Git LFS and will be downloaded automatically when you clone the repository.

---

### Part 2: Training the Model

If you want to train the U-Net model on the existing gold dataset:

**Prerequisites:**
- Python 3.13+
- CUDA- or MPS-compatible GPU (recommended for training)
- The gold dataset (automatically available via Git LFS)

**Steps:**
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start MLflow UI in a separate terminal (for tracking model metrics):**
   ```bash
   mlflow server --host 0.0.0.0 --port 5001
   ```
   Access MLflow UI at [http://localhost:5001](http://localhost:5001)

3. **Train the model:**
   ```bash
   python scripts/train.py --model_name resnet34_v10 --epochs 50
   ```

The training script uses the gold dataset (`data/gold/IberFire_coarse32_time1.zarr`) and logs all metrics to MLflow for experiment tracking.

---

### Part 3: Full Experimentation

If you want to experiment with the dataset, coarsening, feature engineering, etc.:

**Prerequisites:**
- Python 3.13+
- Significant disk space (~20-50 GB for raw NetCDF data)
- Computer connected to power (overnight processing recommended)

**Steps:**
1. **Download the original NetCDF dataset from Zenodo:**
   - Visit the IberFire dataset page on Zenodo
   - Download the NetCDF files
   - Place them in `data/bronze/`

2. **Convert NetCDF to Zarr format:**
   ```bash
   python scripts/netcdf_to_zarr.py
   ```
   **Note:** This conversion is best done overnight with your computer connected to power. In the future, this dataset will be published in Zarr format directly.

3. **Customize chunking and compression (optional):**
   You are free to change the chunking and compression settings in the scripts. To match the access pattern used in this project (1 image at a time fed into U-Net):
   ```bash
   python scripts/rechunk.py
   ```

4. **Apply coarsening and max pooling to the target:**
   ```bash
   python scripts/coarsen.py
   ```

5. After processing, follow **Part 2** to train your models with the new dataset configurations.

---

## Data Processing

The project includes scripts for processing the IberFire dataset:
- `netcdf_to_zarr.py`: Convert NetCDF to Zarr format for efficient data access
- `coarsen.py`: Reduce spatial resolution for faster experimentation
- `rechunk.py`: Optimize data chunking for processing

## Future Plans

We are actively working to improve wildfire prediction capabilities. Future directions include:

- **Bigger Models**: Experimenting with larger U-Net architectures and more sophisticated encoders for improved prediction accuracy
- **Finer Resolution**: Training models on higher spatial resolution data to capture more detailed fire risk patterns
- **Real-Time Data Ingestion**: Implementing a pipeline for real-time data ingestion to predict fires for tomorrow based on current conditions

If you're interested in contributing to any of these areas, please see the Contributing section below.

## Contributing

Interested in contributing to this project? We welcome contributions from the community! Here are some guidelines to get started:

### How to Contribute

1. **Fork the repository** and create a new branch for your feature or bug fix
2. **Make your changes** following the existing code style and conventions
3. **Test your changes** thoroughly to ensure they work as expected
4. **Write clear commit messages** that describe what your changes do
5. **Submit a pull request** with a detailed description of your changes and the problem they solve

### Contribution Ideas

- Work on **real-time data ingestion** for tomorrow's fire predictions
- Experiment with **bigger models** and different architectures
- Improve **spatial resolution** by training on finer-grained data
- Add new features to the web application
- Improve documentation or fix bugs
- Create tutorials or example notebooks

### Questions or Discussions

If you have questions, ideas, or want to discuss potential contributions, feel free to:
- Open an issue on GitHub
- Reach out at **vladimv.morozov@gmail.com**

We appreciate all contributions, whether it's code, documentation, bug reports, or feature suggestions!

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
- **Long-tail learning via logit adjustment**: Menon, A. K., Jayasumana, S., Rawat, A. S., Jain, H., Veit, A., & Kumar, S. (2020). arXiv preprint arXiv:2007.07314. https://arxiv.org/abs/2007.07314
  - Used for addressing class imbalance in the wildfire prediction model through logit-adjusted loss

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

