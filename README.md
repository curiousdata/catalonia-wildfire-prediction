# Catalonia Wildfire Prediction

This project aims to predict wildfire risks in Catalonia using the **IberFire dataset**, a comprehensive datacube containing environmental, meteorological, and geographical data. The project leverages machine learning techniques, specifically **XGBoost**, to classify areas at risk of wildfires based on historical data and environmental features.

---

## **Problem Statement**

Wildfires pose a significant threat to ecosystems, human lives, and infrastructure. Predicting wildfire risks can help authorities take preventive measures and allocate resources effectively. However, wildfire prediction is challenging due to the complex interplay of environmental, meteorological, and human factors.

This project addresses the problem by:
1. Extracting relevant data for Catalonia from the IberFire dataset.
2. Engineering features that influence wildfire risks.
3. Training a machine learning model to classify areas as "fire" or "no fire."

---

## **Project Structure**

The project is organized as follows:

```
catalonia-wildfire-prediction/
├── data/                          # Contains raw and processed datasets
│   ├── IberFire.nc                # Original IberFire dataset (not included in the repo)
│   ├── iberfire_catalonia.nc      # Subset of IberFire for Catalonia
│   ├── IberFire_demo.parquet      # Processed dataset for training
├── logs/                          # Logs generated during training
├── model/                         # Saved models (ignored by Git)
│   ├── IberFire_demo_model.ubj    # Trained XGBoost model
├── notebooks/                     # Jupyter notebooks for data preparation
│   ├── create_catalonia_dataset.ipynb
│   ├── create_demo_dataset.ipynb
├── trainer/                       # Training scripts
│   ├── train.py                   # Main training script
├── docker-compose.yaml            # Docker configuration for MLflow
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
```

---

## **Data Pipeline**

### **1. Data Extraction**
- The **IberFire dataset** is a NetCDF datacube containing environmental and meteorological data for Spain.
- The `create_catalonia_dataset.ipynb` notebook extracts data specific to Catalonia using a mask for the region.

### **2. Data Filtering**
- The `create_demo_dataset.ipynb` notebook:
  - Filters data from 2020 onward.
  - Selects relevant features such as temperature, humidity, vegetation indices, and terrain characteristics.
  - Converts the dataset to a tabular format (`Parquet`) for machine learning.

### **3. Data Preprocessing**
- The `train.py` script:
  - Loads the processed dataset (`IberFire_demo.parquet`).
  - Splits the data into features (`X`) and target (`y`), where `is_fire` is the target variable.
  - Handles missing values and ensures compatibility with XGBoost by converting integer columns to `float64`.

---

## **Model Training**

### **1. Model**
- The project uses **XGBoost**, a gradient boosting algorithm, for binary classification (`fire` vs. `no fire`).

### **2. Metrics**
The following metrics are used to evaluate the model:
- **Accuracy**: Overall correctness of predictions.
- **Precision**: Proportion of predicted fires that are actual fires.
- **Recall**: Proportion of actual fires that are correctly predicted.
- **F1-Score**: Harmonic mean of precision and recall.
- **ROC-AUC**: Ability to distinguish between fire and no-fire classes.
- **Log Loss**: Confidence of the model in its predictions.

### **3. Logging**
- **MLflow** is used to track experiments, log metrics, and save models.
- Metrics and the trained model are logged to an MLflow server running locally via Docker.

---

## **How to Run the Project**

### **1. Prerequisites**
- Python 3.13 or later
- Docker and Docker Compose

### **2. Install Dependencies**
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### **3. Start the MLflow Server**
Start the MLflow server using Docker Compose:
```bash
docker-compose up -d
```

The MLflow UI will be available at [http://localhost:5001](http://localhost:5001).

### **4. Prepare the Dataset**
Run the Jupyter notebooks in the following order:
1. `create_catalonia_dataset.ipynb`: Extract Catalonia-specific data.
2. `create_demo_dataset.ipynb`: Filter and preprocess the dataset.

### **5. Train the Model**
Run the training script:
```bash
python trainer/train.py
```

### **6. View Results**
- Check the logs in the `logs/` directory for detailed training information.
- View the metrics and model in the MLflow UI.

---

## **Next Steps**
- Address class imbalance using techniques like oversampling, undersampling, or class weighting.
- Perform hyperparameter tuning to improve model performance.
- Explore additional features for better predictions.
- Deploy the trained model using MLflow or a cloud-based service.

---

## **Acknowledgments**
- **IberFire Dataset**: Provided by Julen Ercibengoa Calvo.
- **MLflow**: Used for experiment tracking and model management.
- **XGBoost**: Core machine learning algorithm used in this project.

---

## **License**
This project is licensed under the MIT License. See the `LICENSE` file for details.

