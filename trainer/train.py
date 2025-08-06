import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
import logging
import os
import time
from datetime import datetime

# Create a logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Configure logging
log_filename = f"logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename),  # Save logs to a file
        logging.StreamHandler()  # Print logs to the console
    ]
)

logger = logging.getLogger(__name__)
# Log the start of the training process
# Load the dataset
logger.info("Loading the dataset...")
data = pd.read_parquet("data/IberFire_demo.parquet")

logger.info("Starting the training process...")

# Split into features and target
X = data.drop(columns=["is_fire"])  
y = data["is_fire"]

logger.info("Dataset loaded successfully with shape: %s", data.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Enable MLflow autologging
mlflow.set_tracking_uri("http://localhost:5001")  # Ensure it points to the running MLflow server
mlflow.set_experiment("IberFire_Demo_Experiment")
mlflow.xgboost.autolog()
logger.info("MLflow autologging enabled.")

time.sleep(1)

logger.info("Checking for invalid values in y_train...")
logger.info(f"Number of NaN values in y_train: {y_train.isna().sum()}")
logger.info(f"Number of infinite values in y_train: {y_train.isin([float('inf'), float('-inf')]).sum()}")
logger.info(f"Maximum value in y_train: {y_train.max()}")
logger.info(f"Minimum value in y_train: {y_train.min()}")
# Drop rows with NaN or infinite values in y_train
logger.info("Dropping rows with invalid values in y_train and test...")
valid_indices = y_train.notna() & ~y_train.isin([float('inf'), float('-inf')])
X_train = X_train[valid_indices]
y_train = y_train[valid_indices]

valid_indices_test = y_test.notna() & ~y_test.isin([float('inf'), float('-inf')])
X_test = X_test[valid_indices_test]
y_test = y_test[valid_indices_test]

# Start an MLflow run
with mlflow.start_run():
    # Define the XGBoost model
    logger.info("Defining the XGBoost model...")
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        objective='binary:logistic',  # Use binary logistic regression for binary classification
    )

    # Train the model
    logger.info("Training the model...")
    model.fit(X_train, y_train)

    # Make predictions
    logger.info("Making predictions on the test set...")
    y_pred = model.predict__proba(X_test)[:, 1]  # Get probabilities for the positive class

    # Calculate metrics
    logger.info("Calculating metrics...")
    accuracy = accuracy_score(y_test, y_pred.round())
    precision = precision_score(y_test, y_pred.round(), zero_division=0)
    recall = recall_score(y_test, y_pred.round(), zero_division=0)
    f1 = f1_score(y_test, y_pred.round(), zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred)
    logloss = log_loss(y_test, y_pred)
    logger.info("Metrics calculated successfully.")
    # Log metrics to MLflow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("log_loss", logloss)
    # Log the model
    mlflow.xgboost.log_model(model, "model")
    logger.info("Model logged to MLflow.")
# Save the model locally
model.save_model("model/IberFire_demo_model.json")

logger.info("Training completed successfully. Model saved and logged to MLflow.")