import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
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
logger.info("Dropping rows with invalid values in y_train...")
valid_indices = y_train.notna() & ~y_train.isin([float('inf'), float('-inf')])
X_train = X_train[valid_indices]
y_train = y_train[valid_indices]

# Start an MLflow run
with mlflow.start_run():
    # Define the XGBoost model
    logger.info("Defining the XGBoost model...")
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )

    # Train the model
    logger.info("Training the model...")
    model.fit(X_train, y_train)

    # Make predictions
    logger.info("Making predictions on the test set...")
    y_pred = model.predict(X_test)

    # Calculate metrics
    logger.info("Calculating metrics...")
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    # Log the metric manually (optional, since autologging already logs it)
    mlflow.log_metric("mse", mse)

logger.info("Training completed successfully. MSE: %f", mse)