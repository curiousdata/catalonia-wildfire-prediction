import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv("data/IberFire_demo.csv")

# Split into features and target
X = data.drop(columns=["is_fire"])  
y = data["is_fire"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Enable MLflow autologging
mlflow.set_tracking_uri("http://localhost:5001")  # Ensure it points to the running MLflow server
mlflow.set_experiment("IberFire_Demo_Experiment")
mlflow.xgboost.autolog()

# Start an MLflow run
with mlflow.start_run():
    # Define the XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    # Log the metric manually (optional, since autologging already logs it)
    mlflow.log_metric("mse", mse)