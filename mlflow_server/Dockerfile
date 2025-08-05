# Use the official MLflow image
FROM mlflow/mlflow:latest

# Expose the MLflow server port
EXPOSE 5000

# Set the default command to run the MLflow server
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "file:/mlflow/mlruns"]