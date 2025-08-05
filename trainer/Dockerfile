# Use a lightweight Python image
FROM python:3.13-slim

# Set the working directory
WORKDIR /app

# Install Poetry
RUN pip install --no-cache-dir poetry

# Copy Poetry files
COPY pyproject.toml poetry.lock ./

# Install dependencies using Poetry
RUN poetry install --no-root

# Copy the training script
COPY train.py .

# Copy the dataset
COPY ../data ./data

# Run the training script
CMD ["poetry", "run", "python", "train.py"]