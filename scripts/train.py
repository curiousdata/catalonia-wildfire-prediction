"""
Training script for a machine learning model using PyTorch.
This script handles data loading, model training, validation, and saving the trained model and artifacts.

Arguments: 
- model_path: The local model to train.
- epochs: Number of training epochs.
"""

# Setup
import sys
from pathlib import Path
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch
import tqdm
from src.data.datasets import SimpleIberFireSegmentationDataset
from torch.utils.data import DataLoader

project_root = Path.cwd().parent
sys.path.insert(0, str(project_root))

from torch.utils.data import DataLoader

ZARR_PATH = project_root / "data" / "silver" / "IberFire.zarr"

feature_vars = [
    "wind_speed_mean",
    "t2m_mean",
    "RH_mean",
    "total_precipitation_mean",
    "is_holiday",
    "popdens_2020",
    "elevation_mean",
    "dist_to_railways_mean",
    "dist_to_roads_mean",
    "dist_to_waterways_mean",
    "roughness_mean",
    "slope_mean",
    "CLC_2018_forest_proportion",
    "CLC_2018_open_space_proportion",
]

in_channels = len(feature_vars)
TRAIN_STATS_PATH = project_root / "stats" / "simple_iberfire_stats_train.json"
train_ds = SimpleIberFireSegmentationDataset(
    zarr_path=ZARR_PATH,
    time_start="2015-01-01",
    time_end="2020-12-31",
    feature_vars=feature_vars,
    label_var="is_near_fire",
    spatial_downsample=4,
    lead_time=0,         # predict today
    compute_stats=True,  # or precompute & pass stats
    stats_path = TRAIN_STATS_PATH
)

train_loader = DataLoader(
    train_ds,
    batch_size=6,       
    shuffle=True,
    num_workers=4,
    persistent_workers=True,
    pin_memory=False
)

X = train_ds[0][0].unsqueeze(0)  # add batch dim
y = train_ds[0][1].unsqueeze(0)  # add batch dim
assert len(train_ds) > 0, "Training dataset is empty!"
assert X.shape[1] == in_channels, f"Expected {in_channels} input channels, got {X.shape[1]}"
assert y.shape[1] == 1, f"Expected 1 output channel, got {y.shape[1]}"
assert X.shape[2:] == y.shape[2:], "Input and output spatial dimensions do not match"

# test dataset
test_ds = SimpleIberFireSegmentationDataset(
    zarr_path=ZARR_PATH,
    time_start="2021-01-01",
    time_end="2021-12-31",
    feature_vars=feature_vars,
    label_var="is_near_fire",
    spatial_downsample=4,
    lead_time=0,
    compute_stats=False,
    stats_path=TRAIN_STATS_PATH
)
test_loader = DataLoader(
    test_ds,
    batch_size=6,
    shuffle=False,
    num_workers=4,
    persistent_workers=True,
    pin_memory=False 
)

model = smp.Unet(
    encoder_name="resnet34",      # or "timm-efficientnet-b0", etc.
    encoder_weights="imagenet",          # or None if you don't want pretrained
    in_channels=in_channels,               # IberFire: number of feature channels per pixel
    classes=1,                    # 1 output channel for fire / no-fire probability
    activation=None               # we'll apply sigmoid later in the loss/metrics
)

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device)
pos_weight = torch.tensor([10.0], device=device)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Load the checkpoint
checkpoint_path = project_root / "models" / "unet_iberfire_.pth" #TODO: specify correct path from input
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state"])
optimizer.load_state_dict(checkpoint["optimizer_state"])

start_epoch = checkpoint["epoch"] + 1
print("Resuming from epoch:", start_epoch)

NUM_EPOCHS = 20
model.train()
for epoch in range(NUM_EPOCHS):
    train_loss = 0.0
    pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", ncols = 100)
    for X_batch, y_batch in pbar:
        X_batch = X_batch.to(device).float()
        y_batch = y_batch.to(device).float()

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * X_batch.size(0)

        pbar.set_postfix({"loss": loss.item()})

    train_loss /= len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Training Loss: {train_loss:.4f}")

    # Test the model's loss on the test set
model.eval()
test_loss = 0.0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device).float()
        y_batch = y_batch.to(device).float()

        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        test_loss += loss.item() * X_batch.size(0)
test_loss /= len(test_loader.dataset)
print(f"Test Loss: {test_loss:.4f}")

