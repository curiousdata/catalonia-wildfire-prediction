import argparse
import time
import sys
from pathlib import Path

# Add project root to path BEFORE importing from src
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Now import from src
import segmentation_models_pytorch as smp
import torch
import tqdm
from torch.utils.data import DataLoader
from src.data.datasets import SimpleIberFireSegmentationDataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model file inside models/")
    parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs")
    args = parser.parse_args()

    model_name = args.model_name

    ZARR_PATH = project_root / "data" / "silver" / "IberFire.zarr"

    feature_vars = [
        "wind_speed_mean",
        "t2m_mean",
        "RH_mean",
        "total_precipitation_mean",
        "is_holiday",
        "popdens_2020",
        "dist_to_railways_mean",
        "dist_to_roads_mean",
        "slope_mean",
        "CLC_2018_forest_proportion",
    ]

    in_channels = len(feature_vars)
    TRAIN_STATS_PATH = project_root / "stats" / "simple_iberfire_stats_train.json"
    train_ds = SimpleIberFireSegmentationDataset(
        zarr_path=ZARR_PATH,
        time_start="2015-01-01",
        time_end="2020-12-31",
        feature_vars=feature_vars,
        label_var="is_near_fire",
        spatial_downsample=6,
        lead_time=0,
        compute_stats=True,
        stats_path=TRAIN_STATS_PATH,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        #persistent_workers=True,
        pin_memory=False,
    )

    X = train_ds[0][0].unsqueeze(0)
    y = train_ds[0][1].unsqueeze(0)
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
        spatial_downsample=6,
        lead_time=0,
        compute_stats=False,
        stats_path=TRAIN_STATS_PATH,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        #persistent_workers=True,
        pin_memory=False,
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = smp.Unet(
        encoder_name="mobilenet_v2",
        encoder_weights=None,
        in_channels=in_channels,
        classes=1,
        activation=None,
    )

    model = model.to(device)
    pos_weight = torch.tensor([10.0], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    checkpoint_path = project_root / "models" / model_name

    if checkpoint_path.exists():
        print(f"Loading existing model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
    else:
        print(f"No model found at {checkpoint_path}. Initializing new model.")
        os.makedirs(checkpoint_path.parent, exist_ok=True)
        start_epoch = 0

    NUM_EPOCHS = args.epochs
    overall_start = time.time()
    model.train()
    for epoch in range(NUM_EPOCHS):
        train_loss = 0.0
        pbar = tqdm.tqdm(
            train_loader,
            desc=f"Epoch {start_epoch + epoch + 1}/{start_epoch + NUM_EPOCHS}",
            ncols=100,
            file=sys.stdout,  # force stdout
            dynamic_ncols=False,  # fixed width
        )
        for X_batch, y_batch in pbar:
            X_batch = X_batch.to(device).float()
            y_batch = y_batch.to(device).float()

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss /= len(train_loader.dataset)
        print(f"\nEpoch {start_epoch + epoch + 1}/{start_epoch + NUM_EPOCHS}, Training Loss: {train_loss:.4f}")

        # Validation after each epoch
        model.eval()
        with torch.no_grad():
            test_loss = 0.0
            test_pbar = tqdm.tqdm(
                test_loader,
                desc="Validation",
                ncols=100,
                file=sys.stdout,
                dynamic_ncols=False,
            )
            for X_val, y_val in test_pbar:
                X_val = X_val.to(device).float()
                y_val = y_val.to(device).float()
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val)
                test_loss += val_loss.item() * X_val.size(0)
                test_pbar.set_postfix({"val_loss": f"{val_loss.item():.4f}"})
        
        test_loss /= len(test_loader.dataset)
        print(f"Epoch {start_epoch + epoch + 1}: Test Loss: {test_loss:.4f}\n")
        model.train()

    # Save the model
    checkpoint = {
        "epoch": start_epoch + NUM_EPOCHS,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)
    total_duration = time.time() - overall_start
    print(f"Total training time: {total_duration:.2f} seconds")
    print(f"Average time per epoch: {(total_duration / NUM_EPOCHS):.2f} seconds")
    print(f"Model saved to {checkpoint_path}")