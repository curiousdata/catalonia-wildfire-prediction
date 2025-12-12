import argparse
import time
import sys
import os
from pathlib import Path
import mlflow
import mlflow.pytorch
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

# Add project root to path BEFORE importing from src
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Now import from src
import segmentation_models_pytorch as smp
import torch
import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Focal loss for binary classification with logits.

        Args:
            logits: raw model outputs (before sigmoid), shape (N, 1, H, W) or similar.
            targets: binary targets in {0, 1}, same shape as logits.
        """
        # Ensure targets are float
        targets = targets.type_as(logits)
        # Binary cross entropy with logits, no reduction
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        # Convert to p_t as in the focal loss paper
        p_t = torch.exp(-bce)
        focal_term = (1 - p_t) ** self.gamma
        loss = self.alpha * focal_term * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


from src.data.datasets import SimpleIberFireSegmentationDataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model file inside models/")
    parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs")
    args = parser.parse_args()

    model_name = args.model_name
    # Ensure we have a consistent local filename with .pth extension
    if model_name.endswith(".pth"):
        model_file_name = model_name
        model_name = model_name[:-4]  # logical name without extension
    else:
        model_file_name = model_name + ".pth"

    mlflow.set_experiment("iberfire_unet_experiments")
    with mlflow.start_run(run_name=model_name):

        mlflow.log_param("model_name", model_name)

        ZARR_PATH = project_root / "data" / "gold" / "IberFire_coarse32_time1.zarr"

        mlflow.log_param("zarr_path", str(ZARR_PATH))
        mlflow.log_param("coarsen_factor", 32)

        train_time_start = "2008-01-01"
        train_time_end = "2022-12-31"
        val_time_start = "2023-01-01"
        val_time_end = "2024-12-31"
        spatial_downsample = 1
        lead_time = 1
        batch_size = 8
        mlflow.log_param("train_time_start", train_time_start)
        mlflow.log_param("train_time_end", train_time_end)
        mlflow.log_param("val_time_start", val_time_start)
        mlflow.log_param("val_time_end", val_time_end)
        mlflow.log_param("spatial_downsample", spatial_downsample)
        mlflow.log_param("lead_time", lead_time)
        mlflow.log_param("batch_size", batch_size)

        feature_vars = [
            # Dynamic features (time-dependent)
            "FAPAR",
            "FWI",
            "LAI",
            "LST",
            "NDVI",
            "RH_max",
            "RH_mean",
            "RH_min",
            "RH_range",
            "SWI_001",
            "SWI_005",
            "SWI_010",
            "SWI_020",
            "is_holiday",
            "is_near_fire",
            "surface_pressure_max",
            "surface_pressure_mean",
            "surface_pressure_min",
            "surface_pressure_range",
            "t2m_max",
            "t2m_mean",
            "t2m_min",
            "t2m_range",
            "total_precipitation_mean",
            "wind_direction_at_max_speed",
            "wind_direction_mean",
            "wind_speed_max",
            "wind_speed_mean",
            # CLC Level-3 classes (year-aware bases)
            "CLC_1",
            "CLC_2",
            "CLC_3",
            "CLC_4",
            "CLC_5",
            "CLC_6",
            "CLC_7",
            "CLC_8",
            "CLC_9",
            "CLC_10",
            "CLC_11",
            "CLC_12",
            "CLC_13",
            "CLC_14",
            "CLC_15",
            "CLC_16",
            "CLC_17",
            "CLC_18",
            "CLC_19",
            "CLC_20",
            "CLC_21",
            "CLC_22",
            "CLC_23",
            "CLC_24",
            "CLC_25",
            "CLC_26",
            "CLC_27",
            "CLC_28",
            "CLC_29",
            "CLC_30",
            "CLC_31",
            "CLC_32",
            "CLC_33",
            "CLC_34",
            "CLC_35",
            "CLC_36",
            "CLC_37",
            "CLC_38",
            "CLC_39",
            "CLC_40",
            "CLC_41",
            "CLC_42",
            "CLC_43",
            "CLC_44",
            # CLC aggregated proportions (year-aware bases)
            "CLC_agricultural_proportion",
            "CLC_arable_land_proportion",
            "CLC_artificial_proportion",
            "CLC_artificial_vegetation_proportion",
            "CLC_forest_and_semi_natural_proportion",
            "CLC_forest_proportion",
            "CLC_heterogeneous_agriculture_proportion",
            "CLC_industrial_proportion",
            "CLC_inland_waters_proportion",
            "CLC_inland_wetlands_proportion",
            "CLC_marine_waters_proportion",
            "CLC_maritime_wetlands_proportion",
            "CLC_mine_proportion",
            "CLC_open_space_proportion",
            "CLC_permanent_crops_proportion",
            "CLC_scrub_proportion",
            "CLC_urban_fabric_proportion",
            "CLC_waterbody_proportion",
            "CLC_wetlands_proportion",
            # Other static features
            "aspect_1",
            "aspect_2",
            "aspect_3",
            "aspect_4",
            "aspect_5",
            "aspect_6",
            "aspect_7",
            "aspect_8",
            "aspect_NODATA",
            "dist_to_railways_mean",
            "dist_to_railways_stdev",
            "dist_to_roads_mean",
            "dist_to_roads_stdev",
            "dist_to_waterways_mean",
            "dist_to_waterways_stdev",
            "elevation_mean",
            "elevation_stdev",
            "is_natura2000",
            "is_sea",
            "is_spain",
            "is_waterbody",
            "roughness_mean",
            "roughness_stdev",
            "slope_mean",
            "slope_stdev",
            # Year-aware population density (popdens_YYYY family)
            "popdens",
        ]

        in_channels = len(feature_vars)
        mlflow.log_param("architecture", f"Unet({encoder_name},imagenet,in={in_channels})")
        mlflow.log_param("in_channels", in_channels)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("feature_vars", ",".join(feature_vars))
        lr = 1e-4
        weight_decay = 2e-3
        decoder_dropout = 0.10  # try 0.20 next if still overfitting
        encoder_name = "resnet34"
        mlflow.log_param("encoder_name", encoder_name)
        mlflow.log_param("lr", lr)
        mlflow.log_param("weight_decay", weight_decay)
        mlflow.log_param("decoder_dropout", decoder_dropout)

        TRAIN_STATS_PATH = project_root / "stats" / "simple_iberfire_stats_train.json"
        FIRE_DAY_INDICES_PATH = project_root / "stats" / "fire_day_indices.json"
        train_ds = SimpleIberFireSegmentationDataset(
            zarr_path=ZARR_PATH,
            time_start=train_time_start,
            time_end=train_time_end,
            feature_vars=feature_vars,
            label_var="is_fire",
            spatial_downsample=spatial_downsample,
            lead_time=lead_time,
            compute_stats=True,
            stats_path=TRAIN_STATS_PATH,
            mode="balanced_days",
            day_indices_path=FIRE_DAY_INDICES_PATH,
            balanced_ratio=1.0,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )

        # Lightweight dataset sanity check (single sample access)
        sample_X, sample_y = train_ds[0]
        assert len(train_ds) > 0, "Training dataset is empty!"
        assert sample_X.shape[0] == in_channels, f"Expected {in_channels} input channels, got {sample_X.shape[0]}"
        assert sample_y.shape[0] == 1, f"Expected 1 output channel, got {sample_y.shape[0]}"
        assert sample_X.shape[1:] == sample_y.shape[1:], "Input and output spatial dimensions do not match"

        # test dataset
        test_ds = SimpleIberFireSegmentationDataset(
            zarr_path=ZARR_PATH,
            time_start=val_time_start,
            time_end=val_time_end,
            feature_vars=feature_vars,
            label_var="is_fire",
            spatial_downsample=spatial_downsample,
            lead_time=lead_time,
            compute_stats=False,
            stats_path=TRAIN_STATS_PATH,
            mode="all",
            day_indices_path=FIRE_DAY_INDICES_PATH,
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        # ==========================
        # Sanity checks: positive ratios
        # ==========================
        def compute_pos_ratio(loader):
            total_pos = 0
            total_pixels = 0
            for _, yb in loader:
                total_pos += yb.sum().item()
                total_pixels += yb.numel()
            return total_pos, total_pixels, (total_pos / total_pixels if total_pixels > 0 else 0.0)

        train_pos, train_pix, train_ratio = compute_pos_ratio(train_loader)
        val_pos, val_pix, val_ratio = compute_pos_ratio(test_loader)

        print("=== SANITY CHECKS ===")
        print(f"Train positives: {train_pos} out of {train_pix} pixels (ratio={train_ratio:.8f})")
        print(f"Val positives:   {val_pos} out of {val_pix} pixels (ratio={val_ratio:.8f})")
        print("======================")

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=1,
            activation=None,
            decoder_dropout=decoder_dropout,
        )

        model = model.to(device)
        # Pos-weight based on observed pixel imbalance in the (effective) training loader.
        # For BCEWithLogitsLoss, a tensor of shape [1] correctly broadcasts to [B, 1, H, W].
        pos_weight_value = float((train_pix - train_pos) / (train_pos + 1e-6))
        pos_weight = torch.tensor([pos_weight_value], device=device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        mlflow.log_param("criterion", "BCEWithLogitsLoss")
        mlflow.log_param("pos_weight", pos_weight_value)
        mlflow.log_param("train_pos_ratio", float(train_ratio))
        mlflow.log_param("val_pos_ratio", float(val_ratio))
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        checkpoint_path = project_root / "models" / model_file_name

        if checkpoint_path.exists():
            print(f"Loading existing model from {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
        else:
            print(f"No model found at {checkpoint_path}. Initializing new model.")
            os.makedirs(checkpoint_path.parent, exist_ok=True)

        NUM_EPOCHS = args.epochs
        overall_start = time.time()
        model.train()
        best_val_loss = float("inf")
        patience = 5
        min_epochs_before_stop = 10
        epochs_no_improve = 0
        best_model_state = None
        for epoch in range(NUM_EPOCHS):
            train_loss_sum = 0.0
            train_pixels = 0
            pbar = tqdm.tqdm(
                train_loader,
                desc=f"Epoch: {epoch + 1}/{NUM_EPOCHS}",
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

                train_loss_sum += loss.item() * y_batch.numel()
                train_pixels += y_batch.numel()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            train_loss_per_pixel = train_loss_sum / max(train_pixels, 1)
            mlflow.log_metric("train_loss_per_pixel", train_loss_per_pixel, step=epoch + 1)

            model.eval()
            with torch.no_grad():
                test_loss_sum = 0.0
                val_pixels = 0
                all_probs = []
                all_targets = []

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
                    test_loss_sum += val_loss.item() * y_val.numel()
                    val_pixels += y_val.numel()
                    test_pbar.set_postfix({"val_loss": f"{val_loss.item():.4f}"})

                    # collect probabilities and targets for metrics
                    probs_batch = torch.sigmoid(val_outputs).detach().cpu().view(-1)
                    targets_batch = y_val.detach().cpu().view(-1)
                    all_probs.append(probs_batch)
                    all_targets.append(targets_batch)

            test_loss = test_loss_sum / max(val_pixels, 1)

            # concatenate all batches
            all_probs = torch.cat(all_probs).numpy()
            all_targets = torch.cat(all_targets).numpy()

            # ROC-AUC and PR-AUC over full validation set (threshold-free)
            try:
                roc_auc = roc_auc_score(all_targets, all_probs)
            except ValueError:
                roc_auc = float("nan")
            try:
                pr_auc = average_precision_score(all_targets, all_probs)
            except ValueError:
                pr_auc = float("nan")

            mlflow.log_metric("val_loss_per_pixel", test_loss, step=epoch + 1)
            mlflow.log_metric("validation_roc_auc", roc_auc, step=epoch + 1)
            mlflow.log_metric("validation_pr_auc", pr_auc, step=epoch + 1)

            # Early stopping and best-model tracking based on validation loss
            if test_loss < best_val_loss - 1e-4:
                best_val_loss = test_loss
                epochs_no_improve = 0
                best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                # Only start early-stopping checks after a minimum number of epochs
                if (epoch + 1) >= min_epochs_before_stop:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print(f"Early stopping triggered at epoch {epoch + 1}")
                        break

            model.train()

        # Restore best model (by validation loss) if early stopping was triggered
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Save the model weights locally for resume training
        torch.save(model.state_dict(), checkpoint_path)

        # Move model to CPU for MLflow logging and signature inference
        model_cpu = model.to("cpu").eval()

        # Log the model to MLflow for traceability and later loading
        input_example = sample_X.unsqueeze(0).cpu().numpy().astype("float32")
        mlflow.pytorch.log_model(
            model_cpu,
            name=model_name,
            input_example=input_example,
        )
        total_duration = time.time() - overall_start
        epochs_ran = (epoch + 1) if "epoch" in locals() else 0
        avg_time = total_duration / epochs_ran if epochs_ran > 0 else 0.0
        print(f"Total training time: {total_duration:.2f} seconds")
        print(f"Average time per epoch: {avg_time:.2f} seconds over {epochs_ran} epochs")
        print(f"Model saved to {checkpoint_path}")