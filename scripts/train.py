import argparse
import time
import sys
import os
from pathlib import Path
import mlflow
import mlflow.pytorch
from sklearn.metrics import roc_auc_score
import numpy as np

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

    mlflow.set_experiment("iberfire_unet_experiments")
    with mlflow.start_run(run_name=model_name):

        mlflow.log_param("model_name", model_name)

        ZARR_PATH = project_root / "data" / "silver" / "IberFire_time1_xyfull.zarr"
        mlflow.log_param("zarr_path", str(ZARR_PATH))

        train_time_start = "2015-01-01"
        train_time_end = "2020-12-31"
        val_time_start = "2021-01-01"
        val_time_end = "2021-12-31"
        spatial_downsample = 4
        lead_time = 0
        mlflow.log_param("train_time_start", train_time_start)
        mlflow.log_param("train_time_end", train_time_end)
        mlflow.log_param("val_time_start", val_time_start)
        mlflow.log_param("val_time_end", val_time_end)
        mlflow.log_param("spatial_downsample", spatial_downsample)
        mlflow.log_param("lead_time", lead_time)

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
        mlflow.log_param("architecture", f"Unet(mobilenet_v2,in={in_channels})")
        mlflow.log_param("encoder_name", "mobilenet_v2")
        mlflow.log_param("in_channels", in_channels)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("feature_vars", ",".join(feature_vars))
        lr = 1e-4
        mlflow.log_param("lr", lr)

        TRAIN_STATS_PATH = project_root / "stats" / "simple_iberfire_stats_train.json"
        train_ds = SimpleIberFireSegmentationDataset(
            zarr_path=ZARR_PATH,
            time_start=train_time_start,
            time_end=train_time_end,
            feature_vars=feature_vars,
            label_var="is_near_fire",
            spatial_downsample=spatial_downsample,
            lead_time=lead_time,
            compute_stats=True,
            stats_path=TRAIN_STATS_PATH,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=1,
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
            label_var="is_near_fire",
            spatial_downsample=spatial_downsample,
            lead_time=lead_time,
            compute_stats=False,
            stats_path=TRAIN_STATS_PATH,
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
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
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        checkpoint_path = project_root / "models" / model_name

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
        for epoch in range(NUM_EPOCHS):
            train_loss = 0.0
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

                train_loss += loss.item() * X_batch.size(0)
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            train_loss /= len(train_loader.dataset)
            mlflow.log_metric("train_loss", train_loss, step= epoch + 1)

            model.eval()
            with torch.no_grad():
                test_loss = 0.0
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
                    test_loss += val_loss.item() * X_val.size(0)
                    test_pbar.set_postfix({"val_loss": f"{val_loss.item():.4f}"})

                    # collect probabilities and targets for metrics
                    probs_batch = torch.sigmoid(val_outputs).detach().cpu().view(-1)
                    targets_batch = y_val.detach().cpu().view(-1)
                    all_probs.append(probs_batch)
                    all_targets.append(targets_batch)

            test_loss /= len(test_loader.dataset)

            # concatenate all batches
            all_probs = torch.cat(all_probs).numpy()
            all_targets = torch.cat(all_targets).numpy()

            # Compute precision, recall, F1 (threshold = 0.5)
            preds = (all_probs > 0.5).astype(np.int32)
            tp = float(((preds == 1) & (all_targets == 1)).sum())
            fp = float(((preds == 1) & (all_targets == 0)).sum())
            fn = float(((preds == 0) & (all_targets == 1)).sum())

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            # ROC-AUC over full validation set
            try:
                roc_auc = roc_auc_score(all_targets, all_probs)
            except ValueError:
                roc_auc = float("nan")

            mlflow.log_metric("val_loss", test_loss, step=epoch + 1)
            mlflow.log_metric("precision", precision, step=epoch + 1)
            mlflow.log_metric("recall", recall, step=epoch + 1)
            mlflow.log_metric("f1_score", f1, step=epoch + 1)
            mlflow.log_metric("roc_auc", roc_auc, step=epoch + 1)

            model.train()

        # Save the model weights locally for resume training
        torch.save(model.state_dict(), checkpoint_path)

        # Log the model to MLflow for traceability and later loading
        # Use `name` (artifact_path is deprecated) and provide an input_example
        input_example = sample_X.unsqueeze(0).to("cpu").float()
        mlflow.pytorch.log_model(
            model,
            name="unet_model",
            input_example=input_example,
        )
        total_duration = time.time() - overall_start
        print(f"Total training time: {total_duration:.2f} seconds")
        print(f"Average time per epoch: {(total_duration / NUM_EPOCHS):.2f} seconds")
        print(f"Model saved to {checkpoint_path}")