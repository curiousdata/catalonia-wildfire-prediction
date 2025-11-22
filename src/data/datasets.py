import zarr
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from typing import List, Dict, Literal, Optional


class CNNIberFireDataset(Dataset):
    """
    PyTorch Dataset for IberFire wildfire prediction.
    
    Loads data directly from Zarr format with on-the-fly normalization.
    Supports both tile-level classification and pixel-level segmentation.
    Includes temporal stratified sampling to handle class imbalance.
    
    Args:
        zarr_path: Path to IberFire.zarr directory
        time_start: Start date (e.g., "2018-01-01")
        time_end: End date (e.g., "2020-12-31")
        feature_vars: List of feature variable names
        label_var: Label variable name (e.g., "is_near_fire")
        spatial_downsample: Spatial downsampling factor (e.g., 4 for 4x4 pooling)
        task: "tile_classification" or "segmentation"
        sample_strategy: "all", "stratified", or "balanced"
        fire_oversample_ratio: Oversampling ratio for fire events (only for stratified)
        stats_path: Optional path to precomputed normalization stats JSON
        compute_stats: Whether to compute stats if not provided
    
    Example:
        >>> dataset = CNNIberFireDataset(
        ...     zarr_path="data/processed/IberFire.zarr",
        ...     time_start="2018-01-01",
        ...     time_end="2020-12-31",
        ...     feature_vars=["wind_speed_mean", "t2m_mean", "RH_mean"],
        ...     label_var="is_near_fire",
        ...     task="tile_classification",
        ...     sample_strategy="stratified"
        ... )
        >>> loader = DataLoader(dataset, batch_size=8, shuffle=True)
    """
    
    def __init__(
        self,
        zarr_path: str,
        time_start: str,
        time_end: str,
        feature_vars: List[str],
        label_var: str,
        spatial_downsample: int = 4,
        task: Literal["tile_classification", "segmentation"] = "tile_classification",
        sample_strategy: Literal["all", "stratified", "balanced"] = "stratified",
        fire_oversample_ratio: float = 3.0,
        stats_path: Optional[str] = None,
        compute_stats: bool = True,
    ):
        self.zarr_path = Path(zarr_path)
        self.feature_vars = feature_vars
        self.label_var = label_var
        self.downsample = spatial_downsample
        self.task = task
        self.sample_strategy = sample_strategy
        self.fire_oversample_ratio = fire_oversample_ratio
        
        # Open Zarr dataset (read-only)
        print(f"Opening Zarr dataset: {self.zarr_path}")
        self.root = zarr.open(str(self.zarr_path), mode="r")
        
        # Get time indices for the specified range
        print(f"Filtering time range: {time_start} to {time_end}")
        time = self.root["time"][:]
        mask = (time >= np.datetime64(time_start)) & (time <= np.datetime64(time_end))
        all_indices = np.where(mask)[0]
        print(f"Total time steps in range: {len(all_indices)}")
        
        # Separate fire vs non-fire time steps
        self._analyze_fire_events(all_indices)
        
        # Apply sampling strategy
        self.time_indices = self._apply_sampling_strategy(all_indices)
        print(f"Final dataset size: {len(self.time_indices)} samples")
        
        # Load or compute normalization stats
        self.stats = self._load_or_compute_stats(stats_path, compute_stats)
    
    def _analyze_fire_events(self, all_indices):
        """Analyze which time steps have fire events."""
        print("Analyzing fire events...")
        self.fire_indices = []
        self.no_fire_indices = []
        
        for idx in all_indices:
            y = self.root[self.label_var][idx]
            if np.any(y > 0.5):  # any fire pixels
                self.fire_indices.append(idx)
            else:
                self.no_fire_indices.append(idx)
        
        n_fire = len(self.fire_indices)
        n_no_fire = len(self.no_fire_indices)
        total = n_fire + n_no_fire
        
        print(f"  Fire days: {n_fire} ({100*n_fire/total:.2f}%)")
        print(f"  No-fire days: {n_no_fire} ({100*n_no_fire/total:.2f}%)")
        if n_fire > 0:
            print(f"  Imbalance ratio: {n_no_fire/n_fire:.1f}:1")
    
    def _apply_sampling_strategy(self, all_indices):
        """Apply sampling strategy to handle class imbalance."""
        if self.sample_strategy == "all":
            print("Using all samples (no resampling)")
            return all_indices
        
        elif self.sample_strategy == "stratified":
            # Oversample fire events
            n_fire_samples = int(len(self.fire_indices) * self.fire_oversample_ratio)
            sampled_fire = np.random.choice(
                self.fire_indices,
                size=min(n_fire_samples, len(self.fire_indices) * 10),  # cap at 10x
                replace=True
            )
            sampled_no_fire = np.array(self.no_fire_indices)
            
            indices = np.concatenate([sampled_fire, sampled_no_fire])
            np.random.shuffle(indices)
            
            print(f"Stratified sampling: {len(sampled_fire)} fire + {len(sampled_no_fire)} no-fire")
            return indices
        
        elif self.sample_strategy == "balanced":
            # Equal fire/no-fire samples
            n_samples = min(len(self.fire_indices), len(self.no_fire_indices))
            sampled_fire = np.random.choice(self.fire_indices, size=n_samples, replace=False)
            sampled_no_fire = np.random.choice(self.no_fire_indices, size=n_samples, replace=False)
            
            indices = np.concatenate([sampled_fire, sampled_no_fire])
            np.random.shuffle(indices)
            
            print(f"Balanced sampling: {n_samples} fire + {n_samples} no-fire")
            return indices
        
        else:
            raise ValueError(f"Unknown sample_strategy: {self.sample_strategy}")
    
    def _load_or_compute_stats(self, stats_path, compute_stats):
        """Load precomputed stats or compute from data."""
        if stats_path and Path(stats_path).exists():
            print(f"Loading normalization stats from: {stats_path}")
            with open(stats_path) as f:
                stats = json.load(f)
            print(f"  Loaded stats for {len(stats)} variables")
            return stats
        
        elif compute_stats:
            print("Computing normalization stats from training data...")
            stats = {}
            for v in self.feature_vars:
                # Sample 100 time steps for efficiency
                sample_indices = np.random.choice(
                    self.time_indices,
                    size=min(100, len(self.time_indices)),
                    replace=False
                )
                
                data = np.concatenate([
                    self.root[v][idx].flatten() 
                    for idx in sample_indices
                ])
                
                mean = float(np.nanmean(data))
                std = float(np.nanstd(data))
                stats[v] = {
                    "mean": mean,
                    "std": std if std > 1e-6 else 1.0
                }
                print(f"  {v}: mean={mean:.4f}, std={std:.4f}")
            
            return stats
        
        else:
            print("⚠️  No stats provided and compute_stats=False. Using mean=0, std=1.")
            return {v: {"mean": 0.0, "std": 1.0} for v in self.feature_vars}
    
    def get_pos_weight(self):
        """
        Compute pos_weight for BCEWithLogitsLoss.
        
        Returns:
            float: Ratio of negative to positive samples/pixels
        """
        if self.task == "tile_classification":
            n_fire = len(self.fire_indices)
            n_no_fire = len(self.no_fire_indices)
            pos_weight = n_no_fire / max(1, n_fire)
        
        else:  # segmentation
            # Sample to estimate pixel ratio
            sample_indices = np.random.choice(
                self.fire_indices + self.no_fire_indices,
                size=min(100, len(self.time_indices)),
                replace=False
            )
            
            total_fire_pixels = 0
            total_no_fire_pixels = 0
            
            for idx in sample_indices:
                y = self.root[self.label_var][idx, ::self.downsample, ::self.downsample]
                total_fire_pixels += (y > 0.5).sum()
                total_no_fire_pixels += (y <= 0.5).sum()
            
            pos_weight = total_no_fire_pixels / max(1, total_fire_pixels)
        
        return float(pos_weight)
    
    def __len__(self):
        return len(self.time_indices)
    
    def __getitem__(self, idx):
        """
        Get a single sample.
        
        Returns:
            X: Tensor of shape [C, H, W] (features)
            y: Tensor of shape [1] (tile classification) or [1, H, W] (segmentation)
        """
        t = self.time_indices[idx]
        
        # Load and normalize features
        X_arrays = []
        for v in self.feature_vars:
            arr = self.root[v][t, ::self.downsample, ::self.downsample]
            
            # Normalize
            mean = self.stats[v]["mean"]
            std = self.stats[v]["std"]
            arr = (arr - mean) / std
            
            X_arrays.append(arr)
        
        X = np.stack(X_arrays, axis=0).astype("float32")  # [C, H, W]
        
        # Load label
        y = self.root[self.label_var][t, ::self.downsample, ::self.downsample]
        y_bin = (y > 0.5).astype("float32")
        
        if self.task == "tile_classification":
            # Tile-level: any fire in tile?
            y_out = np.array([1.0 if y_bin.sum() > 0 else 0.0], dtype="float32")
        else:  # segmentation
            # Pixel-level: fire mask
            y_out = y_bin[np.newaxis, ...]  # [1, H, W]
        
        return torch.from_numpy(X), torch.from_numpy(y_out)
    
    def save_stats(self, path: str):
        """Save normalization stats to JSON file."""
        with open(path, "w") as f:
            json.dump(self.stats, f, indent=2)
        print(f"Saved normalization stats to: {path}")
    
    def get_sample_info(self, idx):
        """Get metadata for a sample (for debugging)."""
        t = self.time_indices[idx]
        time_value = self.root["time"][t]
        has_fire = t in self.fire_indices
        
        return {
            "index": idx,
            "time_index": t,
            "time_value": str(time_value),
            "has_fire": has_fire,
        }


# Example usage and testing
if __name__ == "__main__":
    # Example configuration
    dataset = CNNIberFireDataset(
        zarr_path="data/processed/IberFire.zarr",
        time_start="2018-01-01",
        time_end="2020-12-31",
        feature_vars=[
            "wind_speed_mean",
            "t2m_mean",
            "RH_mean",
            "total_precipitation_mean",
        ],
        label_var="is_near_fire",
        spatial_downsample=4,
        task="tile_classification",
        sample_strategy="stratified",
        fire_oversample_ratio=3.0,
    )
    
    # Print dataset info
    print(f"\nDataset size: {len(dataset)}")
    print(f"pos_weight for loss: {dataset.get_pos_weight():.2f}")
    
    # Test loading a sample
    X, y = dataset[0]
    print(f"\nSample 0:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  X range: [{X.min():.3f}, {X.max():.3f}]")
    print(f"  y value: {y.item()}")
    
    # Save stats for later use
    dataset.save_stats("data/processed/stats.json")
    
    # Test with DataLoader
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    
    X_batch, y_batch = next(iter(loader))
    print(f"\nBatch:")
    print(f"  X_batch shape: {X_batch.shape}")
    print(f"  y_batch shape: {y_batch.shape}")
    print(f"  Fire tiles in batch: {y_batch.sum().item()}")