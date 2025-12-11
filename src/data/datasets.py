import zarr
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from typing import List, Dict, Literal, Optional

import xarray as xr


# --- NumPy pooling helpers ---
def max_pool_np(arr: np.ndarray, k: int) -> np.ndarray:
    """
    Block max pooling for a 2D array with factor k.
    If k <= 1, returns the input array unchanged.
    """
    if k <= 1:
        return arr
    H, W = arr.shape
    Hk, Wk = H // k, W // k
    if Hk == 0 or Wk == 0:
        return arr
    arr_trim = arr[: Hk * k, : Wk * k]
    arr_reshaped = arr_trim.reshape(Hk, k, Wk, k)
    pooled = arr_reshaped.max(axis=(1, 3))
    return pooled


def mean_pool_np(arr: np.ndarray, k: int) -> np.ndarray:
    """
    Block mean pooling for a 2D array with factor k.
    If k <= 1, returns the input array unchanged.
    """
    if k <= 1:
        return arr
    H, W = arr.shape
    Hk, Wk = H // k, W // k
    if Hk == 0 or Wk == 0:
        return arr
    arr_trim = arr[: Hk * k, : Wk * k]
    arr_reshaped = arr_trim.reshape(Hk, k, Wk, k)
    pooled = arr_reshaped.mean(axis=(1, 3))
    return pooled

class SimpleIberFireSegmentationDataset(Dataset):
    """
    Minimal PyTorch Dataset for IberFire-style wildfire *segmentation* (heatmap output).

    This is a simpler MVP version focused on:
      - Full-image inputs (no tile-level classification)
      - Pixel-wise fire mask as target (for U-Net / FCN-style models)
      - Optional lead time (e.g., predict fire at t+1 from features at t)
      - Optional on-the-fly normalization

    Args:
        zarr_path: Path to IberFire.zarr directory
        time_start: Start date (e.g., "2018-01-01")
        time_end: End date (e.g., "2020-12-31")
        feature_vars: List of feature variable names
        label_var: Label variable name (e.g., "is_near_fire")
        spatial_downsample: Spatial downsampling factor (e.g., 4 for 4x4 pooling)
        lead_time: Predict label at t+lead_time (0 = same day, 1 = tomorrow, etc.)
        stats: Optional dict with precomputed normalization stats
               {var_name: {"mean": float, "std": float}}
        compute_stats: Whether to compute stats if not provided
        stats_path: Optional path to load/save normalization stats JSON.

    Usage (typical for U-Net MVP):
        >>> dataset = SimpleIberFireSegmentationDataset(
        ...     zarr_path="data/processed/IberFire.zarr",
        ...     time_start="2018-01-01",
        ...     time_end="2020-12-31",
        ...     feature_vars=["wind_speed_mean", "t2m_mean", "RH_mean"],
        ...     label_var="is_near_fire",
        ...     spatial_downsample=1,
        ...     lead_time=1,  # predict tomorrow's fire heatmap
        ...     compute_stats=True,
        ... )
    """

    def __init__(
        self,
        zarr_path: str,
        time_start: str,
        time_end: str,
        feature_vars: List[str],
        label_var: str,
        spatial_downsample: int = 1,
        lead_time: int = 1,
        stats: Optional[Dict[str, Dict[str, float]]] = None,
        compute_stats: bool = False,
        stats_path: Optional[str] = None,
        mode: str = "fire_only",
        day_indices_path: Optional[str] = None,
        balanced_ratio: float = 1.0,
        seed: int = 42,
    ):
        self.zarr_path = Path(zarr_path)
        self.feature_vars = feature_vars
        self.label_var = label_var
        self.downsample = spatial_downsample
        self.lead_time = lead_time
        if self.downsample != 1:
            raise ValueError(
                "SimpleIberFireSegmentationDataset expects spatial_downsample=1 "
                "when using the coarsened Zarr. Further pooling should be handled "
                "in a separate dataset variant."
            )
        self.mode = mode
        self.balanced_ratio = balanced_ratio
        self.seed = seed
        self.stats_path: Optional[Path] = Path(stats_path) if stats_path is not None else None
        self.day_indices_path: Optional[Path] = (
            Path(day_indices_path) if day_indices_path is not None else None
        )

        print(f"[SimpleDataset] Opening Zarr dataset: {self.zarr_path}")
        self.ds = xr.open_zarr(
            self.zarr_path,
            consolidated=True,
            decode_times=True,
            chunks="auto",
        )
        # Also open raw Zarr root for fast array access in __getitem__
        self.root = zarr.open(str(self.zarr_path), mode="r")

        print(f"[SimpleDataset] Filtering time range: {time_start} to {time_end}")
        time = self.ds["time"].values  # this is datetime64 already
        mask = (time >= np.datetime64(time_start)) & (time <= np.datetime64(time_end))
        all_indices = np.where(mask)[0]

        # Ensure we can look ahead by lead_time
        if self.lead_time > 0:
            max_valid = len(time) - self.lead_time - 1
            all_indices = all_indices[all_indices <= max_valid]

        if len(all_indices) == 0:
            raise ValueError("No valid time indices found for the given range and lead_time.")

        self.time_indices = all_indices
        print(f"[SimpleDataset] Total usable time steps: {len(self.time_indices)}")
        # Optionally adjust the list of time indices based on fire/no-fire day information
        self._apply_day_mode()
        print(f"[SimpleDataset] Time steps after mode='{self.mode}': {len(self.time_indices)}")

        # Determine which variables are dynamic (have time dim) vs static (no time dim)
        self.dynamic_vars: List[str] = []
        self.static_vars: List[str] = []
        for v in self.feature_vars:
            da = self.ds[v]
            if "time" in da.dims:
                self.dynamic_vars.append(v)
            else:
                self.static_vars.append(v)
        print(f"[SimpleDataset] Dynamic vars (time-dependent): {self.dynamic_vars}")
        print(f"[SimpleDataset] Static vars (no time dimension, broadcast in time): {self.static_vars}")

        # Cache static variables in memory to avoid repeated disk reads
        self.static_cache: Dict[str, np.ndarray] = {}
        for v in self.static_vars:
            arr = self.root[v][:, :].astype("float32")
            self.static_cache[v] = arr
            print(
                f"[SimpleDataset] Cached static var '{v}' with shape {arr.shape} "
                f"and dtype {arr.dtype}"
            )

        # Load or compute normalization stats
        if stats is not None:
            print("[SimpleDataset] Using provided normalization stats.")
            self.stats = stats

        elif compute_stats:
            overwrite = True

            # If a stats file already exists, ask whether to overwrite or reuse it
            if self.stats_path is not None and self.stats_path.exists():
                resp = input(
                    f"[SimpleDataset] Stats found at {self.stats_path}. Do you want to overwrite? [y/N]: "
                ).strip().lower()
                if resp not in ("y", "yes"):
                    print(f"[SimpleDataset] Keeping existing stats from: {self.stats_path}")
                    with open(self.stats_path) as f:
                        self.stats = json.load(f)
                    overwrite = False
                else:
                    print(f"[SimpleDataset] Overwriting stats at: {self.stats_path}")

            if overwrite:
                print("[SimpleDataset] Computing normalization stats from data...")
                self.stats = self._compute_stats()

                # If no explicit stats_path was provided, choose a sensible default:
                # <zarr_parent>/stats/simple_iberfire_stats.json
                if self.stats_path is None:
                    default_dir = self.zarr_path.parent / "stats"
                    self.stats_path = default_dir / "simple_iberfire_stats.json"

                self.save_stats(self.stats_path)

        elif self.stats_path is not None and self.stats_path.exists():
            print(f"[SimpleDataset] Loading normalization stats from: {self.stats_path}")
            with open(self.stats_path) as f:
                self.stats = json.load(f)

        else:
            print("[SimpleDataset] No stats provided, using mean=0, std=1 for all vars.")
            self.stats = {v: {"mean": 0.0, "std": 1.0} for v in self.feature_vars}

        # Cache aligned stats arrays for faster __getitem__
        self._means = np.array([self.stats[v]["mean"] for v in self.feature_vars], dtype="float32")
        self._stds = np.array(
            [max(self.stats[v]["std"], 1e-6) for v in self.feature_vars],
            dtype="float32",
        )

    def _compute_stats(self) -> Dict[str, Dict[str, float]]:
        """Compute simple per-variable mean/std from a subset of time steps."""
        stats: Dict[str, Dict[str, float]] = {}
        # Sample up to 100 time steps for efficiency
        sample_indices = np.random.choice(
            self.time_indices,
            size=min(100, len(self.time_indices)),
            replace=False,
        )

        for v in self.feature_vars:
            data_list = []
            if v in self.dynamic_vars:
                # Time-varying variable: sample across time
                for idx in sample_indices:
                    arr = self.root[v][idx, :, :]
                    data_list.append(arr.ravel())
            else:
                # Static variable: no time dimension, reuse cached array
                arr = self.static_cache[v]
                data_list.append(arr.ravel())
            data = np.concatenate(data_list)

            mean = float(np.nanmean(data))
            std = float(np.nanstd(data))
            if std < 1e-6:
                std = 1.0

            stats[v] = {"mean": mean, "std": std}
            print(f"[SimpleDataset] {v}: mean={mean:.4f}, std={std:.4f}")

        return stats

    def _apply_day_mode(self) -> None:
        """
        Modify self.time_indices according to self.mode and an optional
        precomputed fire/no-fire day index file (indices are interpreted in
        label-day space, i.e. days where the label at time t has fire/no fire).

        Modes:
            - "all": keep all time indices (no change)
            - "fire_only": keep only days that have at least one fire pixel
            - "balanced_days": keep all fire days plus a random sample of no-fire days
        """
        # If no mode requested or no day-index file provided, do nothing.
        if self.mode == "all" or self.day_indices_path is None:
            return

        if not self.day_indices_path.exists():
            print(
                f"[SimpleDataset] Day index file not found at {self.day_indices_path}, "
                f"mode='{self.mode}' will be ignored (using all time steps)."
            )
            return

        import json

        with self.day_indices_path.open("r") as f:
            data = json.load(f)

        fire_days_global = np.array(data.get("fire_days", []), dtype=int)
        no_fire_days_global = np.array(data.get("no_fire_days", []), dtype=int)

        if fire_days_global.size == 0 and self.mode in ("fire_only", "balanced_days"):
            print(
                "[SimpleDataset] No fire_days found in the index file; "
                "mode will be ignored and all time steps will be used."
            )
            return

        base_indices = np.array(self.time_indices, dtype=int)

        # Map feature-day indices (t) to their corresponding label-day indices (t + lead_time)
        # The JSON file is assumed to store fire/no-fire indices in label-day space.
        lead = int(getattr(self, "lead_time", 0))
        label_indices_for_features = base_indices + lead

        # A feature day is considered "fire" if its label day (t + lead_time) is in fire_days_global.
        fire_mask = np.isin(label_indices_for_features, fire_days_global)
        no_fire_mask = np.isin(label_indices_for_features, no_fire_days_global)

        fire_in_range = base_indices[fire_mask]
        no_fire_in_range = base_indices[no_fire_mask]

        if self.mode == "fire_only":
            if fire_in_range.size == 0:
                raise ValueError(
                    "Mode 'fire_only' selected, but no fire days found in the given time range."
                )
            self.time_indices = fire_in_range.tolist()
            return

        if self.mode == "balanced_days":
            if fire_in_range.size == 0:
                raise ValueError(
                    "Mode 'balanced_days' selected, but no fire days found in the given time range."
                )

            n_fire = fire_in_range.size
            n_no_fire_target = int(self.balanced_ratio * n_fire)

            if no_fire_in_range.size == 0:
                # Nothing to balance with; fall back to fire-only
                chosen_no_fire = np.array([], dtype=int)
            else:
                rng = np.random.default_rng(self.seed)
                n_no_fire_target = min(n_no_fire_target, no_fire_in_range.size)
                chosen_no_fire = rng.choice(
                    no_fire_in_range, size=n_no_fire_target, replace=False
                )

            combined = np.concatenate([fire_in_range, chosen_no_fire])

            # Shuffle combined indices to mix fire and no-fire days
            rng = np.random.default_rng(self.seed)
            combined = rng.permutation(combined)

            self.time_indices = combined.tolist()
            return

        raise ValueError(
            f"Unknown mode '{self.mode}'. Expected 'all', 'fire_only', or 'balanced_days'."
        )

    def __len__(self) -> int:
        # One sample per time step
        return len(self.time_indices)

    def __getitem__(self, idx: int):
        """
        Returns:
            X: Tensor of shape [C, H, W]  (features at time t)
            y: Tensor of shape [1, H, W]  (fire mask at time t + lead_time)
        """
        t = self.time_indices[idx]
        t_label = t + self.lead_time

        # Load and normalize features
        X_arrays = []
        for i, v in enumerate(self.feature_vars):
            if v in self.dynamic_vars:
                # Dynamic variable: read slice directly from coarsened Zarr
                arr = self.root[v][t, :, :]
            else:
                # Static variable: reuse cached array
                arr = self.static_cache[v]
            mean = self._means[i]
            std = self._stds[i]
            arr = (arr - mean) / std
            X_arrays.append(arr)

        X = np.stack(X_arrays, axis=0).astype("float32")  # [C, H, W]

        # Load label at t + lead_time directly from coarsened Zarr
        y = self.root[self.label_var][t_label, :, :]
        y_bin = (y > 0.5).astype("float32")[np.newaxis, ...]  # [1, H, W]

        return torch.from_numpy(X), torch.from_numpy(y_bin)

    def save_stats(self, path: str):
        """Save normalization stats to JSON file."""
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(path_obj, "w") as f:
            json.dump(self.stats, f, indent=2)
        print(f"[SimpleDataset] Saved normalization stats to: {path_obj}")

    def get_time_value(self, idx: int) -> str:
        """Return the datetime string for a given sample index (for debugging)."""
        t = self.time_indices[idx]
        time_value = self.ds["time"].values[t]
        return str(time_value)