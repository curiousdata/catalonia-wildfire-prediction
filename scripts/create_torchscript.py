# scripts/create_torchscript.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import segmentation_models_pytorch as smp


def _resolve_project_root() -> Path:
    # scripts/ is directly under the repo root in your project
    return Path(__file__).resolve().parents[1]


def _normalize_model_name(name: str) -> tuple[str, str]:
    """
    Returns:
      logical_name (no extension)
      filename (with .pth)
    """
    if name.endswith(".pth"):
        return name[:-4], name
    return name, f"{name}.pth"


def build_model(in_channels: int) -> torch.nn.Module:
    # Mirrors scripts/train.py
    encoder_name = "resnet34"
    decoder_dropout = 0.10

    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=1,
        activation=None,
        decoder_dropout=decoder_dropout,
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a trained Unet checkpoint to TorchScript for deployment.")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model checkpoint name in (root)/models (with or without .pth), e.g. resnet34_v9",
    )
    parser.add_argument(
        "--out_name",
        type=str,
        default=None,
        help="Optional output base name (no extension). Default: same as model_name.",
    )
    parser.add_argument(
        "--h",
        type=int,
        default=115,
        help="Dummy input height for tracing (does not need to match training exactly).",
    )
    parser.add_argument(
        "--w",
        type=int,
        default=149,
        help="Dummy input width for tracing (does not need to match training exactly).",
    )
    args = parser.parse_args()

    project_root = _resolve_project_root()

    logical_name, ckpt_filename = _normalize_model_name(args.model_name)
    out_base = args.out_name or logical_name

    # --- Must match your training feature_vars list (scripts/train.py) ---
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
        # "is_near_fire",  # commented out in train.py
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
        # CLC Level-3 classes
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
        # CLC aggregated proportions
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
        # Year-aware population density
        "popdens",
    ]
    in_channels = len(feature_vars)

    weights_path = project_root / "models" / ckpt_filename
    if not weights_path.exists():
        print(f"[ERROR] Weights file not found: {weights_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = project_root / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{out_base}.torchscript"

    print(f"[INFO] Project root: {project_root}")
    print(f"[INFO] Loading weights: {weights_path}")
    print(f"[INFO] Exporting TorchScript to: {out_path}")
    print(f"[INFO] in_channels={in_channels}, dummy_input=(1,{in_channels},{args.h},{args.w})")

    # Build model and load weights (state_dict)
    model = build_model(in_channels=in_channels)
    state = torch.load(weights_path, map_location="cpu")
    state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing or unexpected:
        print(f"[WARN] Missing keys: {len(missing)}; Unexpected keys: {len(unexpected)}")
        # Uncomment to print details:
        # print("[WARN] Missing:", missing)
        # print("[WARN] Unexpected:", unexpected)

    model.eval()

    # Trace to TorchScript
    example = torch.randn(1, in_channels, args.h, args.w, dtype=torch.float32)
    with torch.no_grad():
        ts = torch.jit.trace(model, example)
        ts = torch.jit.freeze(ts.eval())

    ts.save(str(out_path))
    print("[OK] TorchScript saved.")


if __name__ == "__main__":
    main()