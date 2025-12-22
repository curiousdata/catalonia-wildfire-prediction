import shutil
from pathlib import Path

import xarray as xr
from dask.diagnostics import ProgressBar
from numcodecs import Blosc
import argparse 

# Collect arguments for coarsen factor
parser = argparse.ArgumentParser(description="Coarsen Zarr dataset.")
parser.add_argument(
    "--factor",
    type=int,
    default=32,
    help="Coarsening factor for spatial dimensions (default: 32)",
)

# Configuration
OLD_ZARR = Path("data/silver/IberFire.zarr")
OUT_DIR = Path("data/gold")
COARSEN_FACTOR = 32

NEW_ZARR = OUT_DIR / f"IberFire_coarse{COARSEN_FACTOR}_time1.zarr"
LABEL_VARS = ["is_fire"]

COMPRESSOR = Blosc(cname="zstd", clevel=3, shuffle=2)


def main():
    if not OLD_ZARR.exists():
        raise FileNotFoundError(f"Source Zarr not found: {OLD_ZARR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if NEW_ZARR.exists():
        resp = input(f"{NEW_ZARR} exists. Overwrite? [y/N]: ").strip().lower()
        if resp not in ("y", "yes"):
            print("Cancelled.")
            return
        shutil.rmtree(NEW_ZARR)

    ds = xr.open_zarr(OLD_ZARR, consolidated=True)

    coarsener = ds.coarsen(y=COARSEN_FACTOR, x=COARSEN_FACTOR, boundary="trim")
    ds_coarse = coarsener.mean()

    for lv in LABEL_VARS:
        if lv in ds:
            ds_coarse[lv] = ds[lv].coarsen(
                y=COARSEN_FACTOR, x=COARSEN_FACTOR, boundary="trim"
            ).max()

    chunks = {
        "time": 1,
        "y": ds_coarse.sizes["y"],
        "x": ds_coarse.sizes["x"],
    }
    ds_coarse = ds_coarse.chunk(chunks)
    encoding = {name: {"compressor": COMPRESSOR} for name in ds_coarse.data_vars}

    with ProgressBar():
        ds_coarse.to_zarr(NEW_ZARR, mode="w", encoding=encoding, consolidated=True)

    xr.open_zarr(NEW_ZARR, consolidated=True)
    print("Coarsening complete:", NEW_ZARR)


if __name__ == "__main__":
    main()