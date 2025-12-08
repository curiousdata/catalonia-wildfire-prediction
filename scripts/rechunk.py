import xarray as xr
from pathlib import Path
from numcodecs import Blosc
from dask.diagnostics import ProgressBar
import shutil
import time
from datetime import datetime

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
OLD_ZARR = Path("data/silver/IberFire.zarr")
NEW_ZARR = Path("data/silver/IberFire_time1_xyfull.zarr")  # stays in silver for now

# For reference: original dims are (time=6241, y=920, x=1188)
CHUNKS = {
    "time": 1,    # 1 day per chunk → perfect for per-day training
    "y": 920,     # full spatial extent
    "x": 1188,
}

COMPRESSOR = Blosc(
    cname="zstd",  
    clevel=3,
    shuffle=2,
)


def main():
    print("=" * 70)
    print("Rechunk IberFire Zarr")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    if not OLD_ZARR.exists():
        raise FileNotFoundError(f"Source Zarr not found: {OLD_ZARR}")

    print(f"Source: {OLD_ZARR}")
    print(f"Target: {NEW_ZARR}")
    print(f"New chunks: {CHUNKS}")
    print(f"Compressor: {COMPRESSOR}\n")

    if NEW_ZARR.exists():
        resp = input(f"⚠️ {NEW_ZARR} already exists. Overwrite? [y/N]: ").strip().lower()
        if resp not in ("y", "yes"):
            print("Cancelled.")
            return
        print(f"Removing existing {NEW_ZARR}...")
        shutil.rmtree(NEW_ZARR)

    # 1) Open existing Zarr
    print("Step 1/3: Opening source Zarr...")
    t0 = time.time()
    ds = xr.open_zarr(OLD_ZARR, consolidated=True)
    print(f"  ✓ Opened in {time.time() - t0:.1f}s")
    print(f"  Dimensions: {dict(ds.dims)}")
    print(f"  Variables: {len(ds.data_vars)} data variables\n")

    # 2) Rechunk
    print(f"Step 2/3: Rechunking with {CHUNKS}...")
    t1 = time.time()
    ds_rechunked = ds.chunk(CHUNKS)
    print(f"  ✓ Rechunked graph built in {time.time() - t1:.1f}s")

    # 3) Write new Zarr
    print("Step 3/3: Writing rechunked dataset to new Zarr...")
    encoding = {name: {"compressor": COMPRESSOR} for name in ds_rechunked.data_vars}

    t2 = time.time()
    with ProgressBar():
        ds_rechunked.to_zarr(
            NEW_ZARR,
            mode="w",
            encoding=encoding,
            consolidated=True,
        )
    elapsed = time.time() - t2
    print(f"\n  ✓ Written in {elapsed / 60:.1f} minutes ({elapsed / 3600:.2f} hours)")

    # Quick verification
    print("\nVerifying...")
    ds_check = xr.open_zarr(NEW_ZARR, consolidated=True)
    print("  ✓ New Zarr opens")
    print(f"  New chunks example ({list(ds_check.data_vars)[0]}):",
          ds_check[list(ds_check.data_vars)[0]].data.chunks)

    print("\n" + "=" * 70)
    print("✅ RECHUNK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()