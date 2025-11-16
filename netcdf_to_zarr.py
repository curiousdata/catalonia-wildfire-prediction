"""
Convert IberFire NetCDF dataset to Zarr format for faster preprocessing.

This is a ONE-TIME conversion that may take a long time for a 730 GB dataset,
but once done, Zarr will generally allow much faster and more flexible preprocessing.

Usage:
    python netcdf_to_zarr.py
"""

import xarray as xr
import zarr
import sys
from pathlib import Path
import time
from datetime import datetime
from numcodecs import Blosc  # NEW: import Blosc from numcodecs


# ============================================================================
# CONFIGURATION
# ============================================================================

# Input NetCDF file
NETCDF_PATH = Path("data/IberFire.nc")

# Output Zarr directory
ZARR_PATH = Path("data/IberFire.zarr")

# Chunking strategy (balanced chunking for time + space access)
CHUNKS = {
    "time": 64,    # 64 time steps per chunk (good for sequential windows)
    "y": 256,      # spatial tiling
    "x": 256,
}

# Compression settings
COMPRESSOR = Blosc(
    cname="zstd",    # zstd: good compression + fast decompression
    clevel=3,        # level 3: balance speed/compression (1-9)
    shuffle=2        # bit-shuffle: better for floating point
)

# ============================================================================
# MAIN CONVERSION
# ============================================================================

def main():
    print("=" * 70)
    print("NetCDF → Zarr Conversion")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Check input exists
    if not NETCDF_PATH.exists():
        print(f"❌ ERROR: Input file not found: {NETCDF_PATH}")
        print(f"   Current directory: {Path.cwd()}")
        sys.exit(1)
    
    print(f"Input:  {NETCDF_PATH}")
    print(f"Output: {ZARR_PATH}")
    print(f"Chunks: {CHUNKS}")
    print(f"Compressor: {COMPRESSOR}\n")
    
    # Check if output already exists
    if ZARR_PATH.exists():
        response = input(f"⚠️  {ZARR_PATH} already exists. Overwrite? (yes/no): ")
        if response.lower() != "yes":
            print("Cancelled.")
            sys.exit(0)
        print(f"Removing existing {ZARR_PATH}...")
        import shutil
        shutil.rmtree(ZARR_PATH)
    
    # Step 1: Open NetCDF
    print("Step 1/3: Opening NetCDF dataset...")
    start = time.time()
    
    try:
        ds = xr.open_dataset(
            NETCDF_PATH,
            engine="h5netcdf",
            decode_times=True,
        )
    except Exception as e:
        print(f"❌ ERROR opening NetCDF: {e}")
        sys.exit(1)
    
    elapsed = time.time() - start
    print(f"  ✓ Opened in {elapsed:.1f}s")
    print(f"  Variables: {list(ds.data_vars)}")
    print(f"  Dimensions: {dict(ds.dims)}")
    print(f"  Size: {ds.nbytes / 1e9:.1f} GB\n")
    
    # Downcast float64 → float32 where appropriate (leave ints/bools as-is)
    print("Downcasting float64 variables to float32 (if any)...")
    for var in ds.data_vars:
        if ds[var].dtype == "float64":
            ds[var] = ds[var].astype("float32")
    print("  ✓ Downcasting complete\n")
    
    # Step 2: Rechunk dataset
    print(f"Step 2/3: Rechunking with {CHUNKS}...")
    start = time.time()
    
    try:
        ds_rechunked = ds.chunk(CHUNKS)
    except Exception as e:
        print(f"❌ ERROR rechunking: {e}")
        sys.exit(1)
    
    elapsed = time.time() - start
    print(f"  ✓ Rechunked in {elapsed:.1f}s")
    
    # Show chunk info for first variable
    first_var = list(ds.data_vars)[0]
    if hasattr(ds_rechunked[first_var].data, 'chunks'):
        print(f"  Example ({first_var}): {ds_rechunked[first_var].data.chunks}\n")
    
    # Step 3: Write to Zarr
    print(f"Step 3/3: Writing to Zarr (this is the slow part)...")
    print(f"  This may take a long time depending on disk and CPU...")
    print()
    
    start = time.time()
    
    # Prepare encoding for all variables
    encoding = {}
    for var in ds.data_vars:
        encoding[var] = {
            "compressor": COMPRESSOR,
        }
    
    try:
        # Write with progress monitoring
        ds_rechunked.to_zarr(
            ZARR_PATH,
            mode="w",
            encoding=encoding,
            consolidated=True,  # create consolidated metadata for faster opens
        )
    except Exception as e:
        print(f"\n❌ ERROR writing Zarr: {e}")
        sys.exit(1)
    
    elapsed = time.time() - start
    print(f"\n  ✓ Written in {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
    
    # Step 4: Verify output
    print("\nStep 4/4: Verifying output...")
    try:
        ds_zarr = xr.open_zarr(ZARR_PATH)
        print(f"  ✓ Zarr opens successfully")
        print(f"  Variables: {list(ds_zarr.data_vars)}")
        print(f"  Dimensions: {dict(ds_zarr.dims)}")
        
        # Check one variable
        var = list(ds_zarr.data_vars)[0]
        print(f"  Sample check ({var}): shape = {ds_zarr[var].shape}, dtype = {ds_zarr[var].dtype}")
        
    except Exception as e:
        print(f"  ⚠️  Warning: Could not verify: {e}")
    
    # Print final stats
    print("\n" + "=" * 70)
    print("✅ CONVERSION COMPLETE!")
    print("=" * 70)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Size comparison
    netcdf_size = NETCDF_PATH.stat().st_size / 1e9
    
    # Calculate Zarr size (sum of all files in directory)
    zarr_size = sum(f.stat().st_size for f in ZARR_PATH.rglob("*") if f.is_file()) / 1e9
    
    print(f"\nSize comparison:")
    print(f"  NetCDF: {netcdf_size:.1f} GB")
    print(f"  Zarr:   {zarr_size:.1f} GB ({zarr_size/netcdf_size*100:.1f}% of original)")
    
    print(f"\nNext steps:")
    print(f"  1. Update your preprocessing notebook Cell 2:")
    print(f'     ds = xr.open_zarr("{ZARR_PATH}")')
    print(f"  2. Rerun preprocessing (should be 5-10× faster)")
    print(f"  3. Keep {NETCDF_PATH} as backup (can delete Zarr if needed)")
    print("=" * 70)

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Zarr may be incomplete.")
        print(f"   You can safely delete {ZARR_PATH} and restart.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)