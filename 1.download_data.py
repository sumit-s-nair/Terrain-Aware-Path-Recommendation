# download_data.py
"""
Download DEM and trail data for realistic hiking RL environment.

- Reads GPX file (trail).
- Computes bounding box around entire trail.
- Expands bounding box with margin to ensure full coverage.
- Downloads high-resolution DEM from OpenTopography using API key.
- Saves DEM GeoTIFF in data/raw/.
"""

import os
import math
import shutil
from pathlib import Path
import requests
import gpxpy
import rasterio
from rasterio.transform import from_origin
from rasterio.merge import merge
import numpy as np
import tempfile

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ---- Config ----
GPX_FILE = RAW_DIR / "monitor_ridge_route.gpx"
DEM_FILE = RAW_DIR / "dem_st_helens.tif"
LANDCOVER_FILE = RAW_DIR / "landcover_st_helens.tif"

# DEM source: USGS National Map - different endpoint that's more reliable
DEM_API = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"
# Landcover source: USGS National Land Cover Database (NLCD)
LANDCOVER_API = "https://www.mrlc.gov/geoserver/mrlc_display/NLCD_2021_Land_Cover_L48/wms"

# Alternative: Try the WMS endpoint if ImageServer fails
WMS_API = "https://elevation.nationalmap.gov/arcgis/services/3DEPElevation/ImageServer/WMSServer"

# Resolution (meters) - try for 1m, fallback to 10m/30m
DEM_RES = 1  # target resolution

# Margin around trail bounding box (degrees)
MARGIN_DEG = 0.02  # ~2 km margin depending on latitude


def get_trail_bounds(gpx_path: Path):
    """Read GPX file and return lat/lon bounding box (with margin)."""
    with open(gpx_path, "r") as f:
        gpx = gpxpy.parse(f)

    lats = [pt.latitude for trk in gpx.tracks for seg in trk.segments for pt in seg.points]
    lons = [pt.longitude for trk in gpx.tracks for seg in trk.segments for pt in seg.points]

    lat_min, lat_max = min(lats), max(lats)
    lon_min, lon_max = min(lons), max(lons)

    # Add margin
    lat_min -= MARGIN_DEG
    lat_max += MARGIN_DEG
    lon_min -= MARGIN_DEG
    lon_max += MARGIN_DEG

    return lat_min, lat_max, lon_min, lon_max


def calculate_area_km2(lat_min, lat_max, lon_min, lon_max):
    """Calculate approximate area in km2 for the bounding box."""
    # Rough approximation: 1 degree ≈ 111 km at equator
    # Adjust longitude for latitude
    lat_center = (lat_min + lat_max) / 2
    lon_km_per_deg = 111.32 * abs(math.cos(math.radians(lat_center)))
    lat_km_per_deg = 110.54
    
    width_km = (lon_max - lon_min) * lon_km_per_deg
    height_km = (lat_max - lat_min) * lat_km_per_deg
    
    return width_km * height_km


def calculate_tiles(lat_min, lat_max, lon_min, lon_max, max_area_km2=200):
    """
    Split a large bounding box into smaller tiles that fit within area limits.
    Returns list of (lat_min, lat_max, lon_min, lon_max) tuples.
    """
    total_area = calculate_area_km2(lat_min, lat_max, lon_min, lon_max)
    
    if total_area <= max_area_km2:
        return [(lat_min, lat_max, lon_min, lon_max)]
    
    # Calculate how many tiles we need
    tiles_needed = math.ceil(total_area / max_area_km2)
    
    # Try to make roughly square tiles
    tiles_per_side = math.ceil(math.sqrt(tiles_needed))
    
    lat_step = (lat_max - lat_min) / tiles_per_side
    lon_step = (lon_max - lon_min) / tiles_per_side
    
    tiles = []
    for i in range(tiles_per_side):
        for j in range(tiles_per_side):
            tile_lat_min = lat_min + i * lat_step
            tile_lat_max = min(lat_max, lat_min + (i + 1) * lat_step)
            tile_lon_min = lon_min + j * lon_step
            tile_lon_max = min(lon_max, lon_min + (j + 1) * lon_step)
            
            if tile_lat_min < tile_lat_max and tile_lon_min < tile_lon_max:
                tiles.append((tile_lat_min, tile_lat_max, tile_lon_min, tile_lon_max))
    
    return tiles


def download_single_tile(lat_min, lat_max, lon_min, lon_max, temp_dir, tile_idx):
    """Download a single tile from USGS 3DEP and return the filepath."""
    area_km2 = calculate_area_km2(lat_min, lat_max, lon_min, lon_max)
    print(f"  Tile {tile_idx}: {area_km2:.2f} km²")
    
    # USGS 3DEP ImageServer parameters
    params = {
        'bbox': f"{lon_min},{lat_min},{lon_max},{lat_max}",
        'bboxSR': '4326',  # WGS84
        'imageSR': '4326',  # WGS84
        'format': 'tiff',
        'pixelType': 'F32',  # 32-bit float
        'noDataInterpretation': 'esriNoDataMatchAny',
        'interpolation': 'RSP_BilinearInterpolation',
        'f': 'image'
    }
    
    # Try to get the highest resolution available
    # USGS 3DEP has 1m, 1/3 arc-second (~10m), and 1 arc-second (~30m)
    
    # Calculate appropriate image size for ~1m resolution
    # 1m ≈ 0.00001 degrees at this latitude
    width_deg = lon_max - lon_min
    height_deg = lat_max - lat_min
    
    # Target 1m resolution
    target_res = 0.00001  # degrees per meter approximately
    width_pixels = int(width_deg / target_res)
    height_pixels = int(height_deg / target_res)
    
    # Only reduce resolution if absolutely necessary to stay within USGS limits
    # Prefer splitting into more tiles rather than reducing resolution
    max_pixels = 2048
    if width_pixels > max_pixels or height_pixels > max_pixels:
        # If we hit limits, the tile is too large - this should be handled by smaller tiling
        scale_factor = max(width_pixels / max_pixels, height_pixels / max_pixels)
        if scale_factor > 2:
            # If we need to scale down by more than 2x, the tile is way too big
            raise RuntimeError(f"Tile too large for 1m resolution. Need smaller tiles. Scale factor: {scale_factor}")
        
        width_pixels = int(width_pixels / scale_factor)
        height_pixels = int(height_pixels / scale_factor)
        actual_res_m = target_res * scale_factor * 111000
        print(f"    Limited to {width_pixels}x{height_pixels} pixels (~{actual_res_m:.1f}m resolution)")
    else:
        print(f"    Requesting {width_pixels}x{height_pixels} pixels (~1m resolution)")
    
    params['size'] = f"{width_pixels},{height_pixels}"
    
    resp = requests.get(DEM_API, params=params, stream=True)
    
    if resp.status_code == 200:
        # Check if we got actual image data
        content_type = resp.headers.get('content-type', '')
        if 'image' in content_type or 'tiff' in content_type:
            tile_path = temp_dir / f"tile_{tile_idx}_usgs.tif"
            with open(tile_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"    Success with USGS 3DEP")
            return tile_path
        else:
            error_text = resp.text
            print(f"    USGS returned error: {error_text}")
            raise RuntimeError(f"USGS returned error: {error_text}")
    else:
        print(f"    Failed: {resp.status_code} - {resp.text[:200]}...")
        raise RuntimeError(f"USGS request failed for tile {tile_idx}: {resp.status_code}")


def merge_tiles(tile_paths, output_path):
    """Merge multiple DEM tiles into a single file."""
    print(f"Merging {len(tile_paths)} tiles...")
    
    # Open all tiles
    datasets = []
    for tile_path in tile_paths:
        if tile_path.exists():
            datasets.append(rasterio.open(tile_path))
    
    if not datasets:
        raise RuntimeError("No valid tiles to merge")
    
    # Merge tiles
    mosaic, out_transform = merge(datasets)
    
    # Get metadata from first dataset
    out_meta = datasets[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_transform,
        "compress": "lzw"
    })
    
    # Write merged result
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(mosaic)
    
    # Close all datasets
    for dataset in datasets:
        dataset.close()
    
    print(f"Merged DEM saved to {output_path}")
def download_landcover(lat_min, lat_max, lon_min, lon_max, out_path: Path):
    """
    Download NLCD (National Land Cover Database) data using WMS.
    This provides landcover classification for vegetation analysis.
    """
    print(f"Requesting landcover for bbox: {lat_min},{lon_min} → {lat_max},{lon_max}")
    
    # Calculate image size to match DEM resolution approximately
    width_deg = lon_max - lon_min
    height_deg = lat_max - lat_min
    
    # Use similar resolution to DEM but landcover doesn't need to be as fine
    # NLCD is 30m native resolution anyway
    target_res = 0.0003  # ~30m resolution
    width_pixels = int(width_deg / target_res)
    height_pixels = int(height_deg / target_res)
    
    # Cap at reasonable size
    max_pixels = 2048
    if width_pixels > max_pixels or height_pixels > max_pixels:
        scale_factor = max(width_pixels / max_pixels, height_pixels / max_pixels)
        width_pixels = int(width_pixels / scale_factor)
        height_pixels = int(height_pixels / scale_factor)
    
    print(f"Requesting {width_pixels}x{height_pixels} pixels for landcover")
    
    # WMS parameters for NLCD
    params = {
        'service': 'WMS',
        'version': '1.1.1',
        'request': 'GetMap',
        'layers': 'NLCD_2021_Land_Cover_L48',
        'styles': '',
        'bbox': f"{lon_min},{lat_min},{lon_max},{lat_max}",
        'width': width_pixels,
        'height': height_pixels,
        'srs': 'EPSG:4326',
        'format': 'image/geotiff'
    }
    
    resp = requests.get(LANDCOVER_API, params=params, stream=True)
    
    if resp.status_code == 200:
        content_type = resp.headers.get('content-type', '')
        if 'image' in content_type or 'tiff' in content_type:
            with open(out_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Saved landcover to {out_path}")
            return
        else:
            error_text = resp.text
            print(f"Landcover service returned error: {error_text}")
            print("Landcover download failed - continuing without landcover data")
            return
    else:
        print(f"Landcover request failed: {resp.status_code}")
        print("Continuing without landcover data")


def download_dem(lat_min, lat_max, lon_min, lon_max, out_path: Path, resolution=DEM_RES):

    """
    Download DEM using OpenTopography API with tiling for large areas.
    Automatically splits large requests into smaller tiles and merges them.
    """
    print(f"Requesting DEM for bbox: {lat_min},{lon_min} → {lat_max},{lon_max}")
    
    # Calculate total area
    total_area = calculate_area_km2(lat_min, lat_max, lon_min, lon_max)
    print(f"Total area: {total_area:.2f} km²")
    
    # Determine maximum tile size for USGS requests
    # Use very small tiles to stay within USGS pixel limits while maintaining 1m resolution
    max_tile_area = 2  # Very small tiles to ensure true 1m resolution
    
    # Get tiles - force tiling for better resolution
    tiles = calculate_tiles(lat_min, lat_max, lon_min, lon_max, max_tile_area)
    print(f"Split into {len(tiles)} tiles for high-resolution coverage")
    
    # Always use tiling approach for best resolution, even for single tiles
    # This ensures we stay within USGS limits while maintaining 1m accuracy
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        tile_paths = []
        
        for i, (tile_lat_min, tile_lat_max, tile_lon_min, tile_lon_max) in enumerate(tiles):
            try:
                tile_path = download_single_tile(tile_lat_min, tile_lat_max, tile_lon_min, tile_lon_max, temp_path, i)
                tile_paths.append(tile_path)
            except Exception as e:
                print(f"Warning: Failed to download tile {i}: {e}")
        
        if not tile_paths:
            raise RuntimeError("Failed to download any tiles")
        
        if len(tile_paths) == 1:
            # Single tile - just copy it
            import shutil
            shutil.copy2(tile_paths[0], out_path)
            print(f"Saved single high-resolution tile to {out_path}")
        else:
            # Multiple tiles - merge them
            merge_tiles(tile_paths, out_path)


def main():
    if not GPX_FILE.exists():
        raise FileNotFoundError(f"Missing GPX file at {GPX_FILE}")

    lat_min, lat_max, lon_min, lon_max = get_trail_bounds(GPX_FILE)

    # Download DEM
    if DEM_FILE.exists():
        print(f"DEM already exists at {DEM_FILE}, skipping download.")
    else:
        download_dem(lat_min, lat_max, lon_min, lon_max, DEM_FILE, resolution=DEM_RES)
    
    # Download landcover data
    if LANDCOVER_FILE.exists():
        print(f"Landcover already exists at {LANDCOVER_FILE}, skipping download.")
    else:
        download_landcover(lat_min, lat_max, lon_min, lon_max, LANDCOVER_FILE)


if __name__ == "__main__":
    main()
