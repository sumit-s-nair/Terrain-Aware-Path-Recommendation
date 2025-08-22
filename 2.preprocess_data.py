# preprocess_data.py
"""
Preprocess raw terrain data for the physics-based hiking environment.

Outputs to:
  data/processed/
    - slope.tif                (degrees)
    - stability.tif            (0..1)
    - tpi.tif                  (Topographic Position Index)
    - vegetation_cost.tif      (physics movement cost)
    - terrain_difficulty.tif   (combined difficulty)
    - terrain_rgb.tif          (uint8 RGB HxWx3 for the env)
    - trail_coordinates.npy    (optional, pixel coords of GPX)
    - trail_distance_map.npy   (optional, meters to nearest trail pixel)
"""
from pathlib import Path
import numpy as np
import rasterio
import rioxarray
from scipy.ndimage import gaussian_filter, uniform_filter, distance_transform_edt
import gpxpy
from pyproj import Transformer, CRS
from rich.console import Console

console = Console()

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def preprocess():
    console.rule("[bold blue]Preprocessing terrain for physics-based env[/bold blue]")

    dem_path = RAW_DIR / "dem_st_helens.tif"
    landcover_path = RAW_DIR / "landcover_st_helens.tif"
    gpx_path = RAW_DIR / "monitor_ridge_route.gpx"

    if not dem_path.exists():
        console.print(f"[bold red]DEM not found at {dem_path}[/bold red]")
        return

    # ---- Load DEM and metadata ----
    console.log("Loading DEM...")
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        transform = src.transform
        raster_crs = src.crs
        width = src.width
        height = src.height
        cell_size = float(src.res[0])  # meters per pixel (assumes square)
    console.log(f"DEM shape: {dem.shape}, cell size: {cell_size} m, CRS: {raster_crs}")

    # Slight smoothing to remove micro-noise
    console.log("Smoothing DEM to reduce micro-terrain noise...")
    dem_smooth = gaussian_filter(dem, sigma=0.5)

    # ---- Slope calculation (in degrees) ----
    console.log("Calculating slope (degrees)...")
    # gradients w.r.t actual meters: gradient returns dZ/dx (where x is in meters if we pass spacing)
    # np.gradient expects spacing as the second arg: but we provide cell_size to scale.
    dz_dy, dz_dx = np.gradient(dem_smooth, cell_size, cell_size)  # row (y), col (x)
    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = np.degrees(slope_rad).astype(np.float32)

    # Save slope.tif
    slope_profile = profile.copy()
    slope_profile.update(dtype=rasterio.float32, count=1)
    with rasterio.open(PROCESSED_DIR / "slope.tif", "w", **slope_profile) as dst:
        dst.write(slope_deg, 1)
    console.log("Saved slope.tif")

    # ---- Stability map (combining slope and local roughness) ----
    console.log("Calculating stability map...")
    slope_roughness = np.abs(dem - dem_smooth)
    # simple combination: higher slope and higher roughness reduce stability
    stability = np.clip(1.0 - (slope_deg / 45.0) - (slope_roughness / (np.nanmax(slope_roughness) + 1e-9)), 0.0, 1.0)
    stability = stability.astype(np.float32)
    with rasterio.open(PROCESSED_DIR / "stability.tif", "w", **slope_profile) as dst:
        dst.write(stability, 1)
    console.log("Saved stability.tif")

    # ---- Topographic Position Index (TPI) ----
    console.log("Computing Topographic Position Index (TPI)...")
    mean_elev = uniform_filter(dem_smooth, size=9, mode="reflect")
    tpi = (dem_smooth - mean_elev).astype(np.float32)
    with rasterio.open(PROCESSED_DIR / "tpi.tif", "w", **slope_profile) as dst:
        dst.write(tpi, 1)
    console.log("Saved tpi.tif")

    # ---- Vegetation / landcover alignment and physics movement cost ----
    if not landcover_path.exists():
        console.print(f"[yellow]Landcover file not found at {landcover_path}. Using fallback low-cost everywhere.[/yellow]")
        # fallback: low movement cost everywhere
        veg_cost = np.full(dem.shape, 2.0, dtype=np.float32)
    else:
        console.log("Loading and aligning landcover to DEM grid...")
        land_rx = rioxarray.open_rasterio(landcover_path).squeeze()
        dem_rx = rioxarray.open_rasterio(dem_path).squeeze()

        # Reproject/align landcover to DEM grid
        try:
            land_aligned = land_rx.rio.reproject_match(dem_rx)
        except Exception as e:
            console.log(f"[yellow]reproject_match failed: {e}. Attempting manual reprojection...[/yellow]")
            land_aligned = land_rx.rio.reproject(dem_rx.rio.crs)

        # land_aligned is an xarray.DataArray; bring to numpy
        lc_vals = np.array(land_aligned.values, dtype=np.float32)
        # If landcover has extra dims, try to pick the first band
        if lc_vals.ndim == 3:
            # often shape is (bands, H, W) -> take first
            lc_vals = lc_vals[0]

        # define movement cost mapping (NLCD-like codes -> physics costs)
        # You can expand/adjust these as needed.
        veg_cost = np.full(lc_vals.shape, 2.0, dtype=np.float32)
        mapping = {
            11: 1000.0,  # Open water (impassable)
            12: 3.0,     # Perennial ice/snow
            21: 2.0, 22: 5.0, 23: 10.0, 24: 50.0,
            31: 1.5,
            41: 4.0, 42: 6.0, 43: 5.0,
            51: 3.0, 52: 4.5,
            71: 1.8, 72: 2.5, 73: 2.0, 74: 2.2,
            81: 2.5, 82: 3.0,
            90: 15.0, 95: 20.0
        }
        console.log("Mapping landcover classes to movement costs...")
        for code, cost in mapping.items():
            veg_cost[lc_vals == code] = cost

    # Apply grip bonus (vegetation giving traction on slopes)
    console.log("Applying vegetation grip bonus on slopes...")
    grip_bonus = veg_cost.copy()
    mask = (slope_deg > 15.0) & (veg_cost < 20.0) & (veg_cost > 1.5)
    grip_bonus[mask] = np.clip(veg_cost[mask] * 0.7, 1.0, veg_cost[mask])

    grip_bonus = grip_bonus.astype(np.float32)
    veg_profile = profile.copy(); veg_profile.update(dtype=rasterio.float32, count=1)
    with rasterio.open(PROCESSED_DIR / "vegetation_cost.tif", "w", **veg_profile) as dst:
        dst.write(grip_bonus, 1)
    console.log("Saved vegetation_cost.tif")

    # ---- Terrain difficulty composite (for visualization/eval) ----
    console.log("Combining maps into terrain difficulty...")
    # Normalize components
    slope_norm = np.clip(slope_deg / 45.0, 0.0, 1.0)
    veg_norm = np.clip(grip_bonus / 10.0, 0.0, 1.0)
    instability = (1.0 - stability)  # higher when less stable

    difficulty = (slope_norm * 0.4) + (veg_norm * 0.3) + (instability * 0.3)
    difficulty = np.clip(difficulty, 0.0, 1.0).astype(np.float32)

    with rasterio.open(PROCESSED_DIR / "terrain_difficulty.tif", "w", **veg_profile) as dst:
        dst.write(difficulty, 1)
    console.log("Saved terrain_difficulty.tif")

        # ---- Create RGB visualization for agent perception ----
    console.log("Rendering terrain RGB visualization (agent 'vision')...")

    # Normalize elevation for shading
    elev_norm = (dem_smooth - np.nanmin(dem_smooth)) / (
        np.nanmax(dem_smooth) - np.nanmin(dem_smooth) + 1e-9
    )

    rgb = np.zeros((dem.shape[0], dem.shape[1], 3), dtype=np.uint8)

    # Masks for landcover + slope + elevation
    mask_water = grip_bonus >= 100
    mask_dense = (grip_bonus >= 20) & ~mask_water
    mask_forest = (grip_bonus >= 10) & ~mask_water & ~mask_dense
    mask_midveg = (grip_bonus >= 6) & ~mask_water & ~mask_dense & ~mask_forest
    mask_lightveg = (grip_bonus >= 4) & ~mask_water & ~mask_dense & ~mask_forest & ~mask_midveg
    mask_rock = (slope_deg > 30) & ~(
        mask_water | mask_dense | mask_forest | mask_midveg | mask_lightveg
    )
    mask_snow = (elev_norm > 0.8) & ~(
        mask_water | mask_dense | mask_forest | mask_midveg | mask_lightveg | mask_rock
    )
    mask_grass = (grip_bonus <= 2) & ~(
        mask_water | mask_dense | mask_forest | mask_midveg | mask_lightveg | mask_rock | mask_snow
    )
    mask_sand = ~(mask_water | mask_dense | mask_forest | mask_midveg |
                  mask_lightveg | mask_rock | mask_snow | mask_grass)

    # Base colors
    rgb[mask_water] = [30, 144, 255]
    rgb[mask_dense] = [70, 130, 180]
    rgb[mask_forest] = [34, 139, 34]
    rgb[mask_midveg] = [50, 205, 50]
    rgb[mask_lightveg] = [154, 205, 50]
    rgb[mask_rock] = [105, 105, 105]
    rgb[mask_snow] = [248, 248, 255]
    rgb[mask_grass] = [144, 238, 144]
    rgb[mask_sand] = [210, 180, 140]

    # Brightness adjustment based on slope
    brightness = 1.0 - (slope_deg / 90.0) * 0.3
    brightness = np.clip(brightness, 0.7, 1.0)[..., None]  # shape (H, W, 1)

    rgb = np.clip(rgb * brightness, 0, 255).astype(np.uint8)

    # Save RGB GeoTIFF
    rgb_profile = profile.copy()
    rgb_profile.update(dtype=rasterio.uint8, count=3)
    with rasterio.open(PROCESSED_DIR / "terrain_rgb.tif", "w", **rgb_profile) as dst:
        dst.write(rgb.transpose(2, 0, 1))
    console.log("Saved terrain_rgb.tif")


    # ---- Trail processing (optional) ----
    if gpx_path.exists():
        console.log("Parsing GPX trail and mapping points to raster indices...")
        try:
            with open(gpx_path, "r") as f:
                gpx = gpxpy.parse(f)

            transformer = Transformer.from_crs(CRS("EPSG:4326"), raster_crs, always_xy=True)
            trail_points = []
            for trk in gpx.tracks:
                for seg in trk.segments:
                    for pt in seg.points:
                        try:
                            x, y = transformer.transform(pt.longitude, pt.latitude)  # map coords (x,y)
                            # convert map coords to row,col
                            row, col = rasterio.transform.rowcol(transform, x, y)
                            if 0 <= row < height and 0 <= col < width:
                                trail_points.append([row, col])
                        except Exception:
                            continue

            if len(trail_points) > 0:
                trail_array = np.array(trail_points, dtype=np.int32)
                np.save(PROCESSED_DIR / "trail_coordinates.npy", trail_array)
                console.log(f"Saved trail_coordinates.npy with {len(trail_array)} points")

                # create trail distance map in meters
                mask = np.zeros((height, width), dtype=bool)
                mask[trail_array[:, 0], trail_array[:, 1]] = True
                dist_pix = distance_transform_edt(~mask)
                dist_m = dist_pix * cell_size
                np.save(PROCESSED_DIR / "trail_distance_map.npy", dist_m.astype(np.float32))
                console.log("Saved trail_distance_map.npy")
            else:
                console.print("[yellow]GPX parsed but produced no in-bounds points.[/yellow]")

        except Exception as e:
            console.print(f"[red]Error parsing GPX: {e}[/red]")
    else:
        console.print("[yellow]No GPX found; skipping trail processing.[/yellow]")

    console.print("\n[bold green]Preprocessing complete. Files written to data/processed/[/bold green]")
    console.print("• slope.tif, stability.tif, tpi.tif, vegetation_cost.tif, terrain_difficulty.tif, terrain_rgb.tif")
    console.print("• trail_coordinates.npy (if GPX provided), trail_distance_map.npy (if GPX provided)")


if __name__ == "__main__":
    preprocess()
