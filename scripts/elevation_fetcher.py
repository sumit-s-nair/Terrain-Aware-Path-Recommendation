import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import streamlit as st
from scipy.ndimage import zoom

def show_elevation_map(tif_path: Path, processed_folder: Path):
    """Component to load and display elevation data with summary stats"""
    try:
        with rasterio.open(tif_path) as src:
            elevation = src.read(1)
            transform = src.transform
            nodata = src.nodata
            
        # data cleaning stuff
        elevation = np.where(elevation == nodata, np.nan, elevation)
        elevation = np.where(elevation == 0, np.nan, elevation)
        
        # upscale the image using scipy zoom, we are doing a 3x zoom
        zoom_factor = 3
        elevation_upscaled = zoom(elevation, zoom_factor, order=1, mode='nearest', prefilter=False)
        
        # save both original and upscaled numpy arrays
        npy_path = processed_folder / tif_path.with_suffix(".npy").name
        base_name = tif_path.stem
        npy_upscaled_path = processed_folder / f"{base_name}_upscaled.npy"
        np.save(npy_path, elevation)
        np.save(npy_upscaled_path, elevation_upscaled)
        
        # show summary for time pass
        st.subheader("Elevation Data Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Original Shape", f"{elevation.shape[0]} x {elevation.shape[1]}")
        with col2:
            st.metric("Upscaled Shape", f"{elevation_upscaled.shape[0]} x {elevation_upscaled.shape[1]}")
        with col3:
            st.metric("Min Elevation", f"{np.nanmin(elevation):.0f} m")
        with col4:
            st.metric("Max Elevation", f"{np.nanmax(elevation):.0f} m")
        
        # create grid to map real-world cords for upscaled version
        rows, cols = elevation_upscaled.shape
        left, top = transform * (0, 0)
        right, bottom = transform * (elevation.shape[1], elevation.shape[0])  # use original size for bounds
        
        lon = np.linspace(left, right, cols)
        lat = np.linspace(top, bottom, rows)
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        
        # 2D map
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(elevation_upscaled, cmap="terrain", extent=[left, right, bottom, top], origin='upper')
        plt.colorbar(im, ax=ax, label="Elevation (meters)")
        ax.set_title("2D Elevation Map (Upscaled)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        st.pyplot(fig)
        
        # 3D also cause it looks nice
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(lon_grid, lat_grid, elevation_upscaled, cmap='terrain', linewidth=0, antialiased=False, alpha=0.95)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_zlabel('Elevation (meters)')
        ax.set_title('3D Elevation Surface with Map Coordinates')
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Elevation (meters)')
        st.pyplot(fig)
        
        st.success(f"Original data saved: {npy_path.name}")
        st.success(f"Upscaled data saved: {npy_upscaled_path.name}")
        
        return elevation_upscaled, npy_upscaled_path
        
    except Exception as e:
        st.error(f"Error processing elevation data: {e}")
        return None, None
