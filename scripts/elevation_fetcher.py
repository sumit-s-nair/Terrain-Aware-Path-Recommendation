import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import streamlit as st

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
        
        # save numpy array for later
        npy_path = processed_folder / tif_path.with_suffix(".npy").name
        np.save(npy_path, elevation)
        
        # show summary for time pass
        st.subheader("Elevation Data Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Data Shape", f"{elevation.shape[0]} x {elevation.shape[1]}")
        with col2:
            st.metric("Min Elevation", f"{np.nanmin(elevation):.0f} m")
        with col3:
            st.metric("Max Elevation", f"{np.nanmax(elevation):.0f} m")
        
        # create grid to map real-world cords
        rows, cols = elevation.shape
        left, top = transform * (0, 0)
        right, bottom = transform * (cols, rows)
        
        lon = np.linspace(left, right, cols)
        lat = np.linspace(top, bottom, rows)
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        
        # 2D map
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(elevation, cmap="terrain", extent=[left, right, bottom, top], origin='upper')
        plt.colorbar(im, ax=ax, label="Elevation (meters)")
        ax.set_title("2D Elevation Map")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        st.pyplot(fig)
        
        # 3D also cause it looks nice
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(lon_grid, lat_grid, elevation, cmap='terrain', linewidth=0, antialiased=False, alpha=0.95)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_zlabel('Elevation (meters)')
        ax.set_title('3D Elevation Surface with Map Coordinates')
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Elevation (meters)')
        st.pyplot(fig)
        
        st.success(f"Processed data saved: {npy_path.name}")
        
        return elevation, npy_path
        
    except Exception as e:
        st.error(f"Error processing elevation data: {e}")
        return None, None
