import os
import ee
import streamlit as st
from pathlib import Path
from geopy.geocoders import Nominatim
from geopy.location import Location
import requests

# Import our components
from elevation_fetcher import show_elevation_map
from gee_elevation_data_extractor import download_elevation_data

# Initialize Earth Engine with project
EE_PROJECT = "rapid-462805"
try:
    ee.Initialize(project=EE_PROJECT)
except Exception as e:
    st.error(f"Earth Engine failed to initialize: {e}")
    st.stop()

# Create folders for downloaded files
EXPORT_FOLDER = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_FOLDER = Path(__file__).parent.parent / "data" / "processed"
EXPORT_FOLDER.mkdir(parents=True, exist_ok=True)
PROCESSED_FOLDER.mkdir(parents=True, exist_ok=True)

# Streamlit app setup
st.set_page_config(layout="wide")
st.title("Elevation Data Fetcher & Visualizer")
st.markdown("Search for any location to download and visualize elevation data.")

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    buffer_radius = st.slider("Area radius (meters)", 250, 2000, 500, 50)

# Search interface
st.subheader("Search for a location")
location_query = st.text_input(
    "Enter place name or coordinates (lat, lon):",
    "",
    help="You can paste coordinates from Google Maps here"
)

# Use the download component
file_path, coordinates = download_elevation_data(location_query, buffer_radius, EXPORT_FOLDER)

# If we got a file, use the elevation display component
if file_path and file_path.exists():
    elevation_data, processed_path = show_elevation_map(file_path, PROCESSED_FOLDER)
