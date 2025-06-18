import ee
import streamlit as st
from pathlib import Path

# component imports
from elevation_fetcher import show_elevation_map
from gee_elevation_data_extractor import download_elevation_data

# init gee
EE_PROJECT = "rapid-462805"
try:
    ee.Initialize(project=EE_PROJECT)
except Exception as e:
    st.error("Ohh Earth Engine failed to initialize its your fault definitely nothing wrong with the code, cause maybe you dont have the valid perms but heres the actual error anyway figure it out atb:")
    try:
        ee.Authenticate()
        ee.Initialize(project=EE_PROJECT)
        st.success("Authentication successful! Please refresh the page.")
    except Exception as auth_error:
        st.error(f"Authentication failed. Please run 'earthengine authenticate' in your terminal first. Error: {auth_error}")
        st.stop()

# make folders if they don't exist
EXPORT_FOLDER = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_FOLDER = Path(__file__).parent.parent / "data" / "processed"
EXPORT_FOLDER.mkdir(parents=True, exist_ok=True)
PROCESSED_FOLDER.mkdir(parents=True, exist_ok=True)

# streamlit app config
st.set_page_config(layout="wide")
st.title("Get Elevation Data & Visualise it cause i guess you need it")
st.markdown("Search for any location to download and visualize elevation data.")

# sidebar not needed but why not
with st.sidebar:
    st.header("Settings")
    buffer_radius = st.slider("Area radius (meters)", 250, 2000, 500, 50)

# searchbar for the loc
st.subheader("Search for a location")
location_query = st.text_input(
    "Enter coordinates (lat, lon):",
    "",
    help="You can paste coordinates from Google Maps here"
)

# get file path and cords
file_path, coordinates = download_elevation_data(location_query, buffer_radius, EXPORT_FOLDER)

# confirm existance, use the elevation display component
if file_path and file_path.exists():
    elevation_data, processed_path = show_elevation_map(file_path, PROCESSED_FOLDER)
