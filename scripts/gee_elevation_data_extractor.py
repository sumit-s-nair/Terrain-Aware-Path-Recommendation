import ee
import streamlit as st
import geopy
from geopy.geocoders import Nominatim
from pathlib import Path
import requests

# component to search location and download elevation data from gee
def download_elevation_data(location_query: str, buffer_radius: int, export_folder: Path):
    
    if not location_query:
        return None, None
        
    geolocator = Nominatim(user_agent="terrain_app")
    location = None
    
    try:
        # input validator
        if "," in location_query:
            parts = [x.strip() for x in location_query.split(",")]
            if len(parts) == 2:
                lat, lon = float(parts[0]), float(parts[1])
                location = geopy.location.Location("", (lat, lon), {})
        
        # search by name // not preferred but fallback as we got morons
        if not location:
            location = geolocator.geocode(location_query)
            
        if location:
            lat, lon = location.latitude, location.longitude
            st.success(f"Found location: {lat:.5f}, {lon:.5f}")
            st.map(data={"lat": [lat], "lon": [lon]})
            
            # give filename and check if it exists
            filename = f"dem_{lat:.4f}_{lon:.4f}.tif"
            file_path = export_folder / filename

            if not file_path.exists():
                with st.spinner("Downloading elevation data..."):
                    # create the area of interest with the given radius
                    point = ee.Geometry.Point([lon, lat])
                    roi = point.buffer(buffer_radius).bounds()
                    
                    
                    # Get SRTM elevation data
                    srtm = ee.Image("USGS/SRTMGL1_003").clip(roi)
                    
                    # save it where you want
                    url = srtm.getDownloadURL({
                        'scale': 30,
                        'crs': 'EPSG:4326',
                        'region': roi,
                        'format': 'GEO_TIFF'
                    })
                    
                    response = requests.get(url)
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                        
                st.success(f"File saved: {file_path.name}")
                st.info(f"Full path: {file_path.absolute()}")
            else:
                st.info(f"File already exists: {file_path.name}")
            
            return file_path, (lat, lon)
            
        else:
            st.error("Location not found. Try a different search.")
            return None, None
            
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None
