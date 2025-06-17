# Terrain-Aware Path Recommendation

A beautiful app that fetches elevation data from Google Earth Engine for now

## Features

- I'll write them later

## Setup

### Prerequisites

1. **Google Earth Engine Account**
   - Sign up at [https://earthengine.google.com/](https://earthengine.google.com/)
   - Create a Google Cloud Project with Earth Engine API enabled
   - Note your project ID (you'll need this)

2. **Authentication**
   ```bash
   # First time setup - authenticate with Google Earth Engine
   earthengine authenticate
   ```

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Terrain-Aware-Path-Recommendation
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Update the project ID in the code:
   - Open `scripts/main.py`
   - Change `EE_PROJECT = "rapid-462805"` to your project ID

## Usage

### Using the Streamlit App

1. Run the application:
   ```bash
   streamlit run scripts/main.py
   ```

2. Enter a location:
   - **Coordinates**: `12.9716, 77.5946` (latitude, longitude)

3. Adjust the area radius using the sidebar slider (250m - 2000m)

### File Structure

```
├── scripts/
│   ├── main.py                        # Main Streamlit application
│   ├── elevation_fetcher.py           # Elevation visualization component
│   └── gee_elevation_data_extractor.py # Google Earth Engine data fetcher
├── data/
│   ├── raw/                          # Downloaded .tif files
│   └── processed/                    # Processed .npy files
├── main.py                           # Legacy visualization script
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Tips

- Use coordinates for best accuracy
- Larger radius = more detailed terrain context
- Files are cached to avoid re-downloading
- Upscaling improves visual quality but doesn't add real detail

## Troubleshooting

- **Authentication Error**: Run `earthengine authenticate` and update project ID
- **Location Not Found**: Try coordinates instead of place names
- **Download Fails**: Check internet connection and Google Earth Engine quotas

## License

dont have one rn
