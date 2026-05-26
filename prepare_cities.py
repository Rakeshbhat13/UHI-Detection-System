"""
prepare_cities.py
─────────────────
Run this ONCE after downloading India_9Cities_UHI_ML_Dataset.csv from Google Drive.

What it does:
  1. Reads the combined 9-city CSV from GEE
  2. Renames columns to match pipeline.py expectations
  3. Splits into individual city CSV files  → data/cityname_features.csv
  4. Runs the full ML pipeline for each city → models + geojson + stats

Usage:
  python prepare_cities.py

Make sure India_9Cities_UHI_ML_Dataset.csv is in the same folder as this script.
"""

import pandas as pd
import os
import sys

# ─────────────────────────────────────────
# STEP 1: COLUMN NAME MAPPING
# GEE exports columns with these names,
# pipeline.py expects: lst, ndvi, ndbi, fvc, latitude, longitude
# ─────────────────────────────────────────
COLUMN_MAP = {
    'LST':       'lst',
    'NDVI':      'ndvi',
    'NDBI':      'ndbi',
    '.geo':      'geo',       # GEE geometry column
    'City':      'city',
    'system:index': 'index',
    'UHI_Zone':  'uhi_zone',
}

# City name mapping (GEE name → cities.py key)
CITY_KEY_MAP = {
    'Bengaluru':  'bengaluru',
    'Hyderabad':  'hyderabad',
    'Mumbai':     'mumbai',
    'Delhi':      'delhi',
    'Chennai':    'chennai',
    'Mangaluru':  'mangaluru',
    'Ahmedabad':  'ahmedabad',
    'Kolkata':    'kolkata',
    'Pune':       'pune',
}


# ─────────────────────────────────────────
# STEP 2: LOAD THE COMBINED CSV
# ─────────────────────────────────────────
CSV_FILE = 'India_9Cities_UHI_ML_Dataset.csv'

if not os.path.exists(CSV_FILE):
    print(f"ERROR: '{CSV_FILE}' not found!")
    print("Please download it from Google Drive → UHI_Project folder")
    print("and place it in the same folder as this script.")
    sys.exit(1)

print(f"Loading {CSV_FILE}...")
df = pd.read_csv(CSV_FILE)

print(f"Total rows: {len(df)}")
print(f"Columns found: {list(df.columns)}")
print(f"Cities found: {df['City'].unique() if 'City' in df.columns else 'City column not found'}")


# ─────────────────────────────────────────
# STEP 3: EXTRACT LAT/LON FROM .geo COLUMN
# GEE exports geometry as JSON string like:
# {"type":"Point","coordinates":[77.59,12.97]}
# ─────────────────────────────────────────
import json

def extract_coords(geo_str):
    """Extract latitude and longitude from GEE .geo JSON string."""
    try:
        geo = json.loads(geo_str)
        lon, lat = geo['coordinates']
        return pd.Series({'latitude': lat, 'longitude': lon})
    except:
        return pd.Series({'latitude': None, 'longitude': None})

if '.geo' in df.columns:
    print("\nExtracting lat/lon from .geo column...")
    coords = df['.geo'].apply(extract_coords)
    df['latitude']  = coords['latitude']
    df['longitude'] = coords['longitude']
else:
    # If GEE already exported latitude/longitude as separate columns
    if 'latitude' not in df.columns:
        df['latitude']  = None
    if 'longitude' not in df.columns:
        df['longitude'] = None


# ─────────────────────────────────────────
# STEP 4: RENAME COLUMNS
# ─────────────────────────────────────────
df.rename(columns=COLUMN_MAP, inplace=True)

# Add FVC (Fractional Vegetation Cover) from NDVI
# FVC = ((NDVI - NDVImin) / (NDVImax - NDVImin))^2
# This is needed by pipeline.py
ndvi_min = -0.2
ndvi_max =  0.8
df['fvc'] = ((df['ndvi'] - ndvi_min) / (ndvi_max - ndvi_min)) ** 2
df['fvc'] = df['fvc'].clip(0, 1)  # Keep between 0 and 1

print(f"\nFinal columns: {list(df.columns)}")


# ─────────────────────────────────────────
# STEP 5: SPLIT BY CITY AND SAVE
# ─────────────────────────────────────────
os.makedirs('data', exist_ok=True)

city_col = 'city' if 'city' in df.columns else 'City'
saved_cities = []

print("\nSplitting by city...")
for gee_name, city_key in CITY_KEY_MAP.items():
    # Match city (case-insensitive)
    city_df = df[df[city_col].str.lower() == gee_name.lower()].copy()

    if len(city_df) == 0:
        print(f"  ⚠ No data found for {gee_name} — skipping")
        continue

    # Keep only needed columns
    keep_cols = ['lst', 'ndvi', 'ndbi', 'fvc', 'latitude', 'longitude']
    available = [c for c in keep_cols if c in city_df.columns]
    city_df   = city_df[available].dropna()

    # Save
    out_path = f'data/{city_key}_features.csv'
    city_df.to_csv(out_path, index=False)
    saved_cities.append(city_key)
    print(f"  ✅ {gee_name}: {len(city_df)} samples → {out_path}")


# ─────────────────────────────────────────
# STEP 6: RUN ML PIPELINE FOR ALL CITIES
# ─────────────────────────────────────────
print(f"\n{'='*60}")
print("Running ML Pipeline for all cities...")
print(f"{'='*60}")

# Import pipeline
try:
    from pipeline import run_pipeline
except ImportError:
    print("ERROR: pipeline.py not found in the same folder!")
    sys.exit(1)

success = []
failed  = []

for city_key in saved_cities:
    try:
        run_pipeline(city_key)
        success.append(city_key)
    except Exception as e:
        print(f"\n❌ ERROR for {city_key}: {e}")
        failed.append(city_key)

# ─────────────────────────────────────────
# STEP 7: SUMMARY
# ─────────────────────────────────────────
print(f"\n{'='*60}")
print("DONE! Summary:")
print(f"{'='*60}")
print(f"✅ Successful: {', '.join(success) if success else 'None'}")
print(f"❌ Failed:     {', '.join(failed)  if failed  else 'None'}")
print(f"\nYou can now start the Flask API:")
print(f"  python app.py")
print(f"\nAnd the React frontend:")
print(f"  npm run dev")
