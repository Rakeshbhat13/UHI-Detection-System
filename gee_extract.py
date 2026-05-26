"""
Phase 1: Google Earth Engine Data Extraction
Urban Heat Island Detection - Multi-City Support
Course: Advanced Machine Learning (AM2001-1)
"""

import ee
import geemap
import numpy as np
import pandas as pd
import os
import sys

from cities import CITIES, DEFAULT_CITY, get_city

# ─── Step 1: Authenticate & Initialize ───────────────────────────────────────
# Run this ONCE in terminal: earthengine authenticate
ee.Initialize(project='your-gee-project-id')  # Replace with your project ID


# ─── Step 2: Load & Filter Landsat 8 Collection ──────────────────────────────
def get_landsat_collection(roi, start, end, city_name="City"):
    """
    Load Landsat 8 Collection 2 Level-2 imagery.
    Cloud cover filtered to < 20%.
    """
    collection = (
        ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
        .filterBounds(roi)
        .filterDate(start, end)
        .filter(ee.Filter.lt('CLOUD_COVER', 20))
        .sort('CLOUD_COVER')       # best image first
    )
    count = collection.size().getInfo()
    print(f"Found {count} Landsat 8 scenes for {city_name}")
    return collection


# ─── Step 3: Calculate Land Surface Temperature (LST) ────────────────────────
def calculate_lst(image):
    """
    LST from Landsat 8 Band 10 (Thermal Infrared).
    Formula:
      1. Apply scale factors (Landsat C2 requirement)
      2. Convert to Brightness Temperature (BT) in Kelvin
      3. Emissivity correction using NDVI-based method
      4. Convert to Celsius
    """
    # Apply Landsat Collection 2 scale factors
    optical = image.select('SR_B.').multiply(0.0000275).add(-0.2)
    thermal = image.select('ST_B10').multiply(0.00341802).add(149.0)
    image   = image.addBands(optical, overwrite=True).addBands(thermal, overwrite=True)

    # NDVI (needed for emissivity)
    ndvi = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')

    # Fractional Vegetation Cover (FVC)
    ndvi_min = ee.Number(0.2)
    ndvi_max = ee.Number(0.86)
    fvc = (ndvi.subtract(ndvi_min)
               .divide(ndvi_max.subtract(ndvi_min))
               .pow(2)
               .rename('FVC'))

    # Land Surface Emissivity (LSE)
    lse = fvc.multiply(0.004).add(0.986).rename('LSE')

    # Brightness Temperature (K)
    bt = image.select('ST_B10').rename('BT')

    # LST in Celsius using Planck's equation correction
    # LST = BT / (1 + (λ * BT / ρ) * ln(ε))
    # λ = 10.895 μm (Band 10 wavelength), ρ = 14388 μm·K
    lambda_val = ee.Number(10.895)
    rho        = ee.Number(14388)

    lst = bt.divide(
        ee.Image(1).add(
            lambda_val.multiply(bt).divide(rho).multiply(lse.log())
        )
    ).subtract(273.15).rename('LST_Celsius')

    return image.addBands([ndvi, fvc, lse, lst])


# ─── Step 4: Calculate NDBI (Built-up Index) ─────────────────────────────────
def calculate_ndbi(image):
    """
    NDBI = (SWIR - NIR) / (SWIR + NIR)
    Higher NDBI = more concrete/buildings = more heat absorption
    Landsat 8: SWIR = Band 6, NIR = Band 5
    """
    ndbi = image.normalizedDifference(['SR_B6', 'SR_B5']).rename('NDBI')
    return image.addBands(ndbi)


# ─── Step 5: Process & Export for a Single City ──────────────────────────────
def process_and_export(city_key):
    """
    Full pipeline for one city: load → compute LST/NDVI/NDBI → sample → CSV
    """
    city = get_city(city_key)
    city_name  = city['name']
    bbox       = city['bbox']
    start_date = city['start_date']
    end_date   = city['end_date']

    roi = ee.Geometry.Rectangle(bbox)
    output_path = f'data/{city_key}_features.csv'

    print(f"\n{'='*60}")
    print(f"Processing: {city_name}")
    print(f"Bbox: {bbox}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"{'='*60}")

    print("Loading Landsat 8 collection...")
    collection = get_landsat_collection(roi, start_date, end_date, city_name)

    # Use median composite (reduces cloud artifacts)
    image = collection.median()

    print("Calculating LST, NDVI, NDBI...")
    image = calculate_lst(image)
    image = calculate_ndbi(image)

    # Select final bands for ML
    feature_image = image.select(['LST_Celsius', 'NDVI', 'NDBI', 'FVC'])

    # Sample 5000 random pixels from the ROI (enough for clustering)
    print("Sampling pixels...")
    samples = feature_image.sample(
        region=roi,
        scale=30,           # Landsat resolution: 30m per pixel
        numPixels=5000,
        seed=42,
        geometries=True     # Keep lat/lon coordinates
    )

    # Convert to Pandas DataFrame
    print("Exporting to CSV...")
    df = geemap.ee_to_df(samples)

    # Add lat/lon columns
    coords = samples.map(lambda f: f.set({
        'longitude': f.geometry().coordinates().get(0),
        'latitude':  f.geometry().coordinates().get(1)
    }))
    coords_df = geemap.ee_to_df(coords)

    df['latitude']  = coords_df['latitude']
    df['longitude'] = coords_df['longitude']

    # Clean: drop rows with missing values
    df = df.dropna()
    df = df.rename(columns={
        'LST_Celsius': 'lst',
        'NDVI':        'ndvi',
        'NDBI':        'ndbi',
        'FVC':         'fvc'
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nDone! Saved {len(df)} samples to {output_path}")
    print(df.describe())
    return df


# ─── Step 6: Process All Cities ──────────────────────────────────────────────
def process_all_cities():
    """Extract data for every city in the config."""
    for city_key in CITIES:
        try:
            process_and_export(city_key)
        except Exception as e:
            print(f"\nERROR processing {city_key}: {e}")
            continue


# ─── Step 7: Quick sanity check ──────────────────────────────────────────────
def verify_data(city_key):
    csv_path = f'data/{city_key}_features.csv'
    city = get_city(city_key)
    df = pd.read_csv(csv_path)
    print(f"\n{city['name']} — Data shape: {df.shape}")
    print(f"LST range:  {df['lst'].min():.1f}°C  to  {df['lst'].max():.1f}°C")
    print(f"NDVI range: {df['ndvi'].min():.3f}  to  {df['ndvi'].max():.3f}")
    print(f"NDBI range: {df['ndbi'].min():.3f}  to  {df['ndbi'].max():.3f}")
    return df


# ─── Run ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Process a specific city: python gee_extract.py delhi
        city_key = sys.argv[1].lower()
        df = process_and_export(city_key)
        verify_data(city_key)
    else:
        # Process all cities
        print("Processing all cities...")
        process_all_cities()
