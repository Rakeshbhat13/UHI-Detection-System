"""
Phase 2 & 3: ML Pipeline
Urban Heat Island Detection - Multi-City Support
K-Means clustering + Random Forest classification
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import json
import os
import sys

from cities import CITIES, DEFAULT_CITY, get_city

FEATURES = ['lst', 'ndvi', 'ndbi', 'fvc']


def get_paths(city_key):
    """Return all city-specific file paths."""
    return {
        'data':       f'data/{city_key}_features.csv',
        'model_dir':  f'ml/models/{city_key}',
        'geojson':    f'data/{city_key}_heatmap.geojson',
        'stats':      f'data/{city_key}_stats.json',
    }


# ─── Step 1: Load & Preprocess ───────────────────────────────────────────────
def load_data(city_key):
    paths = get_paths(city_key)
    df = pd.read_csv(paths['data'])
    print(f"Loaded {len(df)} samples for {get_city(city_key)['name']}")

    # Remove outliers (LST outside 15-55°C is likely noise)
    df = df[(df['lst'] > 15) & (df['lst'] < 55)]
    df = df[(df['ndvi'] > -0.3) & (df['ndvi'] < 1.0)]

    print(f"After cleaning: {len(df)} samples")
    return df


# ─── Step 2: K-Means Clustering (Heat Zone Discovery) ────────────────────────
def run_kmeans(df, city_key, n_clusters=4):
    """
    Cluster pixels into heat zones using LST, NDVI, NDBI.
    4 clusters naturally map to:
      - Cluster A: Water bodies / Dense vegetation (low LST, high NDVI)
      - Cluster B: Parks / Residential green areas
      - Cluster C: Residential / Mixed urban
      - Cluster D: Industrial / Commercial hotspots (high LST, high NDBI)
    """
    paths = get_paths(city_key)
    model_dir = paths['model_dir']
    os.makedirs(model_dir, exist_ok=True)

    print("\nRunning K-Means clustering...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURES])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    # ── Label clusters by average LST (lowest = 0, highest = 3) ──
    cluster_lst = df.groupby('cluster')['lst'].mean().sort_values()
    label_map   = {old: new for new, old in enumerate(cluster_lst.index)}
    df['heat_zone'] = df['cluster'].map(label_map)

    # Human-readable risk labels
    risk_labels = {0: 'Low', 1: 'Moderate', 2: 'High', 3: 'Critical'}
    df['risk_label'] = df['heat_zone'].map(risk_labels)

    # Print cluster summary
    print("\nCluster Summary:")
    summary = df.groupby('risk_label')[['lst', 'ndvi', 'ndbi']].mean().round(2)
    print(summary)

    # Save scaler and model
    joblib.dump(scaler, f'{model_dir}/scaler.pkl')
    joblib.dump(kmeans, f'{model_dir}/kmeans.pkl')

    # Save cluster centers for reference
    centers = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=FEATURES
    )
    centers.to_csv(f'{model_dir}/cluster_centers.csv', index=False)

    return df, kmeans, scaler


# ─── Step 3: Random Forest Classifier (Heat Risk Classification) ──────────────
def train_random_forest(df, city_key):
    """
    Train RF classifier on K-Means labels.
    This allows us to predict heat_zone for any new pixel
    without re-running clustering.
    """
    paths = get_paths(city_key)
    model_dir = paths['model_dir']

    print("\nTraining Random Forest classifier...")

    X = df[FEATURES]
    y = df['heat_zone']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # Evaluation
    y_pred = rf.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
          target_names=['Low', 'Moderate', 'High', 'Critical']))

    # Feature importance
    importance = dict(zip(FEATURES, rf.feature_importances_.round(3)))
    print(f"\nFeature Importance: {importance}")

    joblib.dump(rf, f'{model_dir}/random_forest.pkl')

    # Save importance for frontend charts
    with open(f'{model_dir}/feature_importance.json', 'w') as f:
        json.dump(importance, f)

    return rf


# ─── Step 4: Generate GeoJSON for Leaflet Map ────────────────────────────────
def generate_geojson(df, city_key):
    """
    Convert sampled pixels to GeoJSON FeatureCollection.
    Each point carries LST, NDVI, NDBI, risk_label.
    React + Leaflet will render this as a heatmap.
    """
    paths = get_paths(city_key)
    output_path = paths['geojson']
    city = get_city(city_key)

    print("\nGenerating GeoJSON...")

    color_map = {
        'Low':      '#2ecc71',   # green
        'Moderate': '#f1c40f',   # yellow
        'High':     '#e67e22',   # orange
        'Critical': '#e74c3c'    # red
    }

    features = []
    for _, row in df.iterrows():
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [row['longitude'], row['latitude']]
            },
            "properties": {
                "lst":        round(row['lst'], 2),
                "ndvi":       round(row['ndvi'], 3),
                "ndbi":       round(row['ndbi'], 3),
                "risk":       row['risk_label'],
                "heat_zone":  int(row['heat_zone']),
                "color":      color_map[row['risk_label']],
                "city":       city['name']
            }
        }
        features.append(feature)

    geojson = {"type": "FeatureCollection", "features": features}

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(geojson, f)

    print(f"Saved {len(features)} points to {output_path}")
    return geojson


# ─── Step 5: Generate Stats for Dashboard ────────────────────────────────────
def generate_stats(df, city_key):
    """
    Summary statistics for the React dashboard cards.
    """
    paths = get_paths(city_key)
    output_path = paths['stats']
    city = get_city(city_key)

    risk_counts = df['risk_label'].value_counts().to_dict()

    stats = {
        "city": city['name'],
        "city_key": city_key,
        "total_pixels": len(df),
        "avg_lst": round(df['lst'].mean(), 2),
        "max_lst": round(df['lst'].max(), 2),
        "min_lst": round(df['lst'].min(), 2),
        "avg_ndvi": round(df['ndvi'].mean(), 3),
        "risk_distribution": risk_counts,
        "zone_stats": {}
    }

    for risk in ['Low', 'Moderate', 'High', 'Critical']:
        zone = df[df['risk_label'] == risk]
        if len(zone) > 0:
            stats["zone_stats"][risk] = {
                "count":    len(zone),
                "avg_lst":  round(zone['lst'].mean(), 2),
                "avg_ndvi": round(zone['ndvi'].mean(), 3),
                "avg_ndbi": round(zone['ndbi'].mean(), 3),
                "pct":      round(len(zone) / len(df) * 100, 1)
            }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nStats saved to {output_path}")
    print(json.dumps(stats, indent=2))
    return stats


# ─── Run Pipeline for One City ────────────────────────────────────────────────
def run_pipeline(city_key):
    """Run the full ML pipeline for a single city."""
    city = get_city(city_key)
    print(f"\n{'='*60}")
    print(f"  ML Pipeline: {city['name']}")
    print(f"{'='*60}")

    df                 = load_data(city_key)
    df, kmeans, scaler = run_kmeans(df, city_key)
    rf                 = train_random_forest(df, city_key)
    generate_geojson(df, city_key)
    generate_stats(df, city_key)
    print(f"\n[OK] Pipeline complete for {city['name']}!")


# ─── Run Pipeline for All Cities ──────────────────────────────────────────────
def run_all_cities():
    """Run the ML pipeline for every city that has extracted data."""
    for city_key in CITIES:
        paths = get_paths(city_key)
        if os.path.exists(paths['data']):
            try:
                run_pipeline(city_key)
            except Exception as e:
                print(f"\nERROR processing {city_key}: {e}")
                continue
        else:
            print(f"\nSkipping {city_key}: no data file at {paths['data']}")


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Process a specific city: python pipeline.py delhi
        city_key = sys.argv[1].lower()
        run_pipeline(city_key)
    else:
        # Process all cities that have data
        run_all_cities()
    print("\nML Pipeline complete!")
