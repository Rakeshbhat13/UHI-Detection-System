"""
Flask API - Urban Heat Island Detection
Serves ML results to React frontend — Multi-City Support
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import json
import os

from cities import CITIES, DEFAULT_CITY, get_city, list_cities

app = Flask(__name__)
CORS(app)  # Allow React frontend to call this API


# ─── Helper: Load cached data ─────────────────────────────────────────────────
def load_json(city_key, filename):
    path = os.path.join('data', f'{city_key}_{filename}')
    with open(path) as f:
        return json.load(f)


# ─── Route 0: Serve frontend ──────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


# ─── Route 1: Health check ────────────────────────────────────────────────────
@app.route('/api/health')
def health():
    return jsonify({"status": "ok", "message": "UHI API running"})


# ─── Route 2: List available cities ───────────────────────────────────────────
@app.route('/api/cities')
def get_cities():
    """Returns list of all supported cities with coordinates."""
    cities = list_cities()
    # Mark which cities have data available
    for c in cities:
        data_file = f'data/{c["key"]}_stats.json'
        c['has_data'] = os.path.exists(data_file)
    return jsonify(cities)


# ─── Route 3: City stats for dashboard cards ─────────────────────────────────
@app.route('/api/stats')
def get_stats():
    """Returns summary statistics for the dashboard. Use ?city=delhi"""
    city_key = request.args.get('city', DEFAULT_CITY).lower()
    try:
        get_city(city_key)  # Validate city exists
        stats = load_json(city_key, 'stats.json')
        return jsonify(stats)
    except KeyError as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError:
        return jsonify({"error": f"No data for {city_key}. Run ML pipeline first."}), 404


# ─── Route 4: GeoJSON heatmap data for Leaflet map ───────────────────────────
@app.route('/api/heatmap')
def get_heatmap():
    """
    Returns GeoJSON FeatureCollection of sampled pixels.
    Query params: ?city=delhi&risk=Critical
    """
    city_key = request.args.get('city', DEFAULT_CITY).lower()
    try:
        get_city(city_key)
        geojson = load_json(city_key, 'heatmap.geojson')

        risk_filter = request.args.get('risk')
        if risk_filter:
            geojson['features'] = [
                f for f in geojson['features']
                if f['properties']['risk'] == risk_filter
            ]

        return jsonify(geojson)
    except KeyError as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError:
        return jsonify({"error": f"No data for {city_key}. Run ML pipeline first."}), 404


# ─── Route 5: Feature importance for bar chart ───────────────────────────────
@app.route('/api/feature-importance')
def get_feature_importance():
    """Returns feature importance. Use ?city=delhi"""
    city_key = request.args.get('city', DEFAULT_CITY).lower()
    try:
        get_city(city_key)
        model_dir = f'ml/models/{city_key}'
        path = os.path.join(model_dir, 'feature_importance.json')
        with open(path) as f:
            importance = json.load(f)

        # Format for Recharts
        data = [
            {"feature": "LST (°C)",  "importance": importance.get("lst",  0)},
            {"feature": "NDVI",      "importance": importance.get("ndvi", 0)},
            {"feature": "NDBI",      "importance": importance.get("ndbi", 0)},
            {"feature": "FVC",       "importance": importance.get("fvc",  0)},
        ]
        return jsonify(data)
    except KeyError as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError:
        return jsonify({"error": f"No model for {city_key}. Run ML pipeline first."}), 404


# ─── Route 6: Mitigation suggestions based on zone stats ─────────────────────
@app.route('/api/mitigation/<risk_level>')
def get_mitigation(risk_level):
    """
    Returns data-driven mitigation suggestions for a given risk zone.
    """
    suggestions = {
        "Low": {
            "summary": "This zone has healthy vegetation and low surface temperature.",
            "actions": [
                "Maintain existing green cover and tree canopy.",
                "Use permeable pavements to preserve groundwater recharge.",
                "Monitor NDVI seasonally to detect deforestation early."
            ],
            "priority": "Preventive"
        },
        "Moderate": {
            "summary": "Rising temperatures detected. Vegetation loss likely contributing.",
            "actions": [
                "Plant trees along roads and pavements (target +10% green cover).",
                "Install green roofs on residential buildings.",
                "Increase park density — at least one park per 500m radius.",
                "Use light-colored road surfaces to reduce heat absorption."
            ],
            "priority": "Medium"
        },
        "High": {
            "summary": "Significant UHI effect. Dense built-up areas reducing cooling.",
            "actions": [
                "Mandate cool roofs (reflective coating) for new constructions.",
                "Create urban forests with native species along arterial roads.",
                "Implement water bodies / fountains in public spaces.",
                "Zoning policy: require 25% green area in new developments.",
                "Reduce dark asphalt coverage in parking lots."
            ],
            "priority": "High"
        },
        "Critical": {
            "summary": "Extreme heat hotspot. Immediate intervention recommended.",
            "actions": [
                "Urgent: plant shade trees along all major roads in this zone.",
                "Retrofit existing buildings with cool roofs and wall insulation.",
                "Create emergency cooling centers for vulnerable populations.",
                "Restrict new construction until green infrastructure plan is in place.",
                "Install misting systems at public transit stops.",
                "Restore or create water bodies (lakes, retention ponds)."
            ],
            "priority": "Critical"
        }
    }

    risk_level = risk_level.capitalize()
    if risk_level not in suggestions:
        return jsonify({"error": "Invalid risk level"}), 400

    return jsonify({
        "risk_level": risk_level,
        **suggestions[risk_level]
    })


# ─── Route 7: Predict zone for a new pixel ───────────────────────────────────
@app.route('/api/predict', methods=['POST'])
def predict():
    """
    POST: { "city": "delhi", "lst": 38.5, "ndvi": 0.12, "ndbi": 0.18, "fvc": 0.08 }
    Returns: { "risk": "High", "heat_zone": 2 }
    """
    try:
        data = request.json
        city_key = data.get('city', DEFAULT_CITY).lower()
        get_city(city_key)

        model_dir = f'ml/models/{city_key}'
        features = [[
            data['lst'],
            data['ndvi'],
            data['ndbi'],
            data.get('fvc', 0.1)
        ]]

        scaler = joblib.load(f'{model_dir}/scaler.pkl')
        rf     = joblib.load(f'{model_dir}/random_forest.pkl')

        zone = int(rf.predict(features)[0])
        risk_labels = {0: 'Low', 1: 'Moderate', 2: 'High', 3: 'Critical'}

        return jsonify({
            "heat_zone": zone,
            "risk":      risk_labels[zone],
            "city":      city_key
        })
    except KeyError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
