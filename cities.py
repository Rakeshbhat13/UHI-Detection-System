"""
City Configuration for Urban Heat Island Detection
Each city has: name, center coords, bounding box, and peak summer date range.
"""

CITIES = {
    "bengaluru": {
        "name": "Bengaluru",
        "lat": 12.97,
        "lon": 77.59,
        "bbox": [77.45, 12.83, 77.78, 13.14],
        "start_date": "2022-01-01",
        "end_date":   "2023-12-31",
    },
    "hyderabad": {
        "name": "Hyderabad",
        "lat": 17.39,
        "lon": 78.49,
        "bbox": [78.30, 17.30, 78.60, 17.55],
        "start_date": "2022-01-01",
        "end_date":   "2023-12-31",
    },
    "mumbai": {
        "name": "Mumbai",
        "lat": 19.08,
        "lon": 72.88,
        "bbox": [72.77, 18.88, 73.05, 19.15],
        "start_date": "2022-01-01",
        "end_date":   "2023-12-31",
    },
    "delhi": {
        "name": "Delhi",
        "lat": 28.61,
        "lon": 77.21,
        "bbox": [76.90, 28.45, 77.35, 28.80],
        "start_date": "2022-01-01",
        "end_date":   "2023-12-31",
    },
    "chennai": {
        "name": "Chennai",
        "lat": 13.08,
        "lon": 80.27,
        "bbox": [80.15, 12.90, 80.35, 13.15],
        "start_date": "2022-01-01",
        "end_date":   "2023-12-31",
    },
    "mangaluru": {
        "name": "Mangaluru",
        "lat": 12.87,
        "lon": 74.88,
        "bbox": [74.80, 12.80, 74.95, 12.95],
        "start_date": "2022-01-01",
        "end_date":   "2023-12-31",
    },
    "ahmedabad": {
        "name": "Ahmedabad",
        "lat": 23.02,
        "lon": 72.57,
        "bbox": [72.50, 22.95, 72.70, 23.15],
        "start_date": "2022-01-01",
        "end_date":   "2023-12-31",
    },
    "kolkata": {
        "name": "Kolkata",
        "lat": 22.57,
        "lon": 88.36,
        "bbox": [88.25, 22.45, 88.45, 22.65],
        "start_date": "2022-01-01",
        "end_date":   "2023-12-31",
    },
    "pune": {
        "name": "Pune",
        "lat": 18.52,
        "lon": 73.86,
        "bbox": [73.75, 18.45, 73.95, 18.65],
        "start_date": "2022-01-01",
        "end_date":   "2023-12-31",
    },
}

DEFAULT_CITY = "bengaluru"


def get_city(city_key):
    """Get city config by key. Raises KeyError if not found."""
    city_key = city_key.lower().strip()
    if city_key not in CITIES:
        available = ", ".join(CITIES.keys())
        raise KeyError(f"City '{city_key}' not found. Available: {available}")
    return CITIES[city_key]


def list_cities():
    """Return list of city dicts with key included."""
    return [
        {"key": k, "name": v["name"], "lat": v["lat"], "lon": v["lon"]}
        for k, v in CITIES.items()
    ]
