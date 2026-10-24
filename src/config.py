"""Shared settings: paths, feature lists, flood events."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)

DATASET_CSV = DATA / 'flood_dataset_all_events.csv'

# Google Earth Engine cloud project and Drive export folder
EE_PROJECT = 'flood-mapping-511811'
DRIVE_FOLDER = 'flood_project'
CRS = 'EPSG:32644'          # UTM zone 44N (Telangana / Andhra Pradesh)

SEED = 42
TARGET = 'label'

# Continuous factors (aspect is replaced by aspect_sin / aspect_cos) and categorical factors
NUM = ['elevation', 'slope', 'aspect_sin', 'aspect_cos', 'curvature', 'twi', 'spi',
       'hand', 'dist_river', 'drain_dens', 'ndvi', 'rain_event', 'rain_antec']
CAT = ['lulc', 'soil_tex']

# Columns exported from GEE (dataset CSV and factor stacks), in this order
GEE_COLUMNS = ['elevation', 'slope', 'aspect', 'curvature', 'twi', 'spi', 'hand',
               'dist_river', 'drain_dens', 'lulc', 'ndvi', 'soil_tex',
               'rain_event', 'rain_antec']

# Flood events: study-area box [west, south, east, north] and Sentinel-1 date windows
EVENTS = [
    {'name': 'godavari_2022', 'aoi': [80.3, 17.2, 81.4, 18.3],
     'pre': ('2022-06-01', '2022-07-05'), 'post': ('2022-07-14', '2022-07-23'),
     'flood_date': '2022-07-14'},
    {'name': 'munneru_2024', 'aoi': [79.8, 16.95, 80.9, 17.6],
     'pre': ('2024-08-01', '2024-08-26'), 'post': ('2024-08-31', '2024-09-08'),
     'flood_date': '2024-09-01'},
    {'name': 'budameru_2024', 'aoi': [80.3, 16.3, 81.0, 16.9],
     'pre': ('2024-08-01', '2024-08-26'), 'post': ('2024-08-31', '2024-09-08'),
     'flood_date': '2024-09-01'},
]

# Areas the model never trained on (rainfall scenario instead of a real event)
APPLICATION_AREAS = {
    'hyderabad': {'aoi': [78.20, 17.20, 78.70, 17.60], 'scale': 30},
    'warangal':  {'aoi': [79.35, 17.80, 79.85, 18.15], 'scale': 30},
}

# Typical heavy-monsoon scenario = median of the training data
SCENARIO = {'rain_event': 150, 'rain_antec': 265, 'ndvi_date': '2024-09-01'}

CLASS_NAMES = ['Very low', 'Low', 'Moderate', 'High', 'Very high']
CLASS_COLORS = ['#1a9850', '#91cf60', '#fee08b', '#fc8d59', '#d73027']
