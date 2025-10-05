from flask_cors import CORS
from flask import Flask

app = Flask(__name__)
CORS(app)

MODEL_PATH = "models/model.joblib"
DATA_PATH = "datasets/current_data.csv"

TARGET_COLUMNS = {
    "koi": "koi_pdisposition",
    "toi": "tfopwg_disp",
    "k2": "disposition"
}

# Configuración de columnas por dataset
DATASET_CONFIG = {
    "koi": {
        "disposition": "koi_disposition",
        "period": "koi_period",
        "radius": "koi_prad",
        "temperature": "koi_teq",
        "insolation": "koi_insol",
        "stellar_temp": "koi_steff",
        "stellar_radius": "koi_srad",
        "duration": "koi_duration",    
        "depth": "koi_depth" 
    },
    "toi": {
        "disposition": "tfopwg_disp",
        "period": "pl_orbper",
        "radius": "pl_rade",
        "temperature": "pl_eqt",
        "insolation": "pl_insol",
        "stellar_temp": "st_teff",
        "stellar_radius": "st_rad"
    },
    "k2": {
        "disposition": "disposition",
        "period": "pl_orbper",
        "radius": "pl_rade",
        "stellar_temp": "st_teff",
        "distance": "sy_dist",
        "stellar_radius": "st_rad"
    }
}

# 🔹 palabras clave para detectar la columna objetivo
TARGET_CANDIDATES = [
    "disposition", "koi_pdisposition", "tfopwg_disp"
]

# 🔹 columnas recomendadas para ML (unificando Kepler, TESS y Exoplanet Archive)
USEFUL_COLUMNS = [
    # === características planetarias ===
    "pl_orbper", "koi_period", "pl_orbsmax",
    "pl_rade", "koi_prad", "pl_radj",
    "pl_bmasse", "pl_bmassj", "pl_orbeccen",
    "pl_insol", "koi_insol", "pl_eqt", "koi_teq",
    "pl_trandurh", "koi_duration",
    "pl_trandep", "koi_depth",

    # === características estelares ===
    "st_teff", "koi_steff",
    "st_rad", "koi_srad", "st_mass",
    "st_met", "st_metratio",
    "st_logg", "koi_slogg",
    "st_dist", "sy_dist",
    "sy_vmag", "sy_kmag", "sy_gaiamag",
    "st_tmag",

    # === sistema y contexto ===
    "sy_snum", "sy_pnum",
    "discoverymethod", "disc_year",
    "ra", "dec",
]
