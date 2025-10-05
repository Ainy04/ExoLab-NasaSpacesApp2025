from flask import jsonify, Blueprint
from src.Global import app 

home_bp = Blueprint("home", __name__)

@app.route('/')
def home():
    return jsonify({
        "message": "🪐 KOI Exoplanet API - Flask Backend Optimizado",
        "status": "running",
        "version": "2.0",
        "endpoints": {
            "upload": "POST /upload - Subir CSV y entrenar modelo optimizado",
            "data_info": "GET /data/info - Información del dataset",
            "statistics": "GET /data/statistics - Estadísticas descriptivas",
            "distribution": "POST /data/distribution - Distribución de columna",
            "correlation": "POST /data/correlation - Matriz de correlación",
            "scatter": "POST /data/scatter - Datos para scatter plot",
            "sample": "GET /data/sample?n=10 - Muestra de datos",
            "predict": "POST /predict - Hacer predicción",
            "download": "GET /download-model - Descargar modelo"
        }
    })