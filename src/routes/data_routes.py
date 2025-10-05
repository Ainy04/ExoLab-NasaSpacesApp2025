from flask import Blueprint, request, jsonify
from src.Global import app, DATA_PATH, TARGET_COLUMNS
import pandas as pd
from src.utils import detect_dataset, get_available_columns
import os
import numpy as np

data_bp = Blueprint('data', __name__)

@app.route('/data/info', methods=['GET'])
def get_data_info():
    if not os.path.exists(DATA_PATH):
        return jsonify({"error": "No hay datos cargados"}), 404
    
    df = pd.read_csv(DATA_PATH)
    dataset_type = detect_dataset(df)
    target_col = TARGET_COLUMNS.get(dataset_type, None) if dataset_type else None
    
    info = {
        "dataset_type": dataset_type,
        "target_column": target_col,
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": df.isnull().sum().to_dict(),
        "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist()
    }
    
    return jsonify(info)

@app.route('/data/config', methods=['GET'])
def get_dataset_config():
    """Devuelve la configuración de columnas disponibles para el dataset"""
    if not os.path.exists(DATA_PATH):
        return jsonify({"error": "No hay datos cargados"}), 404
    
    df = pd.read_csv(DATA_PATH)
    dataset_type = detect_dataset(df)
    
    if not dataset_type:
        return jsonify({"error": "Tipo de dataset no reconocido"}), 400
    
    available_cols = get_available_columns(df, dataset_type)
    
    return jsonify({
        "dataset_type": dataset_type,
        "available_columns": available_cols,
        "all_columns": list(df.columns)
    })

@app.route('/data/statistics', methods=['GET'])
def get_statistics():
    if not os.path.exists(DATA_PATH):
        return jsonify({"error": "No hay datos cargados"}), 404
    
    df = pd.read_csv(DATA_PATH)
    numeric_df = df.select_dtypes(include=[np.number])
    stats = numeric_df.describe().to_dict()
    
    return jsonify(stats)

@app.route('/data/distribution', methods=['POST'])
def get_distribution():
    if not os.path.exists(DATA_PATH):
        return jsonify({"error": "No hay datos cargados"}), 404
    
    data = request.json
    column = data.get('column')
    
    if not column:
        return jsonify({"error": "Parámetro 'column' requerido"}), 400
    
    df = pd.read_csv(DATA_PATH)
    
    if column not in df.columns:
        return jsonify({"error": f"Columna '{column}' no existe"}), 404
    
    column_data = df[column].dropna()
    
    if len(column_data) == 0:
        return jsonify({"error": f"La columna '{column}' no tiene valores válidos"}), 400
    
    if pd.api.types.is_numeric_dtype(column_data):
        hist, bins = np.histogram(column_data, bins=30)
        return jsonify({
            "type": "numeric",
            "histogram": hist.tolist(),
            "bins": bins.tolist(),
            "mean": float(column_data.mean()),
            "median": float(column_data.median()),
            "std": float(column_data.std()),
            "valid_count": len(column_data)
        })
    else:
        counts = column_data.value_counts().to_dict()
        return jsonify({
            "type": "categorical",
            "counts": counts,
            "valid_count": len(column_data)
        })
    
@app.route('/data/correlation', methods=['POST'])
def get_correlation():
    if not os.path.exists(DATA_PATH):
        return jsonify({"error": "No hay datos cargados"}), 404
    
    data = request.json
    columns = data.get('columns', [])
    
    df = pd.read_csv(DATA_PATH)
    numeric_df = df.select_dtypes(include=[np.number])
    
    if columns:
        numeric_df = numeric_df[columns]
    
    corr_matrix = numeric_df.corr().to_dict()
    
    return jsonify(corr_matrix)

@app.route('/data/scatter', methods=['POST'])
def get_scatter_data():
    if not os.path.exists(DATA_PATH):
        return jsonify({"error": "No hay datos cargados"}), 404
    
    data = request.json
    x_col = data.get('x_column')
    y_col = data.get('y_column')
    color_col = data.get('color_column')
    
    if not x_col or not y_col:
        return jsonify({"error": "Parámetros 'x_column' y 'y_column' requeridos"}), 400
    
    df = pd.read_csv(DATA_PATH)
    
    cols_to_use = [x_col, y_col]
    if color_col and color_col in df.columns:
        cols_to_use.append(color_col)
    
    df_clean = df[cols_to_use].dropna()
    
    result = {
        "x": df_clean[x_col].tolist(),
        "y": df_clean[y_col].tolist()
    }
    
    if color_col and color_col in df.columns:
        result["color"] = df_clean[color_col].tolist()
    
    return jsonify(result)

@app.route('/data/sample', methods=['GET'])
def get_sample():
    if not os.path.exists(DATA_PATH):
        return jsonify({"error": "No hay datos cargados"}), 404
    
    n = request.args.get('n', default=10, type=int)
    df = pd.read_csv(DATA_PATH)
    sample = df.head(n).to_dict('records')
    
    return jsonify(sample)

