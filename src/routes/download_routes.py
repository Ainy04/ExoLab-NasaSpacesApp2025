from src.Global import app, MODEL_PATH
from flask import Blueprint,send_file, jsonify
import os

download_bp = Blueprint("download", __name__)

@app.route('/download-model', methods=['GET'])
def download_model():
    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": "No hay modelo guardado"}), 404
    return send_file(MODEL_PATH, as_attachment=True)
