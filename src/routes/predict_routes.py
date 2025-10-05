from src.Global import app, MODEL_PATH
from flask import Blueprint, request, jsonify
import os
import joblib
import pandas as pd

predict_bp = Blueprint("predict", __name__)

@app.route('/predict', methods=['POST'])
def predict():
    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": "No hay modelo entrenado"}), 404
    
    model_bundle = joblib.load(MODEL_PATH)
    model = model_bundle["model"]
    scaler = model_bundle.get("scaler")
    label_encoder = model_bundle["label_encoder"]
    features = model_bundle["features"]
    
    data = request.json
    input_features = data.get('features')
    
    if not input_features:
        return jsonify({"error": "Parámetro 'features' requerido"}), 400
    
    input_df = pd.DataFrame([input_features])
    input_df = input_df[features]
    
    if scaler is not None:
        input_df = pd.DataFrame(scaler.transform(input_df), columns=features)
    
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    
    predicted_label = label_encoder.inverse_transform([prediction])[0]
    
    prob_dict = {
        label: float(prob)
        for label, prob in zip(label_encoder.classes_, probabilities)
    }
    
    label_names = {0: "No Candidato", 1: "Candidato"}
    readable_label = label_names.get(int(prediction), str(predicted_label))
    return jsonify({
        "prediction": readable_label,
        "prediction_encoded": int(prediction),
        "probabilities": prob_dict,
        "confidence": float(max(probabilities))
    })