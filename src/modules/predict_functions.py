"""
Funciones de predicción (sin Flask)
Para usar directamente en Streamlit
"""
import pandas as pd
import joblib


def predict(model_path, input_features):
    """
    Realiza predicción con el modelo guardado
    
    Args:
        model_path: ruta al archivo .joblib del modelo
        input_features: dict con las features de entrada
    
    Returns:
        dict con predicción, probabilidades y confianza
    """
    try:
        model_bundle = joblib.load(model_path)
        model = model_bundle["model"]
        scaler = model_bundle.get("scaler")
        label_encoder = model_bundle["label_encoder"]
        features = model_bundle["features"]
        
        # Crear DataFrame con features
        input_df = pd.DataFrame([input_features])
        input_df = input_df[features]
        
        # Escalar si existe scaler
        if scaler is not None:
            input_df = pd.DataFrame(scaler.transform(input_df), columns=features)
        
        # Predicción
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        
        predicted_label = label_encoder.inverse_transform([prediction])[0]
        
        prob_dict = {
            label: float(prob)
            for label, prob in zip(label_encoder.classes_, probabilities)
        }
        
        label_names = {0: "No Candidato", 1: "Candidato"}
        readable_label = label_names.get(int(prediction), str(predicted_label))
        
        return {
            "prediction": readable_label,
            "prediction_encoded": int(prediction),
            "probabilities": prob_dict,
            "confidence": float(max(probabilities))
        }
    
    except FileNotFoundError:
        return {"error": "No hay modelo entrenado"}
    except Exception as e:
        return {"error": f"Error en predicción: {str(e)}"}