from flask import request, Blueprint
from src.limpia import clean_dataset
import joblib

#archivos locales del proyecto
from src.posTrainFunctions import *
from src.preTrainFunctions import *
from src.RandomForest import train_randomForest
from src.XGBOOST import train_xgboost
from src.Global import *
from src.utils import *


upload_bp = Blueprint("upload", __name__)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No se encontró el campo 'file' en la petición."}), 400
    
    f = request.files['file']

    # Obtener modelo seleccionado (por defecto RandomForest)
    selected_model = request.form.get('model', 'RandomForest')
    print(f"🎯 Modelo seleccionado: {selected_model}")
    
    try:
        df = pd.read_csv(f, comment='#')
    except Exception as e:
        try:
            f.seek(0)
            df = pd.read_csv(f, comment='#', on_bad_lines='skip', engine='python')
        except Exception as e2:
            return jsonify({"error": f"Error al leer CSV: {e2}"}), 400
    
    try:
        df_clean, target_col = clean_dataset(df)
    except Exception as e:
        return jsonify({"error": f"No se pudo detectar la columna objetivo: {e}"}), 400
    
    # Guardar versión original textual (por si queremos explorarla sin pérdida)
    if target_col in df_clean.columns:
        df_clean[f"{target_col}_original"] = df_clean[target_col].copy()

    # convertir a binarios los target para el entrenamiento
    df_clean[target_col] = convers_binary(df_clean, target_col)

    # Añadir columna legible (EN MAYÚSCULAS para que coincida con otros endpoints)
    # Esto mantiene la columna numérica para el entrenamiento y una columna textual para visualización
    df_clean[f"{target_col}_label"] = df_clean[target_col].map(
        {0: 'NOT CANDIDATE', 1: 'CANDIDATE'}
    ).fillna(df_clean[target_col].astype(str))
    
    try:
        df_proc, label_encoder = preprocess_optimized(df_clean, target_col)
    except Exception as e:
        return jsonify({"error": f"Error en preprocesamiento: {e}"}), 400
    
    # Guardar datos limpios
    df_clean.to_csv(DATA_PATH, index=False)
    dataset_type = detect_dataset(df_clean)
    
    y = df_proc[target_col]
    X = df_proc.drop(columns=[target_col])
    
    if X.shape[1] == 0:
        return jsonify({"error": "No se encontraron features numéricas para entrenar."}), 400

        
    #valores para el entrenamiento
    values, scaler = prepare_Data_for_Train(df_proc, target_col)
    
    # ============================================
    # 🎯 SELECCIÓN DINÁMICA DEL MODELO
    # ============================================
    if selected_model == 'XGBoost':
        print("🚀 Entrenando XGBoost...")
        clf = train_xgboost(values)
        model_name = "XGBoost"
    else:
        print("🌲 Entrenando RandomForest...")
        clf = train_randomForest(values)
        model_name = "RandomForest"
    
    # Cross-validation
    cv_results = calculate_cv_scores(clf, values)
    
    # Predicciones
    y_pred, metrics = predict_and_metrics(clf, values)
    
    X_scaled = values["X_scaled"]

    # Feature importances
    fi = [{"feature": name, "importance": float(imp)}
          for name, imp in zip(X_scaled.columns, clf.feature_importances_)]
    fi_sorted = sorted(fi, key=lambda x: x['importance'], reverse=True)
    
    # Generar plots
    plots = generate_ml_plots(values, y_pred, clf, label_encoder)
    
    # Guardar modelo
    model_bundle = {
        "model": clf,
        "model_type": model_name,
        "scaler": scaler,
        "values_to_train" : values,
        "label_encoder": label_encoder,
        "features": list(X_scaled.columns),
        "dataset_type": dataset_type,
        "target_column": target_col
    }
    joblib.dump(model_bundle, MODEL_PATH)

    label_mapping = ["NOT CANDIDATE", "CANDIDATE"]
    
    response = {
        "model_type": model_name,
        "dataset_type": dataset_type,
        "target_column": target_col,
        "metrics": metrics,
        "cross_val": cv_results,
        "feature_importances": fi_sorted,
        "model_filename": MODEL_PATH,
        "label_mapping": label_mapping, 
        "train_size": values["train_size"],
        "test_size": values["test_size"],
        "total_rows": len(df),
        "total_features": len(X_scaled.columns),
        "plots": plots
    }
    
    return jsonify(response)