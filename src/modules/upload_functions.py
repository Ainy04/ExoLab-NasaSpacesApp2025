"""
Funciones de carga y entrenamiento (sin Flask)
Para usar directamente en Streamlit
"""
import pandas as pd
import joblib
from src.limpia import clean_dataset
from src.posTrainFunctions import *
from src.preTrainFunctions import *
from src.RandomForest import train_randomForest
from src.XGBOOST import train_xgboost
from src.utils import detect_dataset
from src.Global import TARGET_COLUMNS


def convers_binary(df, target_col):
    """
    Convierte la columna target a valores binarios (0/1)
    Implementa la lógica que necesites según tu dataset
    """
    # Mapeo básico - ajusta según tus necesidades
    mapping = {
        'CONFIRMED': 1,
        'CANDIDATE': 1,
        'FALSE POSITIVE': 0,
        'NOT CANDIDATE': 0,
        0: 0,
        1: 1
    }
    
    # Normalizar a mayúsculas y aplicar mapeo
    df[target_col] = df[target_col].astype(str).str.upper().map(mapping)
    
    # Si hay valores no mapeados, convertir a 0 o 1 según lógica
    df[target_col] = df[target_col].fillna(0).astype(int)
    
    return df[target_col]


def upload_and_train(file_content, selected_model='RandomForest', data_path='data/uploaded.csv', model_path='models/model.joblib'):
    """
    Procesa CSV, entrena modelo y retorna resultados
    
    Args:
        file_content: contenido del archivo CSV (puede ser BytesIO o file object)
        selected_model: 'RandomForest' o 'XGBoost'
        data_path: ruta donde guardar datos limpios
        model_path: ruta donde guardar modelo
    
    Returns:
        dict con métricas, plots, feature importance, etc.
    """
    
    # 1. LEER CSV
    try:
        df = pd.read_csv(file_content, comment='#')
    except Exception as e:
        try:
            file_content.seek(0)
            df = pd.read_csv(file_content, comment='#', on_bad_lines='skip', engine='python')
        except Exception as e2:
            return {"error": f"Error al leer CSV: {e2}"}
    
    # 2. LIMPIAR DATASET
    try:
        df_clean, target_col = clean_dataset(df)
    except Exception as e:
        return {"error": f"No se pudo detectar la columna objetivo: {e}"}
    
    # 3. GUARDAR VERSIÓN ORIGINAL TEXTUAL
    if target_col in df_clean.columns:
        df_clean[f"{target_col}_original"] = df_clean[target_col].copy()

    # 4. CONVERTIR A BINARIOS
    df_clean[target_col] = convers_binary(df_clean, target_col)

    # 5. AÑADIR COLUMNA LEGIBLE (EN MAYÚSCULAS)
    df_clean[f"{target_col}_label"] = df_clean[target_col].map(
        {0: 'NOT CANDIDATE', 1: 'CANDIDATE'}
    ).fillna(df_clean[target_col].astype(str))
    
    # 6. PREPROCESAR
    try:
        df_proc, label_encoder = preprocess_optimized(df_clean, target_col)
    except Exception as e:
        return {"error": f"Error en preprocesamiento: {e}"}
    
    # 7. GUARDAR DATOS LIMPIOS
    df_clean.to_csv(data_path, index=False)
    dataset_type = detect_dataset(df_clean)
    
    # 8. PREPARAR X e Y
    y = df_proc[target_col]
    X = df_proc.drop(columns=[target_col])
    
    if X.shape[1] == 0:
        return {"error": "No se encontraron features numéricas para entrenar."}

    # 9. PREPARAR DATOS PARA ENTRENAMIENTO
    values, scaler = prepare_Data_for_Train(df_proc, target_col)
    
    # 10. ENTRENAR MODELO SEGÚN SELECCIÓN
    if selected_model == 'XGBoost':
        print("Entrenando XGBoost...")
        clf = train_xgboost(values)
        model_name = "XGBoost"
    else:
        print("Entrenando RandomForest...")
        clf = train_randomForest(values)
        model_name = "RandomForest"
    
    # 11. CROSS-VALIDATION
    cv_results = calculate_cv_scores(clf, values)
    
    # 12. PREDICCIONES Y MÉTRICAS
    y_pred, metrics = predict_and_metrics(clf, values)
    
    X_scaled = values["X_scaled"]

    # 13. FEATURE IMPORTANCES
    fi = [{"feature": name, "importance": float(imp)}
          for name, imp in zip(X_scaled.columns, clf.feature_importances_)]
    fi_sorted = sorted(fi, key=lambda x: x['importance'], reverse=True)
    
    # 14. GENERAR PLOTS
    plots = generate_ml_plots(values, y_pred, clf, label_encoder)
    
    # 15. GUARDAR MODELO
    model_bundle = {
        "model": clf,
        "model_type": model_name,
        "scaler": scaler,
        "values_to_train": values,
        "label_encoder": label_encoder,
        "features": list(X_scaled.columns),
        "dataset_type": dataset_type,
        "target_column": target_col
    }
    joblib.dump(model_bundle, model_path)

    label_mapping = ["NOT CANDIDATE", "CANDIDATE"]
    
    # 16. RETORNAR RESULTADOS
    response = {
        "model_type": model_name,
        "dataset_type": dataset_type,
        "target_column": target_col,
        "metrics": metrics,
        "cross_val": cv_results,
        "feature_importances": fi_sorted,
        "model_filename": model_path,
        "label_mapping": label_mapping, 
        "train_size": values["train_size"],
        "test_size": values["test_size"],
        "total_rows": len(df),
        "total_features": len(X_scaled.columns),
        "plots": plots
    }
    
    return response