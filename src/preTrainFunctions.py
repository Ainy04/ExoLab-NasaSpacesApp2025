import numpy as np
from flask import jsonify
from sklearn.model_selection import train_test_split
from src.limpia import preprocess_df
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from src.utils import detect_dataset

def feature_engineering(df, target_col):
    """Crear features derivadas para mejorar predicción"""
    df = df.copy()
    
    try:
        # Ratio duración/periodo
        if 'koi_period' in df.columns and 'koi_duration' in df.columns:
            mask = (df['koi_period'] > 0) & df['koi_period'].notna() & df['koi_duration'].notna()
            df.loc[mask, 'transit_duration_ratio'] = df.loc[mask, 'koi_duration'] / df.loc[mask, 'koi_period']
            df['transit_duration_ratio'].fillna(0, inplace=True)
        
        # Profundidad normalizada por radio estelar
        if 'koi_depth' in df.columns and 'koi_srad' in df.columns:
            mask = (df['koi_srad'] > 0) & df['koi_srad'].notna() & df['koi_depth'].notna()
            df.loc[mask, 'depth_per_stellar_radius'] = df.loc[mask, 'koi_depth'] / (df.loc[mask, 'koi_srad'] ** 2)
            df['depth_per_stellar_radius'].fillna(0, inplace=True)
        
        # Proxy de luminosidad estelar
        if 'koi_steff' in df.columns and 'koi_srad' in df.columns:
            mask = df['koi_steff'].notna() & df['koi_srad'].notna()
            df.loc[mask, 'stellar_luminosity_proxy'] = df.loc[mask, 'koi_steff'] * (df.loc[mask, 'koi_srad'] ** 2)
            df['stellar_luminosity_proxy'].fillna(0, inplace=True)
        
    except Exception as e:
        print(f"Feature engineering warning: {e}")
    
    return df

#preprocesamiento optimizado
def preprocess_optimized(df_clean, target_col):
    """Preprocesamiento mejorado con manejo robusto de NaN"""
    df_proc, label_encoder = preprocess_df(df_clean, target_col)
    
    # Feature engineering
    df_proc = feature_engineering(df_proc, target_col)
    
    # Eliminación total de NaN e infinitos
    df_proc.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    for col in df_proc.columns:
        if df_proc[col].isna().any():
            df_proc[col].fillna(df_proc[col].median() if col != target_col else 0, inplace=True)
    
    # Verificación final
    if df_proc.isna().sum().sum() > 0:
        df_proc.fillna(0, inplace=True)
    
    return df_proc, label_encoder

#preparar datos para entrenar el modelo
def prepare_Data_for_Train(df_proc, target_col):
    
    y = df_proc[target_col] #definir targer  
    X = df_proc.drop(columns=[target_col]) 

    if X.shape[1] == 0:
        return jsonify({"error": "No se encontraron features numéricas para entrenar."}), 400
    
    # Feature selection si hay muchas features
    if X.shape[1] > 50:
        selector = SelectKBest(score_func=f_classif, k=50)
        X_selected = selector.fit_transform(X, y)
        selected_cols = X.columns[selector.get_support()]
        X = pd.DataFrame(X_selected, columns=selected_cols, index=X.index)
    
    # Escalado
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    stratify = y if len(np.unique(y)) > 1 and len(y) >= 10 else None
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        test_size=0.2, 
                                                        random_state=42, 
                                                        stratify=stratify)

    train_size = len(X_train)
    test_size = len(X_test)

    values_train = {
        "X" : X,
        "y" : y,
        "X_train" : X_train, 
        "X_test" : X_test,
        "y_train" : y_train,
        "y_test" : y_test,
        "train_size" : train_size,
        "test_size" : test_size,
        "X_scaled" : X_scaled
    }

    return values_train, scaler

def convers_binary(df_clean, target_col):
    # === 🔽 Conversión binaria personalizada según dataset ===
    dataset_type = detect_dataset(df_clean)

    if target_col in df_clean.columns:
        df_clean[target_col] = df_clean[target_col].astype(str).str.upper().str.strip()

        if dataset_type == "koi":
            df_clean[target_col] = df_clean[target_col].replace({
                "FALSE POSITIVE": 0,
                "CONFIRMED": 0,
                "CANDIDATE": 1
            })

        elif dataset_type == "toi":
            df_clean[target_col] = df_clean[target_col].replace({
                "FP": 0,    # False Positive
                "FA": 0,    # False Alarm
                "KP": 0,    # Known Planet
                "PC": 1,    # Planet Candidate
                "APC": 1    # Ambiguous Planet Candidate
            })

        elif dataset_type == "k2":
            df_clean[target_col] = df_clean[target_col].replace({
                "FALSE POSITIVE": 0,
                "CONFIRMED": 0,
                "CANDIDATE": 1
            })

        # Asegurar que sea numérico binario (0 o 1)
        df_clean[target_col] = pd.to_numeric(df_clean[target_col], errors="coerce").fillna(0).astype(int)

    return df_clean[target_col]
