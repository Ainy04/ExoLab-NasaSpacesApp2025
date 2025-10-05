import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from src.Global import TARGET_CANDIDATES, USEFUL_COLUMNS

def detect_target_column(df):
    """
    Detecta la columna objetivo en un dataframe usando las palabras clave.
    """
    df_cols = [c.lower().strip() for c in df.columns]
    for candidate in TARGET_CANDIDATES:
        if candidate.lower() in df_cols:
            print(f"Columna objetivo detectada: {candidate.lower()}")
            return candidate.lower()
    raise ValueError(f"No se encontró ninguna columna objetivo entre {TARGET_CANDIDATES}. Columnas: {list(df.columns)}")

def clean_dataset(df: pd.DataFrame):
    """
    Limpieza automática del dataset:
    - elimina duplicados
    - rellena nulos
    - normaliza target
    - mantiene solo columnas útiles
    """
    df = df.copy()
    print("=== 🧹 LIMPIEZA AUTOMÁTICA ===")
    df.columns = df.columns.str.strip().str.lower()
    # detectar target
    target = detect_target_column(df)
    # mantener solo columnas útiles + target si existe
    cols_to_keep = [c for c in USEFUL_COLUMNS if c in df.columns]
    if target not in cols_to_keep:
        cols_to_keep.append(target)
    df = df[cols_to_keep]
    # reemplazar inf por NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # conversión numérica
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            pass
    # imputación de nulos
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            df[col] = df[col].fillna(df[col].mean())
        else:
            mode_value = df[col].mode()[0] if not df[col].mode().empty else "UNKNOWN"
            df[col] = df[col].fillna(mode_value)
    df = df.drop_duplicates()
    # normalizar target
    df[target] = df[target].astype(str).str.upper().str.strip()
    df[target] = df[target].replace({
        "CONFIRM": "CONFIRMED",
        "FALSE_POSITIVE": "FALSE POSITIVE",
        "FALSEPOSITIVE": "FALSE POSITIVE",
        "FP": "FALSE POSITIVE"
    })
    print(f"Columnas finales usadas: {list(df.columns)}")
    print("=== LIMPIEZA COMPLETA ===")
    return df, target

def preprocess_df(df, target):
    """
    Preprocesa el dataset:
    - codifica la columna target
    - selecciona solo columnas numéricas
    """
    df = df.copy()
    le = LabelEncoder()
    df[target] = le.fit_transform(df[target].astype(str))
    numeric = df.select_dtypes(include=[np.number]).copy()
    if target not in numeric.columns:
        numeric[target] = df[target]
    numeric = numeric.fillna(numeric.mean())
    return numeric, le
