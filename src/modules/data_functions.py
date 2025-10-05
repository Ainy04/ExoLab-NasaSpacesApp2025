"""
Funciones de análisis de datos (sin Flask)
Para usar directamente en Streamlit
"""
import pandas as pd
import numpy as np
from src.utils import detect_dataset, get_available_columns
from src.Global import TARGET_COLUMNS


def get_data_info(df):
    """Obtiene información básica del dataset"""
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
    
    return info


def get_dataset_config(df):
    """Devuelve la configuración de columnas disponibles para el dataset"""
    dataset_type = detect_dataset(df)
    
    if not dataset_type:
        return {"error": "Tipo de dataset no reconocido"}
    
    available_cols = get_available_columns(df, dataset_type)
    
    return {
        "dataset_type": dataset_type,
        "available_columns": available_cols,
        "all_columns": list(df.columns)
    }


def get_statistics(df):
    """Estadísticas descriptivas de columnas numéricas"""
    numeric_df = df.select_dtypes(include=[np.number])
    stats = numeric_df.describe().to_dict()
    return stats


def get_distribution(df, column):
    """Distribución de una columna específica"""
    if column not in df.columns:
        return {"error": f"Columna '{column}' no existe"}
    
    column_data = df[column].dropna()
    
    if len(column_data) == 0:
        return {"error": f"La columna '{column}' no tiene valores válidos"}
    
    if pd.api.types.is_numeric_dtype(column_data):
        hist, bins = np.histogram(column_data, bins=30)
        return {
            "type": "numeric",
            "histogram": hist.tolist(),
            "bins": bins.tolist(),
            "mean": float(column_data.mean()),
            "median": float(column_data.median()),
            "std": float(column_data.std()),
            "valid_count": len(column_data)
        }
    else:
        counts = column_data.value_counts().to_dict()
        return {
            "type": "categorical",
            "counts": counts,
            "valid_count": len(column_data)
        }


def get_correlation(df, columns=None):
    """Matriz de correlación"""
    numeric_df = df.select_dtypes(include=[np.number])
    
    if columns:
        numeric_df = numeric_df[columns]
    
    corr_matrix = numeric_df.corr().to_dict()
    return corr_matrix


def get_scatter_data(df, x_col, y_col, color_col=None):
    """Datos para scatter plot"""
    if not x_col or not y_col:
        return {"error": "Parámetros 'x_column' y 'y_column' requeridos"}
    
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
    
    return result


def get_sample(df, n=10):
    """Muestra de datos"""
    sample = df.head(n).to_dict('records')
    return sample


def remove_outliers_iqr(data, column):
    """Remover outliers usando método IQR"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 2.0 * IQR
    upper = Q3 + 2.0 * IQR
    return data[(data[column] >= lower) & (data[column] <= upper)]


def get_statistics_no_outliers(df):
    """Estadísticas descriptivas sin outliers usando IQR"""
    numeric_df = df.select_dtypes(include=[np.number])
    
    def remove_outliers_iqr_series(series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return series[(series >= lower_bound) & (series <= upper_bound)]
    
    stats_dict = {}
    for col in numeric_df.columns:
        try:
            clean_data = remove_outliers_iqr_series(numeric_df[col].dropna())
            if len(clean_data) > 0:
                stats_dict[col] = {
                    'count': float(len(clean_data)),
                    'mean': float(clean_data.mean()),
                    'std': float(clean_data.std()),
                    'min': float(clean_data.min()),
                    '25%': float(clean_data.quantile(0.25)),
                    '50%': float(clean_data.quantile(0.50)),
                    '75%': float(clean_data.quantile(0.75)),
                    'max': float(clean_data.max())
                }
        except Exception as e:
            print(f"Error processing {col}: {e}")
            continue
    
    return stats_dict


def scatter_no_outliers(df, x_col, y_col, color_col=None):
    """Scatter plot con outliers removidos por IQR"""
    if not x_col or not y_col:
        return {'error': 'x_column and y_column required'}
    
    cols_to_use = [x_col, y_col]
    if color_col and color_col in df.columns:
        cols_to_use.append(color_col)
    
    df_clean = df[cols_to_use].dropna()
    original_count = len(df_clean)
    df_clean = remove_outliers_iqr(df_clean, x_col)
    df_clean = remove_outliers_iqr(df_clean, y_col)
    cleaned_count = len(df_clean)
    
    result = {
        'x': df_clean[x_col].tolist(),
        'y': df_clean[y_col].tolist(),
        'original_count': original_count,
        'cleaned_count': cleaned_count,
        'outliers_removed': original_count - cleaned_count
    }
    
    if color_col and color_col in df.columns:
        result['color'] = df_clean[color_col].tolist()
    
    return result