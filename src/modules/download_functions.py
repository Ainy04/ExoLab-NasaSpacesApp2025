"""
Funciones de descarga (sin Flask)
Para usar directamente en Streamlit
"""
import os


def get_model_file(model_path):
    """
    Verifica si existe el modelo y retorna la ruta
    
    Args:
        model_path: ruta al archivo del modelo
    
    Returns:
        ruta del archivo si existe, None si no existe
    """
    if os.path.exists(model_path):
        return model_path
    return None


def read_model_bytes(model_path):
    """
    Lee el archivo del modelo como bytes para descarga
    
    Args:
        model_path: ruta al archivo del modelo
    
    Returns:
        bytes del archivo, None si no existe
    """
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return f.read()
    return None