import base64
import matplotlib.pyplot as plt
import io
from src.Global import DATASET_CONFIG
def detect_dataset(df):
    if 'koi_pdisposition' in df.columns:
        return "koi"
    elif 'tfopwg_disp' in df.columns:
        return "toi"
    elif 'disposition' in df.columns:
        return "k2"
    return None

def plot_to_base64(fig):
    """Convertir plot a base64"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64

def get_available_columns(df, dataset_type):
    """Detecta qué columnas están disponibles en el dataset"""
    if dataset_type not in DATASET_CONFIG:
        return {}
    
    config = DATASET_CONFIG[dataset_type]
    available = {}
    
    for key, col_name in config.items():
        if col_name in df.columns:
            available[key] = col_name
    
    return available