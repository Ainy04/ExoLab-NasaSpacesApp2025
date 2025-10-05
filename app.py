"""
app.py - Versión simplificada sin Flask
"""
import matplotlib
matplotlib.use('Agg')
import warnings

# Configuración global
warnings.filterwarnings('ignore')

# Importar funciones locales del proyecto
from src.posTrainFunctions import *
from src.preTrainFunctions import *
from src.RandomForest import train_randomForest
from src.XGBOOST import train_xgboost
from src.Global import *
from src.utils import *

# Importar módulos sin Flask
from src.modules import *

print("ExoLab módulos cargados correctamente")
print("Ejecuta tu dashboard con: streamlit run dashboard.py")