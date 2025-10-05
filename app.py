#librerias
import matplotlib
matplotlib.use('Agg')
import warnings

#archivos locales del proyecto
from src.posTrainFunctions import *
from src.preTrainFunctions import *
from src.RandomForest import train_randomForest
from src.XGBOOST import train_xgboost
from src.Global import *
from src.utils import *

#rutas de Flask
from src.routes.ml_routes import ml_bp
from src.routes.home_routes import home_bp
from src.routes.upload_routes import upload_bp
from src.routes.data_routes import data_bp
from src.routes.predict_routes import predict_bp
from src.routes.download_routes import download_bp

# Registrar rutas
app.register_blueprint(home_bp)
app.register_blueprint(upload_bp)
app.register_blueprint(data_bp)
app.register_blueprint(predict_bp)
app.register_blueprint(download_bp)
app.register_blueprint(ml_bp)

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)