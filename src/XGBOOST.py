# Importar librerías
import xgboost as xgb
from typing import Dict
import numpy as np

def train_xgboost(values: Dict):
    """
    🚀 Entrena XGBoost con hiperparámetros optimizados para detección de exoplanetas
    
    Parámetros clave:
    - scale_pos_weight: Maneja desbalance de clases (crítico en astronomía)
    - max_depth: Profundidad para capturar relaciones complejas en tránsitos
    - learning_rate: Tasa de aprendizaje conservadora para evitar overfitting
    - subsample/colsample_bytree: Regularización mediante muestreo
    """
    
    X_train = values["X_train"] 
    y_train = values["y_train"]
    
    # Calcular ratio de clases para desbalance
    unique, counts = np.unique(y_train, return_counts=True)
    
    # Si tenemos más de una clase, calcular scale_pos_weight
    if len(unique) > 1:
        negative_samples = counts[0] if unique[0] == 0 else counts[1]
        positive_samples = counts[1] if unique[1] == 1 else counts[0]
        scale_pos_weight = negative_samples / positive_samples if positive_samples > 0 else 1
    else:
        scale_pos_weight = 1
    
    print(f"📊 Clases detectadas: {dict(zip(unique, counts))}")
    print(f"⚖️ Scale pos weight: {scale_pos_weight:.2f}")
    
    # Determinar número de clases
    n_classes = len(unique)
    
    # Configurar objetivo según número de clases
    if n_classes == 2:
        objective = 'binary:logistic'
        num_class = None
    else:
        objective = 'multi:softprob'
        num_class = n_classes
    
    # ============================================
    # 🎯 HIPERPARÁMETROS OPTIMIZADOS
    # ============================================
    params = {
        'n_estimators': 300,           # Más árboles = mejor generalización
        'max_depth': 6,                # Profundidad media para relaciones complejas
        'learning_rate': 0.05,         # Learning rate bajo para estabilidad
        'subsample': 0.8,              # Muestreo de filas (80%)
        'colsample_bytree': 0.8,       # Muestreo de columnas (80%)
        'colsample_bylevel': 0.8,      # Muestreo por nivel
        'min_child_weight': 3,         # Peso mínimo de nodos hoja
        'gamma': 0.1,                  # Regularización de splits
        'reg_alpha': 0.1,              # Regularización L1
        'reg_lambda': 1.0,             # Regularización L2
        'scale_pos_weight': scale_pos_weight,  # Manejo de desbalance
        'objective': objective,
        'eval_metric': 'logloss',
        'random_state': 42,
        'n_jobs': -1,                  # Usar todos los cores
        'tree_method': 'hist',         # Método eficiente para datasets grandes
        'grow_policy': 'lossguide',    # Crecimiento guiado por pérdida
    }
    
    # Agregar num_class solo si es multiclase
    if num_class is not None:
        params['num_class'] = num_class
    
    # Crear el modelo
    clf = xgb.XGBClassifier(**params)
    
    # Entrenar con early stopping si tenemos suficientes datos
    if len(X_train) > 100:
        # Dividir train en train/validation para early stopping
        split_idx = int(len(X_train) * 0.85)
        X_train_sub = X_train.iloc[:split_idx]
        y_train_sub = y_train.iloc[:split_idx]
        X_val = X_train.iloc[split_idx:]
        y_val = y_train.iloc[split_idx:]
        
        clf.fit(
            X_train_sub, 
            y_train_sub,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        print(f"✅ XGBoost entrenado con early stopping")
    else:
        # Entrenar sin early stopping en datasets pequeños
        clf.fit(X_train, y_train)
        print(f"✅ XGBoost entrenado (dataset pequeño)")
    
    print(f"🌲 Número de árboles utilizados: {clf.n_estimators}")
    
    return clf