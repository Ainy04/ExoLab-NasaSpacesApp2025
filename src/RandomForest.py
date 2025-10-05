# ============================================
# 🌲 Entrenamiento optimizado con Random Forest
# ============================================

from sklearn.ensemble import RandomForestClassifier
from typing import Dict
import numpy as np

def train_randomForest(values: Dict):
    """
    🚀 Entrena un modelo Random Forest con hiperparámetros optimizados
    para detección de exoplanetas o problemas con desbalance de clases.

    Parámetros clave:
    - n_estimators: número de árboles en el bosque
    - max_depth: controla la complejidad del modelo
    - min_samples_split / min_samples_leaf: previenen overfitting
    - max_features: fracción de características por árbol (sqrt = recomendada)
    - class_weight='balanced': ajusta el peso de clases automáticamente
    """

    # ===============================
    # 📦 Cargar datos de entrenamiento
    # ===============================
    X_train = values["X_train"]
    y_train = values["y_train"]

    # ===============================
    # ⚖️ Calcular distribución de clases
    # ===============================
    unique, counts = np.unique(y_train, return_counts=True)
    class_dist = dict(zip(unique, counts))

    print(f"📊 Clases detectadas: {class_dist}")
    if len(unique) > 1:
        ratio = counts.min() / counts.max()
        print(f"⚖️ Balance entre clases: {ratio:.2f}")
    else:
        print("⚠️ Solo se detectó una clase. Entrenamiento sin balanceo.")
    
    # ===============================
    # 🎯 Configuración de hiperparámetros
    # ===============================
    params = {
        'n_estimators': 300,           # 🌲 Más árboles para mayor robustez
        'max_depth': 20,               # 🌌 Profundidad controlada para evitar overfitting
        'min_samples_split': 8,        # 🪶 División mínima de nodos
        'min_samples_leaf': 3,         # 🍃 Número mínimo de muestras por hoja
        'max_features': 'sqrt',        # ⚙️ Aleatoriedad saludable por árbol
        'bootstrap': True,             # 🔁 Muestreo con reemplazo (clave en RF)
        'class_weight': 'balanced',    # ⚖️ Ajuste automático por frecuencia de clases
        'random_state': 42,            # 🧭 Reproducibilidad
        'n_jobs': -1                   # 💻 Paralelización total
    }

    # ===============================
    # 🧠 Crear y entrenar el modelo
    # ===============================
    clf = RandomForestClassifier(**params)

    clf.fit(X_train, y_train)
    print(f"✅ Random Forest entrenado con {params['n_estimators']} árboles y profundidad máxima {params['max_depth']}")
    print(f"🌲 Cada árbol ve ~{params['max_features']} de las características disponibles")

    # ===============================
    # 📈 Importancia de características
    # ===============================
    try:
        importances = clf.feature_importances_
        top_idx = np.argsort(importances)[::-1][:5]
        print("\n💡 Características más importantes:")
        for i in top_idx:
            print(f" - {X_train.columns[i]}: {importances[i]:.4f}")
    except Exception as e:
        print(f"⚠️ No se pudieron mostrar importancias: {e}")

    return clf
