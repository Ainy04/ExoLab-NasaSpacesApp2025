from flask_cors import CORS
import pandas as pd
import numpy as np
from flask import jsonify
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import plot_to_base64

from typing import Dict

def calculate_cv_scores(clf, values: Dict ):

    X = values["X"]
    y = values["y"]

    cv_accuracy = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    cv_precision = cross_val_score(clf, X, y, cv=5, scoring="precision_weighted")
    cv_recall = cross_val_score(clf, X, y, cv=5, scoring="recall_weighted")  
    cv_f1 = cross_val_score(clf, X, y, cv=5, scoring="f1_weighted")  

    cv_results = {
        "cv_accuracy_mean": float(cv_accuracy.mean()),
        "cv_precision_mean": float(cv_precision.mean()),
        "cv_recall_mean": float(cv_recall.mean()),
        "cv_f1_mean": float(cv_f1.mean())
    }

    return cv_results

def predict_and_metrics(clf, values: Dict):
    
    X_test = values["X_test"]
    y_test = values["y_test"]

    y_pred = clf.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average='weighted', zero_division=0)), 
        "recall": float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),  
        "f1": float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),  
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist() 
    }

    return y_pred, metrics

def generate_ml_plots(values: Dict, y_pred, clf, label_encoder):
    """Generar visualizaciones de ML"""
    y_train = values["y_train"]
    y_test = values["y_test"]

    # Mapeo de valores binarios a etiquetas legibles
    label_map = {0: "Not Candidate", 1: "Candidate"}
    
    plots = {}
    
    try:
        # Distribución del target
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        train_counts = pd.Series(y_train).value_counts().sort_index()
        test_counts = pd.Series(y_test).value_counts().sort_index()
        
        # Mapear valores a etiquetas legibles
        train_labels = [label_map.get(c, str(c)) for c in train_counts.index]
        test_labels = [label_map.get(c, str(c)) for c in test_counts.index]
        
        ax1.bar(train_labels, train_counts.values, color=['#2E86AB', '#F77F00'])
        ax1.set_title('Distribución Train', fontweight='bold')
        ax1.set_ylabel('Cantidad')
        ax1.grid(axis='y', alpha=0.3)
        
        ax2.bar(test_labels, test_counts.values, color=['#2E86AB', '#F77F00'])
        ax2.set_title('Distribución Test', fontweight='bold')
        ax2.set_ylabel('Cantidad')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plots['target_distribution'] = plot_to_base64(fig)
    except Exception as e:
        print(f"Error en plot distribución: {e}")
    
    try:
        # Confusion Matrix
        fig, ax = plt.subplots(figsize=(8, 7))
        cm = confusion_matrix(y_test, y_pred)
        
        # Etiquetas legibles para la matriz de confusión
        tick_labels = [label_map.get(i, str(i)) for i in range(len(cm))]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', ax=ax,
                    cbar_kws={'label': 'Cantidad'}, linewidths=1, linecolor='gray',
                    xticklabels=tick_labels, yticklabels=tick_labels)
        ax.set_xlabel('Predicción', fontsize=12, fontweight='bold')
        ax.set_ylabel('Real', fontsize=12, fontweight='bold')
        ax.set_title('Matriz de Confusión', fontsize=14, fontweight='bold')
        
        plots['confusion_matrix'] = plot_to_base64(fig)
    except Exception as e:
        print(f"Error en confusion matrix: {e}")
    
    try:
        # Feature Importance
        fig, ax = plt.subplots(figsize=(10, 8))
        importances = pd.DataFrame({
            'feature': values["X_train"].columns,
            'importance': clf.feature_importances_
        }).sort_values('importance', ascending=False).head(20)
        
        colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(importances)))
        ax.barh(importances['feature'], importances['importance'], color=colors)
        ax.set_xlabel('Importancia', fontsize=12, fontweight='bold')
        ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
        ax.set_title('Top 20 Features más Importantes', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        plots['feature_importance'] = plot_to_base64(fig)
    except Exception as e:
        print(f"Error en feature importance: {e}")
    
    return plots