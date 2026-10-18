"""
Módulo 4: Entrenamiento del Modelo de Lead Scoring (Scikit-Learn - Gradient Boosting)
Entrena un modelo optimizado para predecir P(Venta) en datos tabulares de Call Center.
"""

import sys
import os

# Agrega la carpeta raíz del proyecto al path de Python para evitar errores con 'from src.features'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from src.features import preparar_datos_para_ml

def entrenar_modelo():
    entrada = "data/processed/base_enriquecida_reglas.csv"
    print("Preparando características para el modelo...")
    X, y, preprocessor, df_aptos = preparar_datos_para_ml(entrada)
    
    # Partición de datos en entrenamiento (80%) y prueba (20%) estratificada por el desbalance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Form conjunto de entrenamiento: {X_train.shape}")
    print(f"Form conjunto de prueba: {X_test.shape}")
    
    # Construir modelo de Machine Learning basado en Gradient Boosting (ideal para tablas de Telco)
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    print("\nIniciando entrenamiento del modelo Gradient Boosting...")
    model.fit(X_train, y_train)
    
    # Evaluación en test
    y_pred_prob = model.predict_proba(X_test)[:, 1] # Probabilidad P(Venta = 1)
    auc_score = roc_auc_score(y_test, y_pred_prob)
    print(f"\n[Evaluación Offline] ROC-AUC Score en Test: {auc_score:.4f}")
    
    # Reporte de clasificación detallado
    y_pred = model.predict(X_test)
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred))
    
    # Guardar el pipeline completo (preprocesador + modelo) usando joblib
    os.makedirs("models", exist_ok=True)
    
    # Empaquetamos ambos para que la inferencia sea directa
    pipeline_completo = {
        "preprocessor": preprocessor,
        "model": model
    }
    
    ruta_modelo = "models/lead_scoring_model.pkl"
    joblib.dump(pipeline_completo, ruta_modelo)
    print(f"Modelo y preprocesador guardados exitosamente en: {ruta_modelo}")
    
    return pipeline_completo

if __name__ == "__main__":
    entrenar_modelo()