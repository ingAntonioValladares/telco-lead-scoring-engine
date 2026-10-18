"""
Módulo 5: Inferencia y Cálculo del Valor Esperado (EV)
Calcula P(Venta), el Valor Esperado de Comisión y prioriza la base de leads diaria.
"""

import sys
import os

# Agrega la carpeta raíz al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import joblib

def ejecutar_inferencia_diaria(ruta_entrada: str, ruta_salida: str):
    if not os.path.exists(ruta_entrada):
        raise FileNotFoundError(f"No se encontró {ruta_entrada}. Ejecuta rules_engine.py primero.")
        
    df = pd.read_csv(ruta_entrada)
    
    # 1. Cargar el pipeline entrenado (preprocesador + modelo)
    ruta_modelo = "models/lead_scoring_model.pkl"
    if not os.path.exists(ruta_modelo):
        raise FileNotFoundError(f"No se encontró el modelo en {ruta_modelo}. Ejecuta model.py primero.")
        
    pipeline = joblib.load(ruta_modelo)
    preprocessor = pipeline["preprocessor"]
    model = pipeline["model"]
    
    # 2. Filtrar leads aptos para marcación
    df_aptos = df[
        (df['estado_operativo'] == 'APROBADO_PARA_SCORING') & 
        (df['cobertura_tecnica_real'].isin(['FTTH', 'HFC']))
    ].copy()
    
    # 3. Preparar features para inferencia
    features_numericas = ['edad', 'gasto_actual_soles', 'score_sisact', 'comision_maxima_soles']
    features_categoricas = ['departamento', 'cobertura_tecnica_real', 'operador_actual', 'max_producto_elegible']
    
    X_Inferencia = df_aptos[features_numericas + features_categoricas]
    
    # Transformar usando el preprocesador ajustado
    X_proc = preprocessor.transform(X_Inferencia)
    
    # 4. Predecir Probabilidad de Venta P(Venta)
    df_aptos['probabilidad_venta'] = model.predict_proba(X_proc)[:, 1]
    
    # 5. Calcular el Valor Esperado (Expected Value - EV) en Soles
    # EV = P(Venta) * Comisión Máxima Permitida por SISACT
    df_aptos['valor_esperado_soles'] = df_aptos['probabilidad_venta'] * df_aptos['comision_maxima_soles']
    
    # 6. Ordenar la base de mayor a menor EV (Priorización inteligente para el Call Center)
    df_priorizado = df_aptos.sort_values(by='valor_esperado_soles', ascending=False).reset_index(drop=True)
    
    # Incluir los rechazados al final de la base para trazabilidad completa
    df_rechazados = df[df['estado_operativo'] != 'APROBADO_PARA_SCORING'].copy()
    df_rechazados['probabilidad_venta'] = 0.0
    df_rechazados['valor_esperado_soles'] = 0.0
    
    df_final_campaña = pd.concat([df_priorizado, df_rechazados], ignore_index=True)
    
    # Guardar la campaña optimizada final
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    df_final_campaña.to_csv(ruta_salida, index=False)
    
    print(f"¡Inferencia completada con éxito!")
    print(f"Campaña priorizada guardada en: {ruta_salida}")
    print(f"\nTop 5 Leads con mayor Valor Esperado (EV) para el Call Center:")
    print(df_priorizado[['dni', 'distrito', 'score_sisact', 'max_producto_elegible', 'probabilidad_venta', 'valor_esperado_soles']].head())

if __name__ == "__main__":
    entrada = "data/processed/base_enriquecida_reglas.csv"
    salida = "data/processed/base_campaña_priorizada.csv"
    ejecutar_inferencia_diaria(entrada, salida)