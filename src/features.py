"""
Módulo 3: Ingeniería de Características (Feature Engineering)
Prepara y transforma los datos del motor de reglas para alimentar la Red Neuronal de TensorFlow.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

def preparar_datos_para_ml(ruta_entrada: str, seed: int = 42):
    """
    Filtra los leads aptos, genera una variable objetivo (target) sintética de venta 
    y construye el pipeline de transformación numérica y categórica.
    """
    np.random.seed(seed)
    
    if not os.path.exists(ruta_entrada):
        raise FileNotFoundError(f"No se encontró {ruta_entrada}. Ejecuta rules_engine.py primero.")
        
    df = pd.read_csv(ruta_entrada)
    
    # 1. Filtrar solo leads aptos para marcación (Aprobados y con cobertura real)
    df_aptos = df[
        (df['estado_operativo'] == 'APROBADO_PARA_SCORING') & 
        (df['cobertura_tecnica_real'].isin(['FTTH', 'HFC']))
    ].copy()
    
    print(f"Leads aptos para entrenamiento y scoring tras filtros: {len(df_aptos)} registros.")
    
    # 2. Simulación de la Variable Objetivo (Target: venta = 1 o 0)
    # Simulamos la propensión a compra basada lógicamente en el score SISACT y el gasto actual
    # A mayor score SISACT y mayor ahorro percibido, mayor probabilidad de compra
    prob_base = 0.02 + (df_aptos['score_sisact'] / 20.0) + (df_aptos['gasto_actual_soles'] / 1000.0)
    # Añadir un factor aleatorio controlado
    ruido = np.random.normal(0, 0.05, size=len(df_aptos))
    propension_final = prob_base + ruido
    
    # Convertimos a binario (1 = Venta exitosa, 0 = No venta) con un umbral para reflejar ~15-20% de conversión en leads calificados
    umbral_conversion = np.percentile(propension_final, 80) # Top 20% con mayor propensión compran
    df_aptos['target_venta'] = (propension_final >= umbral_conversion).astype(int)
    
    print(f"Tasa de conversión simulada en leads aptos: {df_aptos['target_venta'].mean()*100:.2f}%")
    
    # 3. Definición de Features (Características de entrada)
    features_numericas = ['edad', 'gasto_actual_soles', 'score_sisact', 'comision_maxima_soles']
    features_categoricas = ['departamento', 'cobertura_tecnica_real', 'operador_actual', 'max_producto_elegible']
    
    X = df_aptos[features_numericas + features_categoricas]
    y = df_aptos['target_venta']
    
    # 4. Pipeline de Preprocesamiento de Scikit-Learn
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), features_numericas),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), features_categoricas)
        ]
    )
    
    X_procesado = preprocessor.fit_transform(X)
    
    return X_procesado, y, preprocessor, df_aptos

if __name__ == "__main__":
    entrada = "data/processed/base_enriquecida_reglas.csv"
    X_proc, y, prep, df_fin = preparar_datos_para_ml(entrada)
    print(f"Dimensiones de la matriz de features procesadas (X): {X_proc.shape}")
    print(f"Dimensiones del vector target (y): {y.shape}")