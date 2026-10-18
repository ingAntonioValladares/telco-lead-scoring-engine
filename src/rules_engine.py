"""
Módulo 2: Motor de Reglas de Negocio, Cobertura, Antifraude y SISACT (Fase 2)
Aplica los filtros duros y la evaluación crediticia antes del Lead Scoring de IA.
"""

import pandas as pd
import numpy as np
import os

def aplicar_reglas_negocio(ruta_entrada: str, ruta_salida: str, seed: int = 42):
    """
    Lee la base cruda y aplica los filtros de cobertura, anti-carrusel y SISACT (0-10).
    """
    np.random.seed(seed)
    
    if not os.path.exists(ruta_entrada):
        raise FileNotFoundError(f"No se encontró el archivo en {ruta_entrada}. Ejecuta ingestion.py primero.")
    
    df = pd.read_csv(ruta_entrada)
    print(f"Procesando {len(df)} registros para reglas de negocio y SISACT...")
    
    # 1. Validación de Cobertura Técnica Real (Simulando cruce con planos de red)
    # Algunos distritos pueden arrojar SIN_COBERTURA técnica real
    condiciones_cobertura = [
        df['cobertura_teorica_mercado'] == 'FTTH',
        df['cobertura_teorica_mercado'] == 'HFC'
    ]
    # Asignamos una pequeña probabilidad de que un distrito con red teórica esté saturado (SIN_COBERTURA)
    choices_cobertura = ['FTTH', 'HFC']
    df['cobertura_tecnica_real'] = np.select(condiciones_cobertura, choices_cobertura, default='HFC')
    
    # Simular un 5% de casos sin cobertura real de puertos
    mascara_sc = np.random.rand(len(df)) < 0.05
    df.loc[mascara_sc, 'cobertura_tecnica_real'] = 'SIN_COBERTURA'
    
    # 2. Prevención de Venta Carrusel y Deuda (Filtro Antifraude)
    # Simulamos que un % de la base cae en fraude por carrusel o deuda interna previa
    prob_fraude = np.random.rand(len(df))
    estado_operativo = []
    
    for p in prob_fraude:
        if p < 0.06:
            estado_operativo.append('RECHAZADO_CARRUSEL')
        elif p < 0.10:
            estado_operativo.append('RIESGO_DEUDA_INTERNA')
        else:
            estado_operativo.append('APROBADO_PARA_SCORING')
            
    df['estado_operativo'] = estado_operativo
    
    # 3. Simulación de Evaluación Crediticia SISACT (0 a 10 puntos)
    # Los aprobados para scoring reciben un puntaje aleatorio ponderado por su edad y gasto actual
    scores_sisact = []
    for estado in df['estado_operativo']:
        if estado != 'APROBADO_PARA_SCORING':
            scores_sisact.append(0) # Sin score si está rechazado
        else:
            # Distribución realista de score crediticio (tiende más hacia el rango medio/alto)
            score = np.random.choice([2, 3, 4, 5, 6, 7, 8, 9, 10], p=[0.05, 0.08, 0.12, 0.15, 0.20, 0.18, 0.12, 0.07, 0.03])
            scores_sisact.append(score)
            
    df['score_sisact'] = scores_sisact
    
    # 4. Determinación del Producto Máximo Elegible y Comisión Máxima en Soles (S/.)
    max_producto = []
    comision_maxima = []
    
    for score in df['score_sisact']:
        if score <= 3:
            max_producto.append('MONO_INTERNET')
            comision_maxima.append(15.00) # Comisión baja para productos sencillos
        elif score <= 6:
            max_producto.append('DUO_INTERNET_TV')
            comision_maxima.append(35.00) # Comisión intermedia
        else:
            max_producto.append('TRIO_COMPLETO')
            comision_maxima.append(61.25) # Comisión alta para paquetes premium
            
    df['max_producto_elegible'] = max_producto
    df['comision_maxima_soles'] = comision_maxima
    
    # Si no tiene cobertura o está rechazado por carrusel, la comisión potencial se anula
    df.loc[df['cobertura_tecnica_real'] == 'SIN_COBERTURA', 'comision_maxima_soles'] = 0.0
    df.loc[df['estado_operativo'] != 'APROBADO_PARA_SCORING', 'comision_maxima_soles'] = 0.0
    
    # Guardar resultado procesado
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    df.to_csv(ruta_salida, index=False)
    
    print(f"¡Reglas de negocio y SISACT aplicadas con éxito!")
    print(f"Archivo guardado en: {ruta_salida}")
    print("\nResumen de estados operativos generados:")
    print(df['estado_operativo'].value_counts())
    print("\nResumen de productos máximos elegibles:")
    print(df['max_producto_elegible'].value_counts())

if __name__ == "__main__":
    entrada = "data/raw/base_cruda_mercado.csv"
    salida = "data/processed/base_enriquecida_reglas.csv"
    aplicar_reglas_negocio(entrada, salida)