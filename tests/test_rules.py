"""
Pruebas unitarias para el motor de reglas de negocio y SISACT.
"""

import os
import pandas as pd
from src.rules_engine import aplicar_reglas_negocio

def test_aplicacion_reglas_negocio():
    ruta_entrada = "data/raw/base_cruda_mercado.csv"
    ruta_salida = "data/processed/base_enriquecida_reglas.csv"
    
    # Asegurar que existe la base cruda antes de probar
    if os.path.exists(ruta_entrada):
        aplicar_reglas_negocio(ruta_entrada, ruta_salida)
        
        assert os.path.exists(ruta_salida), "El archivo enriquecido de reglas no fue generado."
        
        df = pd.read_csv(ruta_salida)
        assert len(df) > 0, "El DataFrame procesado está vacío."
        assert 'estado_operativo' in df.columns, "Falta la columna 'estado_operativo'."
        assert 'score_sisact' in df.columns, "Falta la columna 'score_sisact'."
        assert 'comision_maxima_soles' in df.columns, "Falta la columna 'comision_maxima_soles'."