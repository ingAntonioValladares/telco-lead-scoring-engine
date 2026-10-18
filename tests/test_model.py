"""
Pruebas unitarias para la persistencia y carga del modelo de Gradient Boosting.
"""

import os
import joblib

def test_modelo_guardado_y_estructura():
    ruta_modelo = "models/lead_scoring_model.pkl"
    
    if os.path.exists(ruta_modelo):
        pipeline = joblib.load(ruta_modelo)
        
        # Verificar que el pipeline empaquete el preprocesador y el estimador
        assert isinstance(pipeline, dict), "El modelo guardado no es un diccionario de pipeline."
        assert "preprocessor" in pipeline, "Falta el transformador de datos en el pipeline."
        assert "model" in pipeline, "Falta el estimador de Machine Learning en el pipeline."