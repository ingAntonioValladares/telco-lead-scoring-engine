"""
Módulo 1: Ingesta y Generación Sintética de Base Cruda (Fase 1)
Simula la adquisición de bases de datos masivas del mercado para el Call Center de Telco.
"""

import pandas as pd
import numpy as np
import os

def generar_base_cruda(num_registros: int = 50000, seed: int = 42) -> pd.DataFrame:
    """
    Genera un DataFrame sintético con la base fría de leads comprados a ciegas.
    """
    np.random.seed(seed)
    
    # Simulación de DNI
    dnis = [str(np.random.randint(10000000, 99999999)) for _ in range(num_registros)]
    
    # Ubigeos y distritos representativos (incluyendo zona norte como Chiclayo/Lambayeque y Lima)
    distritos_data = [
        ("Lima", "Miraflores", "FTTH"),
        ("Lima", "San Isidro", "FTTH"),
        ("Lima", "Surco", "HFC"),
        ("Lima", "San Juan de Lurigancho", "HFC"),
        ("Lambayeque", "Chiclayo", "FTTH"),
        ("Lambayeque", "José Leonardo Ortiz", "HFC"),
        ("La Libertad", "Trujillo", "FTTH"),
        ("Piura", "Piura", "HFC")
    ]
    
    indices_distrito = np.random.choice(len(distritos_data), size=num_registros)
    departamentos = [distritos_data[i][0] for i in indices_distrito]
    distritos = [distritos_data[i][1] for i in indices_distrito]
    cobertura_teorica_base = [distritos_data[i][2] for i in indices_distrito]
    
    # Operadores actuales en el mercado
    operadores = ["Movistar", "WIN", "Entel", "WOW", "Claro (Antiguo/Inactivo)", "Ninguno"]
    probs_operador = [0.45, 0.20, 0.15, 0.10, 0.05, 0.05]
    operador_actual = np.random.choice(operadores, size=num_registros, p=probs_operador)
    
    # Gasto mensual actual estimado en soles (S/.)
    gasto_actual = np.random.choice([69.90, 89.90, 109.90, 129.90, 159.90], size=num_registros)
    
    # Edades de los prospectos
    edades = np.random.randint(18, 70, size=num_registros)
    
    df_crudo = pd.DataFrame({
        "dni": dnis,
        "edad": edades,
        "departamento": departamentos,
        "distrito": distritos,
        "cobertura_teorica_mercado": cobertura_teorica_base,
        "operador_actual": operador_actual,
        "gasto_actual_soles": gasto_actual
    })
    
    return df_crudo

if __name__ == "__main__":
    print("Generando base cruda de prospectos (Fase 1)...")
    df = generar_base_cruda(num_registros=50000)
    
    # Asegurar que exista la carpeta data/raw
    os.makedirs("data/raw", exist_ok=True)
    ruta_salida = "data/raw/base_cruda_mercado.csv"
    df.to_csv(ruta_salida, index=False)
    
    print(f"¡Base cruda generada exitosamente!")
    print(f"Ruta guardada: {ruta_salida}")
    print(f"Total de registros: {len(df)}")
    print(df.head())