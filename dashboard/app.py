"""
Dashboard Operativo y Comercial en Streamlit
Simula la interfaz web para el Supervisor del Call Center, permitiendo 
visualizar métricas de campaña, filtros inteligentes y descarga de leads priorizados por EV.
"""

import streamlit as st
import pandas as pd
import os

# Configuración de la página de Streamlit
st.set_page_config(
    page_title="Telco Lead Scoring & EV Engine",
    page_icon="📞",
    layout="wide"
)

# Título y Descripción Ejecutiva
st.title("📞 Telco Call Center: Lead Scoring & Expected Value Optimization Engine")
st.markdown("""
*Sistema de Inteligencia Artificial y Reglas de Negocio (AWS MLOps Aligned)* para la priorización inteligente 
de bases de llamadas, incorporando filtros de cobertura, prevención de **Venta Carrusel** y evaluación crediticia **SISACT (0-10)**.
""")

# Ruta de la base priorizada
ruta_base = "data/processed/base_campaña_priorizada.csv"

if not os.path.exists(ruta_base):
    st.error("⚠️ No se encontró la base de campaña priorizada. Ejecuta primero todo el pipeline en la terminal (`inference.py`).")
else:
    df_campaña = pd.read_csv(ruta_base)
    
    # --- SIDEBAR: Filtros de Control para el Supervisor ---
    st.sidebar.header("🎛️ Filtros de Campaña")
    
    # Filtro por Estado Operativo
    estados_disponibles = df_campaña['estado_operativo'].unique()
    estado_seleccionado = st.sidebar.selectbox("Estado Operativo del Lead", options=["TODOS"] + list(estados_disponibles))
    
    # Filtro por Producto Máximo Elegible
    productos_disponibles = df_campaña['max_producto_elegible'].dropna().unique()
    producto_seleccionado = st.sidebar.selectbox("Producto Máximo Elegible (SISACT)", options=["TODOS"] + list(productos_disponibles))
    
    # Aplicar filtros en el DataFrame
    df_filtrado = df_campaña.copy()
    if estado_seleccionado != "TODOS":
        df_filtrado = df_filtrado[df_filtrado['estado_operativo'] == estado_seleccionado]
    if producto_seleccionado != "TODOS":
        df_filtrado = df_filtrado[df_filtrado['max_producto_elegible'] == producto_seleccionado]
        
    # --- MÉTRICAS PRINCIPALES (KPIs) ---
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    total_leads = len(df_filtrado)
    aptos_llamada = len(df_filtrado[df_filtrado['estado_operativo'] == 'APROBADO_PARA_SCORING'])
    ev_total_campaña = df_filtrado['valor_esperado_soles'].sum()
    comision_promedio = df_filtrado[df_filtrado['estado_operativo'] == 'APROBADO_PARA_SCORING']['comision_maxima_soles'].mean()
    
    col1.metric("📊 Total Leads en Vista", f"{total_leads:,}")
    col2.metric("🎯 Leads Aptos para Llamada", f"{aptos_llamada:,}")
    col3.metric("💰 Valor Esperado Total (EV)", f"S/. {ev_total_campaña:,.2f}")
    col4.metric("📈 Comisión Prom. Potencial", f"S/. {comision_promedio:.2f}" if not pd.isna(comision_promedio) else "S/. 0.00")
    st.markdown("---")
    
    # --- TABLA DE DATOS OPTIMIZADA ---
    st.subheader("📋 Base de Prospectos Priorizada por Valor Esperado ($EV$)")
    st.markdown("Los leads están ordenados automáticamente maximizando la probabilidad de venta y la comisión autorizada por SISACT.")
    
    # Columnas clave a mostrar
    columnas_mostrar = [
        'dni', 'edad', 'departamento', 'distrito', 'cobertura_tecnica_real', 
        'operador_actual', 'estado_operativo', 'score_sisact', 
        'max_producto_elegible', 'probabilidad_venta', 'valor_esperado_soles'
    ]
    
    st.dataframe(df_filtrado[columnas_mostrar], use_container_width=True)
    
    # --- BOTÓN DE DESCARGA PARA LA CAMPAÑA DIARIA ---
    st.markdown("---")
    st.subheader("📥 Exportar Asignación para el Marker / Asesores")
    
    csv_export = df_filtrado.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Descargar Base de Campaña Optimizada (.csv)",
        data=csv_export,
        file_name="base_campaña_optimizada_call_center.csv",
        mime="text/csv"
    )