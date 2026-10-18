# Telco Lead Scoring & Expected Value (EV) Optimization Engine

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn%20%7C%20Gradient%20Boosting-orange.svg)](https://scikit-learn.org/)
[![MLOps Standards](https://img.shields.io/badge/Architecture-AWS%20MLOps%20Aligned-green.svg)](https://aws.amazon.com/)
[![Dashboard](https://img.shields.io/badge/Frontend-Streamlit-red.svg)](https://streamlit.io/)

Un motor end-to-end de **Lead Scoring y Optimización de Valor Esperado ($EV$)** diseñado para la industria de Telecomunicaciones (Telco Call Centers). Combina un **Motor de Reglas Deterministas de Negocio** (Cobertura FTTH/HFC, Prevención Antifraude de Venta Carrusel y Credit Scoring SISACT) con un **Modelo Predictivo de Machine Learning (Gradient Boosting)** para priorizar dinámicamente las listas de llamadas diarias, maximizando la tasa de conversión y el retorno por comisión.

---

## Problema de Negocio

En los Call Centers masivos de Telecomunicaciones, la adquisición de leads fríos de mercado genera altas ineficiencias operativas:

1. **Llamadas inútiles:** Asesores contactan a clientes sin cobertura técnica real (FTTH/HFC) o bloqueados por deudas internas.
2. **Desperdicio de comisión:** Se priorizan prospectos de baja probabilidad de conversión o con elegibilidad crediticia para productos de bajo valor marginal.
3. **Alto Costo por Adquisición (CPA):** Marcación aleatoria o basada en reglas estáticas simples que ignoran el Valor Esperado ($EV = P(\text{Venta}) \times \text{Comisión}$).

---

## Arquitectura de Datos en Dos Fases & Solución Propuesta

El sistema implementa un flujo de datos modular y escalable dividido en dos fases operativas:

```text
[Fase 1: Base Cruda de Mercado]
            │
            ▼
[Fase 2: Motor de Reglas & Sistema Web del Call Center]
   ├── 1. Filtro de Cobertura (FTTH / HFC / Sin Cobertura)
   ├── 2. Prevención de Venta Carrusel / Antifraude
   └── 3. Evaluación Crediticia SISACT (0-10) ➔ Asignación de Producto Elegible
            │
            ▼
[Machine Learning Core (Gradient Boosting)]
   └── Predicción de Propensión a Compra P(Venta)
            │
            ▼
[Motor de Valor Esperado (EV)]
   └── Cálculo: EV = P(Venta) * Comisión Máxima Permitida (S/.)
            │
            ▼
[Dashboard Interactivo en Streamlit & Descarga de Campaña Priorizada]
```
