import streamlit as st
import requests
from datetime import datetime

# Configuration
API_URL = "http://127.0.0.1:8000"

# App title
st.set_page_config(page_title="Predicción de Demanda E-Commerce", layout="wide")
st.title("📊 Predicción de Demanda E-Commerce")
st.write("""
Sistema de predicción de demanda basado en modelos de Machine Learning.
Utiliza la API FastAPI para realizar predicciones en tiempo real.
""")

# Sidebar
st.sidebar.header("📋 Información del Sistema")

# Check backend health
try:
    health_response = requests.get(f"{API_URL}/api/health", timeout=5)
    health_data = health_response.json()
    
    st.sidebar.subheader("✅ Estado del Backend")
    st.sidebar.success("Backend conectado correctamente")
    
    # Display model info from health
    st.sidebar.subheader("🤖 Modelo Cargado")
    st.sidebar.write(f"Modelo: {health_data.get('model', 'Desconocido')}")
    st.sidebar.write(f"Encoders: {health_data.get('encoders_loaded', 0)}")
    st.sidebar.write(f"Scaler: {'✅' if health_data.get('scaler_loaded') else '❌'}")
    
    # Try to get model-info and display organized info
    try:
        model_info_response = requests.get(f"{API_URL}/api/model-info", timeout=5)
        model_info = model_info_response.json()
        st.sidebar.subheader("📈 Detalles del Modelo")
        
        # Organize model info
        st.sidebar.write(f"Modelo: {model_info.get('model', 'N/A')}")
        st.sidebar.write(f"Versión: {model_info.get('version', 'N/A')}")
        
        # Metrics section
        if 'metrics' in model_info:
            metrics = model_info['metrics']
            for metric in ['rmse', 'mae', 'mape', 'r2']:
                st.sidebar.write(f"{metric.upper()}: {metrics.get(metric, 'N/A')}")
        
    except Exception as e:
        st.sidebar.warning("No se pudo obtener detalles completos del modelo")
        
except requests.exceptions.ConnectionError:
    st.error("❌ No se puede conectar al backend FastAPI. Asegúrese de que esté ejecutándose en http://127.0.0.1:8000")
    st.stop()
except requests.exceptions.Timeout:
    st.error("⏱️ Timeout al conectar con el backend")
    st.stop()
except Exception as e:
    st.error(f"❌ Error al verificar estado del backend: {str(e)}")
    st.stop()

# Prediction form
st.header("🔍 Realizar Predicción")
st.write("Ingrese los datos para realizar la predicción de demanda:")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        stock_code = st.text_input("StockCode", value="1001")
        unit_price = st.number_input("UnitPrice", min_value=0.0, step=0.01, value=19.99)
        customer_id = st.text_input("CustomerID", value="12345")
        country = st.text_input("Country", value="1")
    
    with col2:
        year = st.selectbox("Año", options=[2010, 2011], index=1)
        month = st.selectbox("Mes", options=list(range(1, 13)), index=6)
        day = st.number_input("Día", min_value=1, max_value=31, value=9)
        hour = st.number_input("Hora", min_value=0, max_value=23, value=14)
    
    with col3:
        day_of_week = st.selectbox("DíaSemana", options=list(range(0, 7)), index=3)
        week_of_year = st.number_input("SemanaAño", min_value=1, max_value=53, value=28)
        quarter = st.selectbox("Trimestre", options=[1, 2, 3, 4], index=2)
        is_weekend = st.checkbox("EsFinDeSemana", value=False)
    
    month_name = st.selectbox("MesNombre", options=["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"], index=6)
    
    submit_button = st.form_submit_button(label="Realizar Predicción")

# Handle form submission
if submit_button:
    # Prepare features
    features = {
        "StockCode": stock_code,
        "UnitPrice": unit_price,
        "CustomerID": customer_id,
        "Country": country,
        "Año": year,
        "Mes": month,
        "Día": day,
        "Hora": hour,
        "DíaSemana": day_of_week,
        "SemanaAño": week_of_year,
        "Trimestre": quarter,
        "EsFinDeSemana": 1 if is_weekend else 0,
        "MesNombre": month_name
    }
    
    # Make prediction request
    with st.spinner("Realizando predicción..."):
        try:
            response = requests.post(
                f"{API_URL}/api/predict",
                json={"features": features},
                timeout=10
            )
            response.raise_for_status()
            
            prediction_data = response.json()
            
            # Display results
            st.subheader("✅ Resultado de la Predicción")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Predicción", f"{prediction_data.get('prediction', 0):.2f}")
            
            with col2:
                st.metric("Modelo", prediction_data.get('model', 'N/A'))
            
            with col3:
                st.metric("Tiempo", f"{prediction_data.get('processing_time_ms', 0):.2f} ms")
            
            st.write(f"Timestamp: {prediction_data.get('timestamp')}")
            st.write(f"Status: {prediction_data.get('status')}")
            
        except requests.exceptions.ConnectionError:
            st.error("❌ No se puede conectar al backend para realizar la predicción")
        except requests.exceptions.Timeout:
            st.error("⏱️ Timeout al esperar la predicción")
        except requests.exceptions.HTTPError as e:
            try:
                error_detail = response.json().get('detail', str(e))
            except:
                error_detail = str(e)
            st.error(f"❌ Error en la API: {error_detail}")
        except Exception as e:
            st.error(f"❌ Error inesperado: {str(e)}")
