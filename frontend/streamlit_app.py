import streamlit as st
import requests

# Configuration
API_URL = "https://ecommerce-9u1d.onrender.com"

# App title
st.set_page_config(page_title="Predicción de Demanda E-Commerce", layout="wide")
st.title("📊 Predicción de Demanda E-Commerce")
st.write("""
Sistema de predicción de demanda basado en modelos de Machine Learning.
Utiliza la API FastAPI para realizar predicciones en tiempo real.
""")

from app_links import render_app_navigation
render_app_navigation("prediccion")

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
            st.sidebar.metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
            st.sidebar.metric("MAE", f"{metrics.get('mae', 0):.4f}")
            st.sidebar.metric("MAPE", f"{metrics.get('mape', 0):.2f}")
            st.sidebar.metric("R²", f"{metrics.get('r2', 0):.4f}")
        
    except Exception as e:
        st.sidebar.warning("No se pudo obtener detalles completos del modelo")
        
except requests.exceptions.ConnectionError:
    st.error("❌ No se puede conectar al backend FastAPI. Asegúrese de que esté ejecutándose en https://ecommerce-9u1d.onrender.com")
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
        stock_code = st.text_input("StockCode", value="85123A")
        unit_price = st.number_input("UnitPrice", min_value=0.0, step=0.01, value=19.99)
        customer_id = st.text_input("CustomerID", value="17850")
        country = st.selectbox(
            "Country",
            [
                "Australia",
                "Austria",
                "Bahrain",
                "Belgium",
                "Brazil",
                "Canada",
                "Channel Islands",
                "Cyprus",
                "Czech Republic",
                "Denmark",
                "EIRE",
                "European Community",
                "Finland",
                "France",
                "Germany",
                "Greece",
                "Iceland",
                "Israel",
                "Italy",
                "Japan",
                "Lebanon",
                "Lithuania",
                "Malta",
                "Netherlands",
                "Norway",
                "Poland",
                "Portugal",
                "RSA",
                "Saudi Arabia",
                "Singapore",
                "Spain",
                "Sweden",
                "Switzerland",
                "USA",
                "United Arab Emirates",
                "United Kingdom",
                "Unspecified"
            ],
            index=36   # United Kingdom por defecto
        )
    
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
                st.metric("Demanda estimada", f"{prediction_data.get('prediction',0):.2f} unidades")
            
            with col2:
                st.metric("Modelo", prediction_data.get('model', 'N/A'))
            
            with col3:
                st.metric("Tiempo", f"{prediction_data.get('processing_time_ms', 0):.2f} ms")
            
            st.write(f"Timestamp: {prediction_data.get('timestamp')}")
            if prediction_data.get("status") == "success":
                st.success("Predicción realizada correctamente")
            else:
                st.error(prediction_data.get("status"))

            if prediction_data.get("interpretation"):
                st.info(f"🧠 **Interpretación:** {prediction_data['interpretation']}")
            
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

# ============================================================
# Resultados y Validación del Modelo (salidas en pantalla)
# ============================================================
# Muestra en el dashboard, con tablas + figuras + interpretación,
# los mismos resultados que contienen los reportes descargables
# (PDF/Word/Excel), tal como exige la consigna para "las salidas
# a pantalla".
st.divider()
st.header("📈 Resultados y Validación del Modelo")
st.caption(
    "Estas tablas y figuras provienen del mismo pipeline de entrenamiento que generó "
    "los reportes PDF/Word/Excel. Se muestran aquí en vivo, con su interpretación."
)


@st.cache_data(ttl=600, show_spinner=False)
def get_reports_summary():
    resp = requests.get(f"{API_URL}/api/reports/summary", timeout=15)
    resp.raise_for_status()
    return resp.json()


try:
    summary = get_reports_summary()

    tab_eda, tab_models, tab_cv, tab_stats = st.tabs([
        "🔎 EDA", "🧮 Comparación de Modelos", "🔁 Validación Cruzada", "📐 Pruebas Estadísticas"
    ])

    with tab_eda:
        st.subheader("Estadísticos Descriptivos")
        eda = summary.get("eda", {})
        if eda.get("table"):
            st.dataframe(eda["table"], use_container_width=True)
        else:
            st.warning("No hay tabla de EDA disponible en el backend.")
        if eda.get("interpretation"):
            st.info(f"🧠 **Interpretación:** {eda['interpretation']}")

        col_a, col_b = st.columns(2)
        with col_a:
            st.image(f"{API_URL}/api/reports/figure/correlation_heatmap", caption="Mapa de calor de correlaciones", use_container_width=True)
        with col_b:
            st.image(f"{API_URL}/api/reports/figure/time_series_sales", caption="Serie temporal de ventas", use_container_width=True)

    with tab_models:
        st.subheader("Comparación entre los 5 modelos entrenados (MLP, LSTM, GRU, CNN-LSTM, CNN-GRU)")
        cmp = summary.get("model_comparison", {})
        if cmp.get("table"):
            st.dataframe(cmp["table"], use_container_width=True)
        else:
            st.warning("No hay tabla de comparación de modelos disponible en el backend.")
        if cmp.get("interpretation"):
            st.warning(f"🧠 **Interpretación:** {cmp['interpretation']}")

    with tab_cv:
        st.subheader(f"Validación Cruzada")
        cv = summary.get("cross_validation", {})
        if cv.get("table"):
            st.dataframe(cv["table"], use_container_width=True)
        else:
            st.warning("No hay tabla de validación cruzada disponible en el backend.")
        if cv.get("interpretation"):
            st.info(f"🧠 **Interpretación:** {cv['interpretation']}")
        st.image(f"{API_URL}/api/reports/figure/cross_validation_boxplot", caption="Distribución del error por modelo (folds)", use_container_width=True)

    with tab_stats:
        st.subheader("Ranking y Pruebas Estadísticas Robustas (Friedman / Wilcoxon / Nemenyi)")
        stats = summary.get("statistics", {})
        if stats.get("ranking"):
            st.dataframe(stats["ranking"], use_container_width=True)
        else:
            st.warning("No hay tabla de ranking disponible en el backend.")
        if stats.get("interpretation"):
            st.info(f"🧠 **Interpretación:** {stats['interpretation']}")
        col_a, col_b = st.columns(2)
        with col_a:
            st.image(f"{API_URL}/api/reports/figure/significance_heatmap", caption="Mapa de calor de p-valores", use_container_width=True)
        with col_b:
            st.image(f"{API_URL}/api/reports/figure/critical_difference", caption="Diagrama de diferencia crítica (Nemenyi)", use_container_width=True)

except requests.exceptions.RequestException:
    st.warning(
        "⚠️ No se pudo cargar la sección de resultados en vivo (el backend aún no tiene los "
        "endpoints /api/reports/*, o no está disponible). Los reportes PDF/Word/Excel siguen "
        "disponibles de forma independiente."
    )

# ============================================================
# Descarga de Reportes Finales (generados en vivo por el backend)
# ============================================================
st.divider()
st.header("📥 Descargar Reportes Finales")
st.caption(
    "Genera y descarga, en este momento, el reporte completo (EDA, comparación de modelos, "
    "validación cruzada, hiperparámetros y pruebas estadísticas) directamente desde el "
    "servidor desplegado. Puede tardar unos segundos."
)

_REPORT_DOWNLOAD_OPTIONS = {
    "pdf": {"label": "📄 Descargar PDF", "mime": "application/pdf", "filename": "reporte_final.pdf"},
    "word": {"label": "📝 Descargar Word", "mime": "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "filename": "reporte_final.docx"},
    "excel": {"label": "📊 Descargar Excel", "mime": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "filename": "reporte_final.xlsx"},
}

col_pdf, col_word, col_excel = st.columns(3)
for col, formato in zip([col_pdf, col_word, col_excel], ["pdf", "word", "excel"]):
    opts = _REPORT_DOWNLOAD_OPTIONS[formato]
    with col:
        if st.button(opts["label"], key=f"gen_{formato}", use_container_width=True):
            with st.spinner(f"Generando reporte {formato.upper()}..."):
                try:
                    file_response = requests.get(f"{API_URL}/api/reports/download/{formato}", timeout=60)
                    file_response.raise_for_status()
                    st.download_button(
                        label=f"⬇️ Guardar {opts['filename']}",
                        data=file_response.content,
                        file_name=opts["filename"],
                        mime=opts["mime"],
                        key=f"save_{formato}",
                        use_container_width=True,
                    )
                except requests.exceptions.RequestException as e:
                    st.error(f"❌ No se pudo generar el reporte {formato.upper()}: {e}")
