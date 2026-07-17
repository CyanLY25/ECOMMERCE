import sys
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Añadir el directorio raíz al path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.schemas import PredictionRequest, PredictionResponse
from backend.load_model import model_loader
from ia.utils.logger import setup_logger
from ia.config.config import AIConfig

# Orden exacto de características según el preprocesamiento de entrenamiento
# (producto-día agregado, con historial de demanda). Debe coincidir con las
# columnas de ia/dataset/processed/train.csv (sin la columna 'Quantity').
FEATURE_ORDER = [
    "StockCode", "UnitPrice", "Country", "NumTransacciones",
    "lag_1", "lag_7", "lag_14", "lag_30",
    "rolling_mean_7", "rolling_std_7", "rolling_mean_30",
    "Mes", "Día", "DíaSemana", "SemanaAño", "Trimestre", "EsFinDeSemana", "MesNombre",
]

# Features que se toman del snapshot de historial del producto (ya vienen
# escaladas/codificadas desde el despliegue, no requieren transformación).
HISTORY_FEATURES = {
    "UnitPrice", "Country", "NumTransacciones",
    "lag_1", "lag_7", "lag_14", "lag_30",
    "rolling_mean_7", "rolling_std_7", "rolling_mean_30",
}

MESES_ES = {
    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
    7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre",
}

# Configurar logger
logger = setup_logger("prediction", PROJECT_ROOT / "ia" / "logs" / "inference.log")

# Umbral de R² por debajo del cual se advierte que el modelo tiene baja confiabilidad
LOW_CONFIDENCE_R2_THRESHOLD = 0.3


def _load_historical_quantity_stats():
    """
    Carga (una sola vez, a nivel de módulo) la media/desviación histórica de
    la variable objetivo (Quantity) desde el reporte de EDA, para poder
    contextualizar cada predicción en pantalla. Si el archivo no existe,
    retorna None sin romper el flujo de predicción.
    """
    try:
        config = AIConfig()
        stats_df = pd.read_csv(config.STATISTICS_PATH)
        row = stats_df[stats_df.iloc[:, 0] == "Quantity"]
        if row.empty:
            return None
        row = row.iloc[0]
        desviacion = row.get("Desviación Estándar", row.get("DesviaciónEstándar"))
        return {"media": float(row.get("Media")), "desviacion": float(desviacion)}
    except Exception as e:
        logger.warning(f"No se pudieron cargar estadísticos históricos de Quantity: {e}")
        return None


_HISTORICAL_QUANTITY_STATS = _load_historical_quantity_stats()


def _load_product_history():
    """
    Carga (una sola vez, a nivel de módulo) el snapshot con el último
    estado conocido de cada producto (lags, rolling stats, precio, país),
    generado por ia/deployment/deployer.py durante el despliegue.
    """
    try:
        from backend.config import settings
        with open(settings.PRODUCT_HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"No se pudo cargar el historial de productos: {e}")
        return {}


_PRODUCT_HISTORY = _load_product_history()


def build_prediction_interpretation(prediction: float) -> str:
    """
    Construye el texto de interpretación que acompaña a cada predicción
    mostrada en pantalla: la contextualiza contra el promedio histórico y
    advierte sobre la confiabilidad del modelo según sus métricas (R²/MAPE),
    tal como exige la consigna para las "salidas a pantalla".
    """
    partes = []

    if _HISTORICAL_QUANTITY_STATS:
        media = _HISTORICAL_QUANTITY_STATS["media"]
        desviacion = _HISTORICAL_QUANTITY_STATS["desviacion"]
        diferencia = prediction - media
        if abs(diferencia) <= desviacion:
            posicion = "cercana al promedio histórico"
        elif diferencia > desviacion:
            posicion = "por encima del promedio histórico"
        else:
            posicion = "por debajo del promedio histórico"
        partes.append(
            f"La demanda estimada ({prediction:.2f} unidades) es {posicion} "
            f"({media:.2f} ± {desviacion:.2f} unidades por línea de pedido en el histórico)."
        )

    model_info = model_loader.get_model_info() or {}
    metrics = model_info.get("metrics", {})
    r2 = metrics.get("r2")
    mape = metrics.get("mape")
    if r2 is not None and r2 < LOW_CONFIDENCE_R2_THRESHOLD:
        partes.append(
            f"Nota de confiabilidad: el modelo {model_loader.model_name} tiene un R² de {r2:.4f} "
            f"y un MAPE de {mape:.1f}% en el conjunto de prueba, lo que indica una capacidad "
            "predictiva limitada. Tome este valor como una referencia aproximada, no como una "
            "predicción de alta confianza."
        )
    elif r2 is not None:
        partes.append(f"El modelo {model_loader.model_name} tiene un R² de {r2:.4f} en el conjunto de prueba.")

    return " ".join(partes) if partes else "No hay información histórica disponible para interpretar esta predicción."


def make_prediction(request: PredictionRequest) -> PredictionResponse:
    """
    Realiza una predicción de demanda usando el modelo entrenado.
    
    Combina: (a) el último estado de historial de demanda conocido del
    producto (lags, rolling stats, precio, país — precalculados en el
    despliegue) con (b) las características derivadas de la fecha
    objetivo (mes, día de semana, etc.), en el mismo orden y forma que
    usó el pipeline de entrenamiento.
    
    Args:
        request: Datos de la solicitud de predicción (stock_code, target_date opcional).
        
    Returns:
        Respuesta con la predicción y metadata.
    """
    start_time = time.time()
    request_timestamp = datetime.now().isoformat()
    
    try:
        stock_code = str(request.stock_code)
        target_date = (
            datetime.fromisoformat(request.target_date) if request.target_date
            else datetime.now()
        )
        
        # Paso 1: Obtener el historial de demanda más reciente del producto
        history = _PRODUCT_HISTORY.get(stock_code)
        if history is None:
            raise ValueError(
                f"No hay historial de ventas para el producto '{stock_code}'. "
                "Solo se pueden predecir productos presentes en el histórico de entrenamiento."
            )
        
        # Paso 2: Calcular características derivadas de la fecha objetivo
        mes_nombre_es = MESES_ES[target_date.month]
        date_features = {
            "Mes": target_date.month,
            "Día": target_date.day,
            "DíaSemana": target_date.weekday(),
            "SemanaAño": target_date.isocalendar()[1],
            "Trimestre": (target_date.month - 1) // 3 + 1,
            "EsFinDeSemana": int(target_date.weekday() >= 5),
        }
        
        # Paso 3: Armar el vector de entrada en el orden exacto del entrenamiento
        processed_features = []
        for feature_name in FEATURE_ORDER:
            if feature_name == "StockCode":
                processed_features.append(float(model_loader.encode_feature("StockCode", stock_code)))
            elif feature_name == "MesNombre":
                processed_features.append(float(model_loader.encode_feature("MesNombre", mes_nombre_es)))
            elif feature_name in date_features:
                processed_features.append(float(date_features[feature_name]))
            elif feature_name in HISTORY_FEATURES:
                processed_features.append(float(history[feature_name]))
            else:
                raise ValueError(f"No se supo cómo calcular la característica '{feature_name}'")
        
        # Convertir a array numpy
        features_np = np.array(processed_features, dtype=np.float32)
        
        # Paso 4: Preparar secuencia si es un modelo recurrente
        is_sequential = model_loader.model_name in ["LSTM", "GRU", "CNN-LSTM", "CNN-GRU"]
        if is_sequential:
            sequence_length = model_loader.sequence_length
            sequence = np.tile(features_np, (sequence_length, 1))
            input_data = sequence.reshape(1, sequence_length, len(FEATURE_ORDER))
        else:
            input_data = features_np.reshape(1, -1)
        
        # Paso 5: Realizar la predicción
        prediction = model_loader.predict(input_data)
        
        # Calcular tiempo de procesamiento en milisegundos
        processing_time_ms = round((time.time() - start_time) * 1000, 2)
        
        logger.info(f"Predicción completada para producto {stock_code}: {prediction:.4f}")
        logger.info(f"Modelo: {model_loader.model_name}")
        logger.info(f"Tiempo de procesamiento: {processing_time_ms:.2f} ms")
        
        return PredictionResponse(
            prediction=prediction,
            model=model_loader.model_name,
            processing_time_ms=processing_time_ms,
            timestamp=request_timestamp,
            status="success",
            interpretation=build_prediction_interpretation(prediction)
        )
    
    except Exception as e:
        processing_time_ms = round((time.time() - start_time) * 1000, 2)
        logger.error(f"Error en predicción: {str(e)} (tiempo: {processing_time_ms:.2f} ms)")
        import traceback
        logger.error(traceback.format_exc())
        return PredictionResponse(
            prediction=None,
            model=model_loader.model_name,
            processing_time_ms=processing_time_ms,
            timestamp=request_timestamp,
            status=f"error: {str(e)}"
        )