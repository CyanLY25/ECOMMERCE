import sys
import time
import numpy as np
from pathlib import Path
from datetime import datetime

# Añadir el directorio raíz al path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.schemas import PredictionRequest, PredictionResponse
from backend.load_model import model_loader
from ia.utils.logger import setup_logger

# Orden exacto de características según el preprocesamiento de entrenamiento
FEATURE_ORDER = [
    "StockCode", "UnitPrice", "CustomerID", "Country", "Año", "Mes",
    "Día", "Hora", "DíaSemana", "SemanaAño", "Trimestre", "EsFinDeSemana",
    "MesNombre"
]

# Configurar logger
logger = setup_logger("prediction", PROJECT_ROOT / "ia" / "logs" / "inference.log")


def make_prediction(request: PredictionRequest) -> PredictionResponse:
    """
    Realiza una predicción de demanda usando el modelo entrenado,
    siguiendo el mismo pipeline de preprocesamiento que durante el entrenamiento.
    
    Args:
        request: Datos de la solicitud de predicción.
        
    Returns:
        Respuesta con la predicción y metadata.
    """
    start_time = time.time()
    request_timestamp = datetime.now().isoformat()
    
    try:
        # Paso 1: Extraer características en el orden correcto
        features_dict = request.features
        
        # Paso 2: Preprocesar cada característica
        processed_features = []
        for feature_name in FEATURE_ORDER:
            value = features_dict[feature_name]
            
            if feature_name in ["StockCode", "Country", "CustomerID", "MesNombre"]:
                # Codificar variables categóricas usando los encoders cargados
                encoded_value = model_loader.encode_feature(feature_name, value)
                processed_features.append(encoded_value)
            elif feature_name == "UnitPrice":
                # Escalar usando el scaler cargado
                value_scaled = model_loader.scaler.transform([[value]])[0][0]
                processed_features.append(float(value_scaled))
            else:
                # Variables numéricas/temporales: usar valor directamente
                processed_features.append(float(value))
        
        # Convertir a array numpy
        features_np = np.array(processed_features, dtype=np.float32)
        
        # Paso 3: Preparar secuencia si es un modelo recurrente
        is_sequential = model_loader.model_name in ["LSTM", "GRU", "CNN-LSTM", "CNN-GRU"]
        if is_sequential:
            sequence_length = model_loader.sequence_length
            sequence = np.tile(features_np, (sequence_length, 1))
            input_data = sequence.reshape(1, sequence_length, len(FEATURE_ORDER))
        else:
            input_data = features_np.reshape(1, -1)
        
        # Paso 4: Realizar la predicción
        prediction = model_loader.predict(input_data)
        
        # Calcular tiempo de procesamiento en milisegundos
        processing_time_ms = round((time.time() - start_time) * 1000, 2)
        
        logger.info(f"Predicción completada: {prediction:.4f}")
        logger.info(f"Modelo: {model_loader.model_name}")
        logger.info(f"Tiempo de procesamiento: {processing_time_ms:.2f} ms")
        
        return PredictionResponse(
            prediction=prediction,
            model=model_loader.model_name,
            processing_time_ms=processing_time_ms,
            timestamp=request_timestamp,
            status="success"
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