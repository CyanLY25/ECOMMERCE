from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any
from datetime import datetime


class PredictionRequest(BaseModel):
    """
    Schema para solicitudes de predicción.
    Contiene los datos necesarios para realizar una predicción de demanda.
    """
    product_id: Optional[str] = Field(None, description="ID del producto", examples=["PROD123"])
    features: Dict[str, Any] = Field(
        ...,
        description="Características del producto para la predicción",
        examples=[{
            "StockCode": 1001,
            "UnitPrice": 19.99,
            "CustomerID": 12345,
            "Country": 1,
            "Año": 2024,
            "Mes": 7,
            "Día": 9,
            "Hora": 14,
            "DíaSemana": 3,
            "SemanaAño": 28,
            "Trimestre": 3,
            "EsFinDeSemana": 0,
            "MesNombre": "Julio"
        }]
    )
    
    @field_validator('features')
    @classmethod
    def validate_features(cls, v):
        """Valida que las características necesarias estén presentes."""
        required_features = [
            "StockCode", "UnitPrice", "CustomerID", "Country", "Año", "Mes",
            "Día", "Hora", "DíaSemana", "SemanaAño", "Trimestre", "EsFinDeSemana",
            "MesNombre"
        ]
        missing = [f for f in required_features if f not in v]
        if missing:
            raise ValueError(f"Faltan características requeridas: {', '.join(missing)}")
        return v


class PredictionResponse(BaseModel):
    """
    Schema para respuestas de predicción.
    Contiene el resultado de la predicción y información adicional.
    """
    prediction: Optional[float] = Field(None, description="Valor de la predicción de demanda", examples=[123.45])
    model: Optional[str] = Field(None, description="Modelo utilizado para la predicción", examples=["MLP"])
    processing_time_ms: Optional[float] = Field(None, description="Tiempo de procesamiento en milisegundos", examples=[12.5])
    timestamp: Optional[str] = Field(None, description="Fecha y hora de la predicción en formato ISO", examples=[datetime.now().isoformat()])
    status: str = Field(..., description="Estado de la solicitud", examples=["success"])
    interpretation: Optional[str] = Field(
        None,
        description="Interpretación en lenguaje natural de la predicción, contextualizada contra el histórico y la confiabilidad del modelo.",
        examples=["La demanda estimada (5.2 unidades) está cerca del promedio histórico..."]
    )


class ErrorResponse(BaseModel):
    """
    Schema para respuestas de error.
    """
    status: str = Field(..., description="Estado de error", examples=["error"])
    message: str = Field(..., description="Mensaje de error", examples=["Modelo no disponible"])


class HealthCheckResponse(BaseModel):
    """
    Schema para respuesta de health check.
    """
    status: str
    model_loaded: bool
    model: Optional[str] = None
    encoders_loaded: Optional[int] = None
    scaler_loaded: Optional[bool] = None
