from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime


class PredictionRequest(BaseModel):
    """
    Schema para solicitudes de predicción.
    Solo requiere el código del producto; el resto de las características
    (precio, país, historial de demanda) se toman automáticamente del
    último estado conocido de ese producto, calculado en el despliegue.
    """
    stock_code: str = Field(..., description="Código del producto (StockCode)", examples=["10002"])
    target_date: Optional[str] = Field(
        None,
        description="Fecha objetivo de la predicción, formato YYYY-MM-DD. Si se omite, se usa la fecha actual.",
        examples=["2026-07-20"]
    )


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


# ==================== TIENDA (productos / ordenes) ====================
# Reemplazan a las tablas que antes vivian en Supabase; ahora las sirve
# este mismo backend, respaldado por SQLite (ver backend/store.py).

class ProductCreate(BaseModel):
    name: str = Field(..., examples=["Jumbo Bag Red Retrospot"])
    description: Optional[str] = Field("", examples=["Bolsa de tela grande, estampado retro rojo"])
    price: float = Field(..., ge=0, examples=[12.90])
    stock: int = Field(..., ge=0, examples=[100])
    image_base64: Optional[str] = Field(
        None, description="Data URI de la imagen (data:image/png;base64,...)"
    )
    stock_code: Optional[str] = Field(None, description="Codigo del producto en el dataset historico")


class ProductUpdate(BaseModel):
    name: str
    description: Optional[str] = ""
    price: float = Field(..., ge=0)
    stock: int = Field(..., ge=0)
    image_base64: Optional[str] = None
    remove_image: bool = False


class OrderItem(BaseModel):
    name: str
    quantity: int = Field(..., ge=1)
    price: float = Field(..., ge=0)


class OrderCreate(BaseModel):
    customer_name: str
    customer_email: Optional[str] = ""
    customer_phone: Optional[str] = ""
    shipping_address: Optional[str] = ""
    items: list[OrderItem]


class OrderStatusUpdate(BaseModel):
    status: str = Field(..., examples=["Completado"])