import sys
from pathlib import Path
from fastapi import APIRouter, HTTPException, status

# Añadir el directorio raíz al path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.schemas import PredictionRequest, PredictionResponse, HealthCheckResponse, ErrorResponse
from backend.predict import make_prediction
from backend.load_model import model_loader

router = APIRouter(prefix="/api", tags=["prediction"])


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Realizar predicción de demanda",
    description="Endpoint para realizar predicciones de demanda de productos usando el modelo entrenado.",
    responses={
        200: {"description": "Predicción realizada exitosamente"},
        500: {"description": "Error interno del servidor"},
        503: {"description": "Modelo no disponible", "model": ErrorResponse}
    }
)
async def predict(request: PredictionRequest):
    """
    Endpoint para realizar predicciones de demanda de productos.
    """
    if not model_loader.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo, scaler o encoders no disponibles."
        )
    
    try:
        return make_prediction(request)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno del servidor"
        ) from e


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Verificar estado del servicio",
    description="Endpoint de verificación de estado del servicio y del modelo.",
    responses={
        200: {"description": "Servicio funcionando correctamente"}
    }
)
async def health_check():
    """
    Endpoint de verificación de estado del servicio.
    """
    return HealthCheckResponse(**model_loader.get_health_status())


@router.get(
    "/model-info",
    summary="Obtener información del modelo",
    description="Endpoint para obtener la información del modelo actualmente cargado.",
    responses={
        200: {"description": "Información del modelo obtenida exitosamente"},
        503: {"description": "No existe un modelo seleccionado"}
    }
)
async def get_model_info():
    """
    Endpoint para obtener la información del modelo cargado.
    """
    if not model_loader.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No existe un modelo seleccionado."
        )
    
    return model_loader.get_model_info()