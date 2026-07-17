import sys
import tempfile
import uuid
import json
from pathlib import Path
from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from fastapi.responses import FileResponse

# Añadir el directorio raíz al path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.schemas import (
    PredictionRequest, PredictionResponse, HealthCheckResponse, ErrorResponse,
    ProductCreate, ProductUpdate, OrderCreate, OrderStatusUpdate,
)
from backend.predict import make_prediction
from backend.load_model import model_loader
from backend import store
from ia.config.config import AIConfig
from ia.reports.report_data import load_report_data
from ia.reports.report_generator import ReportGenerator
from ia.reports import interpretations as interp

router = APIRouter(prefix="/api", tags=["prediction"])
store_router = APIRouter(prefix="/api/store", tags=["store"])

_config = AIConfig()
store.init_db()

# Whitelist de figuras que se pueden servir por HTTP (evita exponer el filesystem)
ALLOWED_FIGURES = {
    "correlation_heatmap": "5_correlation_heatmap.png",
    "time_series_sales": "13_time_series_sales.png",
    "country_distribution": "11_country_distribution.png",
    "top_products": "9_top_products.png",
    "cross_validation_boxplot": "cross_validation_boxplot.png",
    "cross_validation_rmse": "cross_validation_rmse.png",
    "significance_heatmap": "significance_heatmap.png",
    "critical_difference": "critical_difference.png",
    "ranking_plot": "ranking_plot.png",
    "tuning_comparison": "tuning_comparison.png",
}


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
    "/products",
    summary="Listar productos disponibles para predicción",
    description="Devuelve los códigos de producto (StockCode) con historial de demanda "
                "conocido, listos para usarse en /api/predict."
)
async def list_products():
    """Lista los productos disponibles para predicción (tienen snapshot de historial)."""
    from backend.config import settings
    try:
        with open(settings.PRODUCT_HISTORY_PATH, "r", encoding="utf-8") as f:
            history = json.load(f)
        return {"products": sorted(history.keys()), "count": len(history)}
    except Exception:
        return {"products": [], "count": 0}


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


@router.get(
    "/reports/summary",
    summary="Obtener tablas y métricas para las salidas en pantalla del dashboard",
    description=(
        "Devuelve, en JSON, las tablas de EDA, comparación de modelos, validación cruzada "
        "y pruebas estadísticas ya generadas por el pipeline de entrenamiento, cada una "
        "acompañada de su interpretación en lenguaje natural, para mostrarlas en el dashboard."
    ),
    responses={
        200: {"description": "Resumen de reportes obtenido exitosamente"}
    }
)
async def get_reports_summary():
    """
    Endpoint que expone al frontend las tablas/figuras/interpretaciones que
    ya usan los reportes PDF/Word/Excel, para que el dashboard en vivo
    también muestre "salidas a pantalla" con tablas e interpretación
    (no solo los archivos descargables).
    """
    try:
        data = load_report_data(_config)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"No se pudieron cargar los datos de reportes: {e}"
        )

    def df_records(df):
        return df.to_dict(orient="records") if df is not None and not df.empty else None

    return {
        "eda": {
            "table": df_records(data.get("eda_statistics")),
            "interpretation": interp.interpret_eda(data.get("eda_statistics")),
        },
        "model_comparison": {
            "table": df_records(data.get("model_comparison")),
            "interpretation": interp.interpret_model_comparison(
                data.get("model_comparison"), data.get("best_model")
            ),
        },
        "cross_validation": {
            "table": df_records(data.get("cv_summary")),
            "interpretation": interp.interpret_cross_validation(data.get("cv_summary")),
        },
        "statistics": {
            "ranking": df_records(data.get("ranking")),
            "interpretation": interp.interpret_statistics(
                data.get("friedman"), data.get("wilcoxon"),
                data.get("nemenyi"), data.get("ranking")
            ),
        },
        "figures": {key: interp.interpret_figure(filename) for key, filename in ALLOWED_FIGURES.items() if (_config.FIGURES_DIR / filename).exists()},
    }


@router.get(
    "/reports/figure/{figure_key}",
    summary="Obtener una figura del reporte como imagen PNG",
    description="Sirve una de las figuras generadas por el pipeline (whitelist fija) para incrustarla en el dashboard.",
    responses={
        200: {"description": "Imagen PNG de la figura solicitada"},
        404: {"description": "Figura no encontrada"}
    }
)
async def get_report_figure(figure_key: str):
    """
    Sirve una figura PNG por clave lógica (no por ruta de archivo directa,
    para no exponer el filesystem del servidor).
    """
    filename = ALLOWED_FIGURES.get(figure_key)
    if not filename:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Figura no reconocida.")

    figure_path = _config.FIGURES_DIR / filename
    if not figure_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="El archivo de la figura no existe en el servidor.")

    return FileResponse(str(figure_path), media_type="image/png")


_REPORT_FORMATS = {
    "pdf": {
        "media_type": "application/pdf",
        "filename": "reporte_final.pdf",
        "method": "generate_pdf",
    },
    "word": {
        "media_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "filename": "reporte_final.docx",
        "method": "generate_word",
    },
    "excel": {
        "media_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "filename": "reporte_final.xlsx",
        "method": "generate_excel",
    },
}


@router.get(
    "/reports/download/{formato}",
    summary="Generar y descargar el reporte final (PDF, Word o Excel) en vivo",
    description=(
        "Genera bajo demanda, en el servidor desplegado, el reporte final con EDA, "
        "comparación de modelos, validación cruzada, hiperparámetros y pruebas "
        "estadísticas, y lo entrega para descarga inmediata. "
        "`formato` debe ser uno de: pdf, word, excel."
    ),
    responses={
        200: {"description": "Archivo generado y listo para descargar"},
        400: {"description": "Formato no soportado"},
        500: {"description": "Error generando el reporte"},
    }
)
async def download_report(formato: str, background_tasks: BackgroundTasks):
    """
    Genera el reporte solicitado (PDF/Word/Excel) en un archivo temporal
    único por request y lo devuelve como descarga. El archivo temporal se
    elimina automáticamente después de enviarse.
    """
    formato = formato.lower()
    if formato not in _REPORT_FORMATS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Formato no soportado: '{formato}'. Use uno de: {list(_REPORT_FORMATS.keys())}."
        )

    spec = _REPORT_FORMATS[formato]
    tmp_dir = Path(tempfile.gettempdir())
    unique_suffix = uuid.uuid4().hex[:8]
    output_path = tmp_dir / f"{spec['filename'].rsplit('.', 1)[0]}_{unique_suffix}.{spec['filename'].rsplit('.', 1)[1]}"

    try:
        data = load_report_data(_config)
        generator = ReportGenerator(_config)
        method = getattr(generator, spec["method"])
        method(data, output_path)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"No se pudo generar el reporte en formato {formato}: {e}"
        )

    # Limpia el archivo temporal después de que la respuesta se haya enviado
    background_tasks.add_task(lambda p=output_path: p.unlink(missing_ok=True))

    return FileResponse(
        str(output_path),
        media_type=spec["media_type"],
        filename=spec["filename"],
        background=background_tasks,
    )

# ==================== TIENDA: PRODUCTOS ====================
# Estos endpoints reemplazan a las llamadas que antes iban directo a
# Supabase desde admin_vendedor.py y tienda_cliente.py. Al vivir en este
# mismo backend (ya desplegado en Render), ambas apps de Streamlit -aunque
# se despliegan por separado- comparten siempre los mismos datos.

@store_router.get("/products", summary="Listar productos de la tienda")
async def store_list_products():
    return {"products": store.list_products()}


@store_router.post("/products", status_code=status.HTTP_201_CREATED, summary="Crear producto")
async def store_create_product(payload: ProductCreate):
    product = store.create_product(
        name=payload.name,
        description=payload.description or "",
        price=payload.price,
        stock=payload.stock,
        image_data=payload.image_base64,
        stock_code=payload.stock_code,
    )
    return product


@store_router.put("/products/{product_id}", summary="Actualizar producto")
async def store_update_product(product_id: str, payload: ProductUpdate):
    existing = store.get_product(product_id)
    if not existing:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Producto no encontrado")

    image_data = None
    update_image = False
    if payload.remove_image:
        image_data, update_image = None, True
    elif payload.image_base64:
        image_data, update_image = payload.image_base64, True

    product = store.update_product(
        product_id,
        name=payload.name,
        description=payload.description or "",
        price=payload.price,
        stock=payload.stock,
        image_data=image_data,
        update_image=update_image,
    )
    return product


@store_router.delete("/products/{product_id}", summary="Eliminar producto")
async def store_delete_product(product_id: str):
    deleted = store.delete_product(product_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Producto no encontrado")
    return {"status": "ok"}


# ==================== TIENDA: ORDENES ====================

@store_router.get("/orders", summary="Listar ordenes")
async def store_list_orders():
    return {"orders": store.list_orders()}


@store_router.post("/orders", status_code=status.HTTP_201_CREATED, summary="Crear orden")
async def store_create_order(payload: OrderCreate):
    products = {p["name"]: p for p in store.list_products()}
    for item in payload.items:
        product = products.get(item.name)
        if not product:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"El producto '{item.name}' ya no esta disponible",
            )
        if product["stock"] < item.quantity:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Stock insuficiente para '{item.name}'. Disponible: {product['stock']}",
            )

    total_amount = sum(item.price * item.quantity for item in payload.items)
    order = store.create_order(
        customer_name=payload.customer_name,
        customer_email=payload.customer_email or "",
        customer_phone=payload.customer_phone or "",
        shipping_address=payload.shipping_address or "",
        items=[item.model_dump() for item in payload.items],
        total_amount=total_amount,
    )
    return order


@store_router.put("/orders/{order_id}/status", summary="Cambiar estado de una orden")
async def store_update_order_status(order_id: str, payload: OrderStatusUpdate):
    if payload.status not in {"Pendiente", "Completado", "Cancelado"}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Estado invalido")
    order = store.update_order_status(order_id, payload.status)
    if not order:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Orden no encontrada")
    return order
