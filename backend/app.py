import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Añadir el directorio raíz al path para imports correctos
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import settings
from backend.routes import router, store_router

# Inicializar la aplicación FastAPI
app = FastAPI(
    title="API de Predicción de Demanda",
    description="API para realizar predicciones de demanda de productos usando modelos de Machine Learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir las rutas
app.include_router(router)
app.include_router(store_router)


@app.get("/", summary="Endpoint raíz", description="Obtiene información básica de la API")
async def root():
    """
    Endpoint raíz de la API.
    """
    return {
        "message": "API de Predicción de Demanda",
        "version": "1.0.0",
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.app:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )
