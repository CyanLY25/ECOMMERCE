from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Configuración del backend FastAPI.
    """
    # Rutas
    BASE_DIR: Path = Path(__file__).parent.parent.resolve()
    BACKEND_MODEL_DIR: Path = BASE_DIR / "backend" / "model"
    BEST_MODEL_PATH: Path = BACKEND_MODEL_DIR / "best_model.keras"
    SCALER_PATH: Path = BACKEND_MODEL_DIR / "scaler.pkl"
    STOCKCODE_ENCODER_PATH: Path = BACKEND_MODEL_DIR / "stockcode_encoder.pkl"
    COUNTRY_ENCODER_PATH: Path = BACKEND_MODEL_DIR / "country_encoder.pkl"
    CUSTOMERID_ENCODER_PATH: Path = BACKEND_MODEL_DIR / "customerid_encoder.pkl"
    MESNOMBRE_ENCODER_PATH: Path = BACKEND_MODEL_DIR / "mesnombre_encoder.pkl"
    MODEL_INFO_PATH: Path = BACKEND_MODEL_DIR / "model_info.json"
    
    # API settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = True


settings = Settings()
