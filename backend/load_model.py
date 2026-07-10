import sys
import pickle
import json
from pathlib import Path
from typing import Optional, Dict, Any
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Añadir el directorio raíz del proyecto al path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import settings
from ia.utils.logger import setup_logger


class BackendModelLoader:
    """
    Cargador singleton de modelo, scaler y encoders para el backend FastAPI.
    Carga todo desde backend/model/.
    """
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if BackendModelLoader._initialized:
            return
        BackendModelLoader._initialized = True
        
        self.logger = setup_logger("backend_model_loader", PROJECT_ROOT / "ia" / "logs" / "inference.log")
        
        self.model: Optional[tf.keras.Model] = None
        self.scaler: Optional[StandardScaler] = None
        self.encoders: Dict[str, LabelEncoder] = {}
        self.model_info: Optional[Dict[str, Any]] = None
        self.is_loaded: bool = False
        self.model_name: Optional[str] = None
        self.sequence_length: int = 10  # Default

        self._load_all()

    def _load_all(self):
        """Carga modelo, scaler, encoders y model_info.json."""
        self.logger.info("=" * 60)
        self.logger.info("Cargando recursos del backend desde backend/model/")
        
        # 1. Cargar model_info.json
        try:
            if settings.MODEL_INFO_PATH.exists():
                with open(settings.MODEL_INFO_PATH, "r", encoding="utf-8") as f:
                    self.model_info = json.load(f)
                self.model_name = self.model_info.get("model")
                self.logger.info(f"Modelo: {self.model_name}")
                
                # Definir sequence_length según el modelo
                if self.model_name in ["LSTM", "GRU", "CNN-LSTM", "CNN-GRU"]:
                    self.sequence_length = 10
                self.logger.info(f"Longitud de secuencia: {self.sequence_length}")
        except Exception as e:
            self.logger.warning(f"No se pudo cargar model_info.json: {e}")
        
        # 2. Cargar modelo
        try:
            if settings.BEST_MODEL_PATH.exists():
                self.model = tf.keras.models.load_model(settings.BEST_MODEL_PATH)
                self.logger.info(f"Modelo cargado correctamente")
        except Exception as e:
            self.logger.error(f"Error al cargar el modelo: {e}")
        
        # 3. Cargar scaler
        try:
            if settings.SCALER_PATH.exists():
                with open(settings.SCALER_PATH, "rb") as f:
                    self.scaler = pickle.load(f)
                self.logger.info("Scaler cargado correctamente")
        except Exception as e:
            self.logger.warning(f"No se pudo cargar el scaler: {e}")
        
        # 4. Cargar encoders
        encoder_configs = [
            ("stockcode", settings.STOCKCODE_ENCODER_PATH),
            ("country", settings.COUNTRY_ENCODER_PATH),
            ("customerid", settings.CUSTOMERID_ENCODER_PATH),
            ("mesnombre", settings.MESNOMBRE_ENCODER_PATH)
        ]
        
        for name, path in encoder_configs:
            try:
                if path.exists():
                    with open(path, "rb") as f:
                        self.encoders[name] = pickle.load(f)
                    self.logger.info(f"Encoder {name} cargado correctamente")
            except Exception as e:
                self.logger.warning(f"No se pudo cargar encoder {name}: {e}")
        
        # Verificar si todos los recursos necesarios están cargados
        self.is_loaded = all([
            self.model is not None,
            self.scaler is not None,
            len(self.encoders) >= 4
        ])
        
        if self.is_loaded:
            self.logger.info("Todos los recursos cargados correctamente")
        else:
            self.logger.warning("Faltan recursos para funcionamiento completo")
        
        self.logger.info("=" * 60)

    def encode_feature(self, feature_name: str, value: Any) -> int:
        """Codifica una característica con su encoder correspondiente."""
        encoder_key = feature_name.lower()
        if encoder_key not in self.encoders:
            self.logger.warning(f"No hay encoder para {feature_name}")
            return 0
        encoder = self.encoders[encoder_key]
        value_str = str(value)
        if value_str not in encoder.classes_:
            self.logger.warning(f"Categoría desconocida para {feature_name}: {value_str}")
            return 0
        return int(encoder.transform([value_str])[0])

    def predict(self, features: Any) -> float:
        """Realiza una predicción con el modelo cargado."""
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Modelo no cargado")
        prediction = self.model.predict(features, verbose=0)
        return float(prediction[0][0])

    def get_model_info(self) -> Dict[str, Any]:
        """Devuelve la información del modelo cargado."""
        return self.model_info or {}

    def get_health_status(self) -> Dict[str, Any]:
        """Devuelve el estado para el endpoint de health check."""
        return {
            "status": "ok",
            "model_loaded": self.model is not None,
            "model": self.model_name,
            "scaler_loaded": self.scaler is not None,
            "encoders_loaded": len(self.encoders)
        }


# Instancia singleton
model_loader = BackendModelLoader()