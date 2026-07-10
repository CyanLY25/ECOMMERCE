#!/usr/bin/env python3
"""
Clase para cargar y usar el mejor modelo seleccionado automáticamente,
incluyendo scaler y encoders.
"""
import json
import sys
import pickle
from pathlib import Path
from typing import Optional, Dict, Any
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Añadir el directorio raíz del proyecto al path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ia.config.config import AIConfig
from ia.utils.logger import setup_logger


class ModelLoader:
    """
    Carga el mejor modelo, scaler y encoders desde los archivos correspondientes.
    Mantiene todo en memoria usando Singleton.
    """
    
    def __init__(self):
        """
        Inicializa el cargador y carga el mejor modelo, scaler y encoders.
        """
        self.config = AIConfig()
        self.logger = setup_logger("model_loader", self.config.LOGS_DIR / "inference.log")
        
        self.model: Optional[tf.keras.Model] = None
        self.model_name: Optional[str] = None
        self.metrics: Optional[Dict[str, Any]] = None
        self.model_path: Optional[Path] = None
        self.is_loaded: bool = False
        
        self.scaler: Optional[StandardScaler] = None
        self.encoders: Dict[str, LabelEncoder] = {}
        
        # Parámetros de secuencia para modelos recurrentes
        self.sequence_length: int = 10
        
        self._load_best_model()
        self._load_scaler_and_encoders()
        
    def _load_best_model(self):
        """
        Lee best_model.json (o final_best_model.json) y carga el modelo correspondiente.
        """
        self.logger.info("=" * 60)
        self.logger.info("Cargando mejor modelo...")
        
        # 1. Leer best_model.json (o final_best_model.json si existe)
        best_model_path = self.config.BEST_MODEL_PATH
        if not best_model_path.exists():
            # Intentar con final_best_model.json
            final_best_path = self.config.FINAL_BEST_MODEL_PATH
            if final_best_path.exists():
                best_model_path = final_best_path
            else:
                self.logger.error(f"Archivo best_model.json/final_best_model.json no encontrado: {best_model_path}")
                self.is_loaded = False
                raise FileNotFoundError(f"No existe un modelo seleccionado: {best_model_path}")
        
        with open(best_model_path, "r", encoding="utf-8") as f:
            best_model_data = json.load(f)
        
        self.model_name = best_model_data["model"]
        self.model_path = Path(best_model_data["path"])
        self.metrics = {
            "rmse": best_model_data["rmse"],
            "mae": best_model_data["mae"],
            "r2": best_model_data["r2"]
        }
        
        # Obtener longitud de secuencia según el modelo
        if self.model_name == "LSTM":
            self.sequence_length = self.config.LSTM_SEQUENCE_LENGTH
        elif self.model_name == "GRU":
            self.sequence_length = self.config.GRU_SEQUENCE_LENGTH
        elif self.model_name == "CNN-LSTM":
            self.sequence_length = self.config.CNN_LSTM_SEQUENCE_LENGTH
        elif self.model_name == "CNN-GRU":
            self.sequence_length = self.config.CNN_GRU_WINDOW_SIZE
        
        self.logger.info(f"Modelo: {self.model_name}")
        self.logger.info(f"Archivo: {self.model_path}")
        self.logger.info(f"Longitud de secuencia: {self.sequence_length}")
        
        # 2. Cargar el modelo .keras
        if not self.model_path.exists():
            self.logger.error(f"Archivo del modelo no encontrado: {self.model_path}")
            self.is_loaded = False
            raise FileNotFoundError(f"El modelo seleccionado no fue encontrado: {self.model_path}")
        
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            self.is_loaded = True
            self.logger.info("Modelo cargado correctamente.")
        except Exception as e:
            self.logger.error(f"Error al cargar el modelo: {str(e)}")
            self.is_loaded = False
            raise RuntimeError(f"Error al cargar el modelo: {str(e)}") from e
        
        self.logger.info("=" * 60)
    
    def _load_scaler_and_encoders(self):
        """
        Carga scaler.pkl y todos los LabelEncoder disponibles.
        """
        self.logger.info("Cargando scaler y encoders...")
        
        # Cargar scaler
        scaler_path = self.config.MODELS_DIR / "scaler.pkl"
        if scaler_path.exists():
            try:
                with open(scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
                self.logger.info(f"Scaler cargado correctamente desde {scaler_path}")
            except Exception as e:
                self.logger.warning(f"No se pudo cargar scaler: {str(e)}")
        else:
            self.logger.warning(f"Archivo scaler.pkl no encontrado en {scaler_path}")
        
        # Cargar encoders
        encoder_names = ["stockcode", "country", "customerid", "mesnombre"]
        for name in encoder_names:
            encoder_path = self.config.MODELS_DIR / f"{name}_encoder.pkl"
            if encoder_path.exists():
                try:
                    with open(encoder_path, "rb") as f:
                        self.encoders[name] = pickle.load(f)
                    self.logger.info(f"Encoder {name} cargado correctamente desde {encoder_path}")
                except Exception as e:
                    self.logger.warning(f"No se pudo cargar encoder {name}: {str(e)}")
            else:
                self.logger.warning(f"Archivo {name}_encoder.pkl no encontrado")
        
        self.logger.info("Carga de scaler y encoders completada.")
    
    def encode_feature(self, feature_name: str, value: Any) -> int:
        """
        Codifica una característica usando el LabelEncoder correspondiente.
        Maneja categorías desconocidas de forma segura.
        
        Args:
            feature_name: Nombre de la característica (ej: "stockcode", "country").
            value: Valor a codificar.
            
        Returns:
            Valor codificado (int).
        """
        encoder_key = feature_name.lower()
        
        if encoder_key not in self.encoders:
            self.logger.warning(f"No hay encoder para {feature_name}, devolviendo 0")
            return 0
        
        encoder = self.encoders[encoder_key]
        value_str = str(value)
        
        if value_str not in encoder.classes_:
            self.logger.warning(f"Categoría desconocida para {feature_name}: {value_str}, devolviendo 0")
            return 0
        
        return int(encoder.transform([value_str])[0])
    
    def predict(self, features: np.ndarray) -> float:
        """
        Realiza una predicción usando el modelo cargado.
        
        Args:
            features: Array numpy con las características de entrada.
            
        Returns:
            Valor predicho (float).
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Modelo no cargado. No se puede realizar la predicción.")
        
        try:
            # Asegurar que las características tengan la forma correcta
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            
            # Realizar predicción
            prediction = self.model.predict(features, verbose=0)
            
            # Devolver el valor como float
            return float(prediction[0][0])
        except Exception as e:
            self.logger.error(f"Error al realizar la predicción: {str(e)}")
            raise RuntimeError(f"Error al realizar la predicción: {str(e)}") from e
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Devuelve la información del modelo cargado.
        
        Returns:
            Diccionario con nombre, métricas, ruta y estado.
        """
        return {
            "model": self.model_name,
            "rmse": self.metrics["rmse"],
            "mae": self.metrics["mae"],
            "r2": self.metrics["r2"],
            "path": str(self.model_path) if self.model_path else "",
            "loaded": self.is_loaded,
            "encoders_loaded": list(self.encoders.keys()),
            "scaler_loaded": self.scaler is not None
        }


# Instancia global (singleton) del cargador de modelos
model_loader_instance = ModelLoader()
