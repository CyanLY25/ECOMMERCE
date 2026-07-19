#!/usr/bin/env python3
"""
Módulo de despliegue automático del mejor modelo.
Copia el modelo, scaler y encoders hacia el directorio backend/model/
"""
import json
import shutil
import sys
import pickle
import pandas as pd
from pathlib import Path
from datetime import datetime

# Añadir el directorio raíz del proyecto al path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ia.utils.logger import setup_logger
from ia.config.config import AIConfig


class ModelDeployer:
    """
    Clase para realizar el despliegue automático del mejor modelo.
    """
    def __init__(self, config: AIConfig):
        self.config = config
        self.logger = setup_logger("deployment", config.LOGS_DIR / "deployment.log")
        
        # Directorio destino en backend
        self.backend_model_dir = project_root / "backend" / "model"
        
        # Nombres de archivos a copiar
        self.artifacts = [
            "scaler.pkl",
            "stockcode_encoder.pkl",
            "customerid_encoder.pkl",
            "country_encoder.pkl",
            "mesnombre_encoder.pkl"
        ]
    
    def _load_best_model_info(self):
        """
        Carga la información del mejor modelo desde best_model.json o final_best_model.json
        """
        # Intentar cargar best_model.json primero
        best_model_path = self.config.BEST_MODEL_PATH
        
        if best_model_path.exists():
            self.logger.info(f"Cargando info desde {best_model_path}")
            with open(best_model_path, "r", encoding="utf-8") as f:
                return json.load(f)
        
        # Si no existe, probar con final_best_model.json
        final_best_path = self.config.FINAL_BEST_MODEL_PATH
        if final_best_path.exists():
            self.logger.info(f"Cargando info desde {final_best_path}")
            with open(final_best_path, "r", encoding="utf-8") as f:
                return json.load(f)
        
        raise FileNotFoundError("No se encontró best_model.json ni final_best_model.json")
    
    def _build_product_history_snapshot(self):
        """
        Genera el último estado y una secuencia histórica real por producto.
        Los modelos secuenciales (incluido TFT) fueron entrenados con ventanas
        de observaciones reales, por lo que no se debe repetir una sola fila
        durante la inferencia.
        """
        self.logger.info("Generando snapshot de historial por producto...")

        train = pd.read_csv(self.config.TRAIN_DATA_PATH)
        val = pd.read_csv(self.config.VALIDATION_DATA_PATH)
        test = pd.read_csv(self.config.TEST_DATA_PATH)
        # El orden train -> val -> test conserva el orden cronológico dentro
        # de cada producto, porque el split fue por corte de fecha.
        full = pd.concat([train, val, test], ignore_index=True)

        stockcode_encoder_path = self.config.MODELS_DIR / "stockcode_encoder.pkl"
        if not stockcode_encoder_path.exists():
            self.logger.warning("No se encontró stockcode_encoder.pkl; se omite el snapshot")
            return
        with open(stockcode_encoder_path, "rb") as f:
            stockcode_encoder = pickle.load(f)

        feature_cols = list(self.config.MODEL_FEATURES)
        missing_features = [column for column in feature_cols if column not in full.columns]
        if missing_features:
            raise ValueError(
                f"No se puede generar historial: faltan características {missing_features}"
            )

        max_sequence_length = max(
            self.config.LSTM_SEQUENCE_LENGTH,
            self.config.GRU_SEQUENCE_LENGTH,
            self.config.CNN_LSTM_SEQUENCE_LENGTH,
            self.config.CNN_GRU_WINDOW_SIZE,
            self.config.TFT_WINDOW_SIZE,
        )

        snapshot = {}
        for encoded_stock_code, product_rows in full.groupby("StockCode", sort=False):
            try:
                raw_code = str(stockcode_encoder.inverse_transform([int(encoded_stock_code)])[0])
            except Exception:
                continue

            recent_rows = product_rows.tail(max_sequence_length)
            latest_row = recent_rows.iloc[-1]
            snapshot[raw_code] = {
                "latest": {column: float(latest_row[column]) for column in feature_cols},
                "sequence": [
                    {column: float(row[column]) for column in feature_cols}
                    for _, row in recent_rows.iterrows()
                ],
            }

        snapshot_path = self.backend_model_dir / "product_history.json"
        with open(snapshot_path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Snapshot de {len(snapshot)} productos guardado en {snapshot_path}")

    def deploy(self):
        """
        Realiza el despliegue completo del modelo y artefactos
        """
        self.logger.info("=" * 80)
        self.logger.info("Iniciando despliegue automático del mejor modelo")
        self.logger.info("=" * 80)
        
        # Paso 1: Cargar información del mejor modelo
        model_info = self._load_best_model_info()
        model_name = model_info["model"]
        model_source_path = Path(model_info["path"])
        
        self.logger.info(f"Mejor modelo: {model_name}")
        self.logger.info(f"Ruta fuente: {model_source_path}")
        
        # Paso 2: Crear directorio backend/model si no existe
        self.backend_model_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Directorio destino: {self.backend_model_dir}")
        
        # Paso 3: Copiar el modelo
        model_dest_path = self.backend_model_dir / "best_model.keras"
        if model_source_path.exists():
            shutil.copy(model_source_path, model_dest_path)
            self.logger.info(f"Modelo copiado a {model_dest_path}")
        else:
            raise FileNotFoundError(f"Archivo del modelo no encontrado: {model_source_path}")
        
        # Paso 4: Copiar scaler y encoders
        for artifact in self.artifacts:
            artifact_source_path = self.config.MODELS_DIR / artifact
            artifact_dest_path = self.backend_model_dir / artifact
            
            if artifact_source_path.exists():
                shutil.copy(artifact_source_path, artifact_dest_path)
                self.logger.info(f"Artefacto copiado: {artifact}")
            else:
                self.logger.warning(f"Artefacto no encontrado: {artifact}")
        
        # Paso 4.5: Generar snapshot de historial por producto (para inferencia)
        self._build_product_history_snapshot()
        
        # Paso 5: Generar model_info.json
        model_info_json = {
            "model": model_name,
            "date": datetime.now().isoformat(),
            "metrics": {
                "rmse": model_info.get("rmse"),
                "mae": model_info.get("mae"),
                "mape": model_info.get("mape"),
                "r2": model_info.get("r2")
            },
            "path": str(model_dest_path.absolute()),
            "version": "1.1",
            "feature_order": list(self.config.MODEL_FEATURES),
            "sequence_length": (
                self.config.TFT_WINDOW_SIZE if model_name == "TFT"
                else self.config.LSTM_SEQUENCE_LENGTH if model_name == "LSTM"
                else self.config.GRU_SEQUENCE_LENGTH if model_name == "GRU"
                else self.config.CNN_LSTM_SEQUENCE_LENGTH if model_name == "CNN-LSTM"
                else self.config.CNN_GRU_WINDOW_SIZE if model_name == "CNN-GRU"
                else None
            ),
        }
        
        model_info_path = self.backend_model_dir / "model_info.json"
        with open(model_info_path, "w", encoding="utf-8") as f:
            json.dump(model_info_json, f, indent=4, ensure_ascii=False)
        
        self.logger.info(f"model_info.json generado: {model_info_path}")
        
        self.logger.info("=" * 80)
        self.logger.info("Despliegue completado exitosamente!")
        self.logger.info("=" * 80)
        
        return model_info_json


def deploy_model():
    """
    Función conveniente para ejecutar el despliegue desde la línea de comandos
    """
    config = AIConfig()
    deployer = ModelDeployer(config)
    deployer.deploy()


if __name__ == "__main__":
    deploy_model()
