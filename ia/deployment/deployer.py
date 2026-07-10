#!/usr/bin/env python3
"""
Módulo de despliegue automático del mejor modelo.
Copia el modelo, scaler y encoders hacia el directorio backend/model/
"""
import json
import shutil
import sys
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
            "version": "1.0"
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
