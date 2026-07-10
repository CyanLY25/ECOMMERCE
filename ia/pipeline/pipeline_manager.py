#!/usr/bin/env python3
"""
Módulo de orquestación del pipeline maestro del proyecto.
"""
import sys
import time
import json
from pathlib import Path
from typing import Callable, Dict, Any, List

# Añadir el directorio raíz del proyecto al path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ia.config.config import AIConfig
from ia.utils.logger import setup_logger
from ia.pipeline.execution_summary import ExecutionSummary


class PipelineManager:
    """
    Orquestador principal del pipeline maestro del proyecto.
    """
    
    def __init__(self, config: AIConfig):
        self.config = config
        self.logger = setup_logger("pipeline_manager", self.config.PROJECT_EXECUTION_LOG)
        self.summary = ExecutionSummary(config)
        
        # Registrar fases del pipeline
        self.phases: List[Dict[str, Any]] = [
            {
                "name": "Preprocessing",
                "flag": "RUN_PREPROCESSING",
                "function": self._run_preprocessing
            },
            {
                "name": "Training",
                "flag": "RUN_TRAINING",
                "function": self._run_training
            },
            {
                "name": "Model Comparison",
                "flag": "RUN_MODEL_COMPARISON",
                "function": self._run_model_comparison
            },
            {
                "name": "Cross Validation",
                "flag": "RUN_CROSS_VALIDATION",
                "function": self._run_cross_validation
            },
            {
                "name": "Hyperparameter Tuning",
                "flag": "RUN_HYPERPARAMETER_TUNING",
                "function": self._run_hyperparameter_tuning
            },
            {
                "name": "Statistical Tests",
                "flag": "RUN_STATISTICAL_TESTS",
                "function": self._run_statistical_tests
            },
            {
                "name": "Deployment",
                "flag": "RUN_DEPLOYMENT",
                "function": self._run_deployment
            }
        ]
        
    def _run_preprocessing(self):
        """
        Ejecuta la fase de preprocesamiento.
        """
        from ia.preprocessing.data_preprocessor import ImprovedDataPreprocessor
        
        preprocessor = ImprovedDataPreprocessor(self.config)
        preprocessor.full_preprocessing_pipeline()
        
    def _run_training(self):
        """
        Ejecuta la fase de entrenamiento de todos los modelos.
        """
        # Ejecutar entrenamiento para cada modelo
        try:
            from ia.training.mlp_trainer import main as train_mlp
            train_mlp()
        except Exception as e:
            self.logger.warning(f"MLP training failed: {e}")
        
        try:
            from ia.training.lstm_trainer import main as train_lstm
            train_lstm()
        except Exception as e:
            self.logger.warning(f"LSTM training failed: {e}")
        
        try:
            from ia.training.gru_trainer import main as train_gru
            train_gru()
        except Exception as e:
            self.logger.warning(f"GRU training failed: {e}")
        
        try:
            from ia.training.cnn_lstm_trainer import main as train_cnn_lstm
            train_cnn_lstm()
        except Exception as e:
            self.logger.warning(f"CNN-LSTM training failed: {e}")
        
        try:
            from ia.training.cnn_gru_trainer import main as train_cnn_gru
            train_cnn_gru()
        except Exception as e:
            self.logger.warning(f"CNN-GRU training failed: {e}")
            
    def _run_model_comparison(self):
        """
        Ejecuta la fase de comparación de modelos y selección del mejor.
        """
        from ia.evaluation.model_selector import ModelSelector
        
        selector = ModelSelector(self.config)
        selector.load_metrics()
        selector.compare_models()
        selector.save_best_model()
        
    def _run_cross_validation(self):
        """
        Ejecuta la fase de validación cruzada.
        """
        from ia.validation.cross_validation_runner import run_all_models_cross_validation
        
        run_all_models_cross_validation()
        
    def _run_hyperparameter_tuning(self):
        """
        Ejecuta la fase de hyperparameter tuning.
        """
        from ia.tuning.tuning_runner import run_all_models_tuning
        
        run_all_models_tuning()
        
    def _run_statistical_tests(self):
        """
        Ejecuta la fase de pruebas estadísticas.
        """
        from ia.statistics.statistics import main as run_statistics
        
        run_statistics()
        
    def _run_deployment(self):
        """
        Ejecuta la fase de despliegue automático al backend.
        """
        from ia.deployment.deployer import deploy_model
        
        deploy_model()
        
    def run(self):
        """
        Ejecuta el pipeline completo.
        """
        self.logger.info("=" * 80)
        self.logger.info("INICIANDO PIPELINE MAESTRO DEL PROYECTO")
        self.logger.info("=" * 80)
        
        # Asegurar que existan los directorios
        self.config.ensure_directories_exist()
        
        # Ejecutar cada fase del pipeline
        for phase_idx, phase in enumerate(self.phases, 1):
            phase_name = phase["name"]
            phase_flag = phase["flag"]
            phase_function = phase["function"]
            
            # Verificar si la fase está habilitada
            if not getattr(self.config, phase_flag, False):
                self.logger.info(f"Saltando fase {phase_idx}: {phase_name} (deshabilitada)")
                continue
                
            # Imprimir encabezado de la fase
            print("\n" + "=" * 80)
            print(f"FASE {phase_idx}")
            print(phase_name)
            
            self.logger.info(f"Ejecutando fase {phase_idx}: {phase_name}")
            
            # Ejecutar la fase
            start = time.time()
            success = True
            error = None
            
            try:
                phase_function()
            except Exception as e:
                success = False
                error = str(e)
                self.logger.error(f"Error en fase {phase_name}: {e}")
                
            end = time.time()
            
            # Registrar la fase en el summary
            self.summary.add_phase(phase_name, start, end, success, error)
            
            # Imprimir estado de la fase
            if success:
                print("Completado")
            else:
                print(f"Fallido: {error}")
                
            print("=" * 80)
            
        # Finalizar el summary
        # Obtener el mejor modelo
        best_model = None
        best_model_metrics = None
        
        try:
            if self.config.BEST_MODEL_PATH.exists():
                with open(self.config.BEST_MODEL_PATH, "r") as f:
                    best_model_info = json.load(f)
                best_model = best_model_info["model"]
                best_model_metrics = {
                    "rmse": best_model_info.get("rmse"),
                    "mae": best_model_info.get("mae"),
                    "r2": best_model_info.get("r2")
                }
        except Exception as e:
            self.logger.warning(f"No se pudo obtener el mejor modelo: {e}")
            
        self.summary.finish(best_model, best_model_metrics)
        
        # Guardar resúmenes
        self.summary.save_json()
        self.summary.save_markdown()
        
        # Verificar archivos
        warnings = self.summary.validate_files()
        
        # Imprimir resumen final
        print("\n" + "=" * 80)
        print("PROYECTO COMPLETADO")
        print("=" * 80)
        
        if best_model:
            print("\nModelo ganador:")
            print("  " + best_model)
            if best_model_metrics:
                print("\nRMSE:")
                print(f"  {best_model_metrics['rmse']:.4f}")
                print("MAE:")
                print(f"  {best_model_metrics['mae']:.4f}")
                print("R²:")
                print(f"  {best_model_metrics['r2']:.4f}")
                
        # Verificar deployment
        deployment_ok = self.config.BACKEND_BEST_MODEL_PATH.exists() and self.config.BACKEND_MODEL_INFO_PATH.exists()
        print("\nDeployment:")
        print("  Correcto" if deployment_ok else "  Incompleto")
        
        print("\nBackend:")
        print("  Disponible")
        
        total_duration = self.summary.format_duration(self.summary.get_total_duration())
        print(f"\nTiempo total: {total_duration}")
        
        print("=" * 80)
        
        # Imprimir advertencias de archivos faltantes
        if warnings:
            print("\nADVERTENCIAS: Faltan los siguientes archivos:")
            for w in warnings:
                print(f"  - {w}")
                
        # Log final
        self.logger.info("=" * 80)
        self.logger.info("PIPELINE MAESTRO FINALIZADO")
        self.logger.info(f"Tiempo total: {total_duration}")
        self.logger.info("=" * 80)
