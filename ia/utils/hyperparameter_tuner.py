import time
import json
import itertools
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from typing import Dict, Any, List, Tuple

from ia.config.config import AIConfig, OptimizerType, ActivationType
from ia.utils.logger import setup_logger
from ia.training.common import (
    set_seed, load_datasets, separate_features_target, calculate_metrics
)
from ia.evaluation.cross_validation import CrossValidationRunner


class HyperparameterTuner:
    """
    Clase para optimizar hiperparámetros de modelos de ML usando Grid Search o Random Search.
    """

    def __init__(self, config: AIConfig):
        """
        Inicializa el optimizador de hiperparámetros con la configuración.
        
        Args:
            config: Objeto de configuración.
        """
        self.config = config
        self.logger = setup_logger("tuning", config.TUNING_LOG_PATH)
        self.all_tuning_results: List[Dict[str, Any]] = []
        self.cv_runner = CrossValidationRunner(config)
        set_seed(config)

    def _generate_param_combinations(self, param_space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """
        Genera todas las combinaciones de hiperparámetros (para Grid Search)
        o una muestra aleatoria (para Random Search).
        
        Args:
            param_space: Espacio de búsqueda de hiperparámetros.
            
        Returns:
            Lista de diccionarios con combinaciones de hiperparámetros.
        """
        keys = list(param_space.keys())
        values = list(param_space.values())
        
        if self.config.TUNING_METHOD == self.config.TUNING_METHOD.GRID_SEARCH:
            combinations = list(itertools.product(*values))
            self.logger.info(f"Grid Search: {len(combinations)} combinaciones a evaluar")
        else:  # Random Search
            total_possible = 1
            for v in values:
                total_possible *= len(v)
            num_samples = min(self.config.TUNING_NUM_ITERATIONS, total_possible)
            self.logger.info(f"Random Search: {num_samples} muestras aleatorias (de {total_possible} posibles)")
            
            combinations = []
            seen = set()
            while len(combinations) < num_samples:
                combo = tuple(random.choice(v) for v in values)
                if combo not in seen:
                    seen.add(combo)
                    combinations.append(combo)
        
        # Convertir a diccionarios
        param_dicts = []
        for combo in combinations:
            param_dict = dict(zip(keys, combo))
            param_dicts.append(param_dict)
            
        return param_dicts

    def _get_param_space(self, model_name: str) -> Dict[str, List[Any]]:
        """
        Obtiene el espacio de búsqueda de hiperparámetros para un modelo específico.
        
        Args:
            model_name: Nombre del modelo.
            
        Returns:
            Diccionario con el espacio de búsqueda.
        """
        base_space = {
            "lr": self.config.TUNING_LR,
            "batch_size": self.config.TUNING_BATCH_SIZE,
            "epochs": self.config.TUNING_EPOCHS,
            "dropout": self.config.TUNING_DROPOUT,
            "optimizer": self.config.TUNING_OPTIMIZERS,
            "activation": self.config.TUNING_ACTIVATIONS
        }
        
        if model_name == "mlp":
            base_space["layers"] = self.config.TUNING_MLP_LAYERS
        elif model_name == "lstm":
            base_space["lstm_units"] = self.config.TUNING_LSTM_UNITS
            base_space["seq_length"] = [8, 10, 12]
        elif model_name == "gru":
            base_space["gru_units"] = self.config.TUNING_GRU_UNITS
            base_space["seq_length"] = [8, 10, 12]
        elif model_name == "cnn_lstm":
            base_space["filters"] = self.config.TUNING_CNN_LSTM_FILTERS
            base_space["lstm_units"] = self.config.TUNING_CNN_LSTM_LSTM_UNITS
            base_space["seq_length"] = [8, 10, 12]
        elif model_name == "cnn_gru":
            base_space["filters"] = self.config.TUNING_CNN_GRU_FILTERS
            base_space["gru_units"] = self.config.TUNING_CNN_GRU_GRU_UNITS
            base_space["seq_length"] = [8, 10, 12]
            
        return base_space

    def _evaluate_params(
        self, 
        model_name: str, 
        params: Dict[str, Any], 
        X: np.ndarray, 
        y: np.ndarray
    ) -> Tuple[Dict[str, Any], float, float, float, float]:
        """
        Evalúa una combinación de hiperparámetros usando Cross Validation.
        
        Args:
            model_name: Nombre del modelo.
            params: Diccionario de hiperparámetros.
            X: Datos de entrada.
            y: Valores objetivo.
            
        Returns:
            Tupla con (resultados CV, promedio RMSE, promedio R², promedio MAE, promedio MAPE).
        """
        # Actualizar config temporalmente con los parámetros
        original_lr = self.config.LEARNING_RATE
        original_batch_size = self.config.BATCH_SIZE
        original_epochs = self.config.EPOCHS
        original_dropout = self.config.DROPOUT_RATE
        original_optimizer = self.config.OPTIMIZER
        original_activation = self.config.ACTIVATION_FUNCTION
        
        self.config.LEARNING_RATE = params["lr"]
        self.config.BATCH_SIZE = params["batch_size"]
        self.config.EPOCHS = params["epochs"]
        self.config.DROPOUT_RATE = params["dropout"]
        self.config.OPTIMIZER = params["optimizer"]
        self.config.ACTIVATION_FUNCTION = params["activation"]
        
        # Ejecutar Cross Validation
        cv_results = self.cv_runner.run_single_model_cv(model_name, X, y, params)
        
        # Restaurar config original
        self.config.LEARNING_RATE = original_lr
        self.config.BATCH_SIZE = original_batch_size
        self.config.EPOCHS = original_epochs
        self.config.DROPOUT_RATE = original_dropout
        self.config.OPTIMIZER = original_optimizer
        self.config.ACTIVATION_FUNCTION = original_activation
        
        # Calcular métricas promedio
        rmse_values = [r["rmse"] for r in cv_results]
        r2_values = [r["r2"] for r in cv_results]
        mae_values = [r["mae"] for r in cv_results]
        mape_values = [r["mape"] for r in cv_results]
        
        avg_rmse = np.mean(rmse_values)
        avg_r2 = np.mean(r2_values)
        avg_mae = np.mean(mae_values)
        avg_mape = np.mean(mape_values)
        
        return cv_results, avg_rmse, avg_r2, avg_mae, avg_mape

    def tune_model(
        self, 
        model_name: str, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Optimiza los hiperparámetros de un solo modelo.
        
        Args:
            model_name: Nombre del modelo.
            X: Datos de entrada.
            y: Valores objetivo.
            
        Returns:
            Tupla con (mejores hiperparámetros, mejores métricas).
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"INICIANDO HYPERPARAMETER TUNING PARA {model_name.upper()}")
        self.logger.info(f"{'='*60}")
        
        param_space = self._get_param_space(model_name)
        param_combinations = self._generate_param_combinations(param_space)
        
        best_rmse = float("inf")
        best_params = None
        best_metrics = None
        
        for i, params in enumerate(param_combinations):
            self.logger.info(f"\n--- Iteración {i+1}/{len(param_combinations)} ---")
            self.logger.info(f"Probando: {params}")
            
            try:
                start_time = time.time()
                cv_results, avg_rmse, avg_r2, avg_mae, avg_mape = self._evaluate_params(model_name, params, X, y)
                total_time = time.time() - start_time
                
                # Guardar resultado
                tuning_result = {
                    "model": model_name,
                    "iteration": i+1,
                    **params,
                    "avg_rmse": avg_rmse,
                    "avg_r2": avg_r2,
                    "avg_mae": avg_mae,
                    "avg_mape": avg_mape,
                    "time": total_time
                }
                self.all_tuning_results.append(tuning_result)
                
                self.logger.info(f"Resultados: RMSE={avg_rmse:.4f}, R²={avg_r2:.4f}, MAE={avg_mae:.4f}, Tiempo={total_time:.2f}s")
                
                # Actualizar mejor modelo
                if avg_rmse < best_rmse:
                    best_rmse = avg_rmse
                    best_params = params.copy()
                    best_metrics = {
                        "rmse": avg_rmse,
                        "r2": avg_r2,
                        "mae": avg_mae,
                        "mape": avg_mape
                    }
                    self.logger.info(f"✅ Nueva mejor configuración! RMSE={best_rmse:.4f}")
                    
            except Exception as e:
                self.logger.error(f"❌ Error en iteración {i+1}: {str(e)}", exc_info=True)
                
        # Guardar resultados del tuning
        tuning_df = pd.DataFrame(self.all_tuning_results)
        tuning_df.to_csv(self.config.TUNING_RESULTS_PATH, index=False)
        self.logger.info(f"\nResultados de tuning guardados en {self.config.TUNING_RESULTS_PATH}")
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"TUNING COMPLETADO PARA {model_name.upper()}")
        self.logger.info(f"Mejores parámetros: {best_params}")
        self.logger.info(f"Mejores métricas: {best_metrics}")
        self.logger.info(f"{'='*60}")
        
        return best_params, best_metrics

    def tune_all_models(self) -> Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]]:
        """
        Optimiza hiperparámetros para todos los modelos.
        
        Returns:
            Diccionario con (mejores params, mejores métricas) para cada modelo.
        """
        self.logger.info("="*80)
        self.logger.info("INICIANDO HYPERPARAMETER TUNING PARA TODOS LOS MODELOS")
        self.logger.info("="*80)
        
        # Cargar datos
        train_df, _, _ = load_datasets(self.config)
        X, y = separate_features_target(train_df, self.config.TARGET_VARIABLE)
        
        all_results = {}
        model_names = ["mlp", "lstm", "gru", "cnn_lstm", "cnn_gru"]
        
        for model_name in model_names:
            best_params, best_metrics = self.tune_model(model_name, X, y)
            all_results[model_name] = (best_params, best_metrics)
            
        # Guardar resultados finales de tuning
        final_tuning_df = pd.DataFrame(self.all_tuning_results)
        final_tuning_df.to_csv(self.config.TUNING_RESULTS_PATH, index=False)
        
        return all_results

    def save_best_config(self, best_config: Dict[str, Any], model_name: str, 
                         output_path: Optional[Path] = None) -> None:
        """
        Guarda la mejor configuración de hiperparámetros en un archivo.
        
        Args:
            best_config: Diccionario con la mejor configuración.
            model_name: Nombre del modelo.
            output_path: Ruta donde guardar la configuración. Si es None, usa config.
        """
        if output_path is None:
            output_path = self.config.MODELS_DIR / f"{model_name}_best_params.json"
            
        with open(output_path, "w") as f:
            json.dump(best_config, f, indent=4)
            
        self.logger.info(f"Mejores parámetros guardados en {output_path}")
