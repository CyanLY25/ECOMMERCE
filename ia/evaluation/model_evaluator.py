import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class ModelEvaluator:
    """
    Clase para evaluar el rendimiento de modelos de predicción de demanda.
    Calcula métricas y compara múltiples modelos.
    """

    def __init__(self, config):
        """
        Inicializa el evaluador con la configuración.
        
        Args:
            config: Objeto de configuración.
        """
        self.config = config

    def calculate_mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calcula el Error Absoluto Medio (MAE).
        
        Args:
            y_true: Valores reales.
            y_pred: Valores predichos.
            
        Returns:
            Valor del MAE.
        """
        raise NotImplementedError("Método calculate_mae() no implementado aún.")

    def calculate_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calcula la Raíz del Error Cuadrático Medio (RMSE).
        
        Args:
            y_true: Valores reales.
            y_pred: Valores predichos.
            
        Returns:
            Valor del RMSE.
        """
        raise NotImplementedError("Método calculate_rmse() no implementado aún.")

    def calculate_r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calcula el Coeficiente de Determinación (R²).
        
        Args:
            y_true: Valores reales.
            y_pred: Valores predichos.
            
        Returns:
            Valor del R².
        """
        raise NotImplementedError("Método calculate_r2() no implementado aún.")

    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Evalúa un modelo calculando todas las métricas disponibles.
        
        Args:
            y_true: Valores reales.
            y_pred: Valores predichos.
            
        Returns:
            Diccionario con todas las métricas calculadas.
        """
        raise NotImplementedError("Método evaluate_model() no implementado aún.")

    def compare_models(self, models_metrics: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Compara múltiples modelos usando sus métricas.
        
        Args:
            models_metrics: Diccionario con métricas de cada modelo.
                Formato: {"modelo1": {"mae": 0.5, ...}, "modelo2": {...}}
                
        Returns:
            DataFrame con la comparación de modelos.
        """
        raise NotImplementedError("Método compare_models() no implementado aún.")

    def save_metrics(self, metrics: Dict[str, float], model_name: str, 
                     output_path: Optional[Path] = None) -> None:
        """
        Guarda las métricas de un modelo en un archivo.
        
        Args:
            metrics: Diccionario con las métricas.
            model_name: Nombre del modelo.
            output_path: Ruta donde guardar las métricas. Si es None, usa config.
        """
        raise NotImplementedError("Método save_metrics() no implementado aún.")
