import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Optional, Any
from sklearn.model_selection import train_test_split


class ModelTrainer:
    """
    Clase para entrenar diferentes tipos de modelos de redes neuronales
    para predicción de demanda de productos.
    """

    def __init__(self, config):
        """
        Inicializa el entrenador de modelos con la configuración.
        
        Args:
            config: Objeto de configuración con parámetros de entrenamiento.
        """
        self.config = config
        tf.random.set_seed(config.RANDOM_SEED)
        np.random.seed(config.RANDOM_SEED)

    def train_mlp(self, X_train: np.ndarray, y_train: np.ndarray, 
                  X_val: Optional[np.ndarray] = None, 
                  y_val: Optional[np.ndarray] = None) -> Any:
        """
        Entrena un modelo de Perceptrón Multicapa (MLP).
        
        Args:
            X_train: Datos de entrenamiento.
            y_train: Etiquetas de entrenamiento.
            X_val: Datos de validación (opcional).
            y_val: Etiquetas de validación (opcional).
            
        Returns:
            Modelo entrenado.
        """
        raise NotImplementedError("Método train_mlp() no implementado aún.")

    def train_lstm(self, X_train: np.ndarray, y_train: np.ndarray, 
                   X_val: Optional[np.ndarray] = None, 
                   y_val: Optional[np.ndarray] = None) -> Any:
        """
        Entrena un modelo de Long Short-Term Memory (LSTM).
        
        Args:
            X_train: Datos de entrenamiento en formato de secuencias.
            y_train: Etiquetas de entrenamiento.
            X_val: Datos de validación (opcional).
            y_val: Etiquetas de validación (opcional).
            
        Returns:
            Modelo entrenado.
        """
        raise NotImplementedError("Método train_lstm() no implementado aún.")

    def train_gru(self, X_train: np.ndarray, y_train: np.ndarray, 
                  X_val: Optional[np.ndarray] = None, 
                  y_val: Optional[np.ndarray] = None) -> Any:
        """
        Entrena un modelo de Gated Recurrent Unit (GRU).
        
        Args:
            X_train: Datos de entrenamiento en formato de secuencias.
            y_train: Etiquetas de entrenamiento.
            X_val: Datos de validación (opcional).
            y_val: Etiquetas de validación (opcional).
            
        Returns:
            Modelo entrenado.
        """
        raise NotImplementedError("Método train_gru() no implementado aún.")

    def train_cnn_lstm(self, X_train: np.ndarray, y_train: np.ndarray, 
                       X_val: Optional[np.ndarray] = None, 
                       y_val: Optional[np.ndarray] = None) -> Any:
        """
        Entrena un modelo híbrido CNN-LSTM.
        
        Args:
            X_train: Datos de entrenamiento en formato de secuencias.
            y_train: Etiquetas de entrenamiento.
            X_val: Datos de validación (opcional).
            y_val: Etiquetas de validación (opcional).
            
        Returns:
            Modelo entrenado.
        """
        raise NotImplementedError("Método train_cnn_lstm() no implementado aún.")

    def train_cnn_gru(self, X_train: np.ndarray, y_train: np.ndarray, 
                      X_val: Optional[np.ndarray] = None, 
                      y_val: Optional[np.ndarray] = None) -> Any:
        """
        Entrena un modelo híbrido CNN-GRU.
        
        Args:
            X_train: Datos de entrenamiento en formato de secuencias.
            y_train: Etiquetas de entrenamiento.
            X_val: Datos de validación (opcional).
            y_val: Etiquetas de validación (opcional).
            
        Returns:
            Modelo entrenado.
        """
        raise NotImplementedError("Método train_cnn_gru() no implementado aún.")

    def save_model(self, model: Any, model_name: str, output_path: Optional[Path] = None) -> None:
        """
        Guarda un modelo entrenado en disco.
        
        Args:
            model: Modelo a guardar.
            model_name: Nombre del modelo para identificarlo.
            output_path: Ruta donde guardar el modelo. Si es None, usa config.
        """
        if output_path is None:
            output_path = self.config.MODELS_DIR / f"{model_name}.h5"
        
        raise NotImplementedError("Método save_model() no implementado aún.")
