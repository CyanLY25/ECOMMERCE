import time
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict, Any, Tuple
from ia.config.config import AIConfig
from ia.training.common import (
    set_seed,
    load_datasets,
    separate_features_target,
    get_callbacks,
    calculate_metrics,
    save_metrics,
    save_history,
    plot_training_history,
    get_optimizer,
    get_activation
)
from ia.utils.logger import setup_logger


class MLPModel:
    """
    Clase para implementar y entrenar un modelo de Perceptrón Multicapa (MLP).
    """
    
    def __init__(self, config: AIConfig):
        """
        Inicializa el modelo MLP.
        
        Args:
            config: Objeto de configuración.
        """
        self.config = config
        self.logger = setup_logger("mlp", config.MLP_LOG_PATH)
        self.model: tf.keras.Model = None
        
    def build_model(self, input_shape: Tuple[int]) -> tf.keras.Model:
        """
        Construye la arquitectura del modelo MLP.
        
        Args:
            input_shape: Forma de los datos de entrada.
            
        Returns:
            Modelo Keras compilado.
        """
        # Limpiar sesión Keras
        tf.keras.backend.clear_session()
        
        inputs = tf.keras.Input(shape=input_shape)
        x = inputs
        
        # Capas ocultas
        for units in self.config.MLP_LAYERS:
            x = tf.keras.layers.Dense(units)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation(get_activation(self.config))(x)
            x = tf.keras.layers.Dropout(self.config.DROPOUT_RATE)(x)
            
        # Capa de salida (regresión)
        outputs = tf.keras.layers.Dense(1)(x)
        
        # Construir y compilar el modelo
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        optimizer = get_optimizer(self.config)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'], run_eagerly=False)
        
        self.model = model
        self.logger.info("Arquitectura MLP construida exitosamente")
        model.summary(print_fn=self.logger.info)
        
        return model
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, float], float]:
        """
        Entrena el modelo MLP.
        
        Args:
            X_train: Datos de entrenamiento.
            y_train: Etiquetas de entrenamiento.
            X_val: Datos de validación.
            y_val: Etiquetas de validación.
            
        Returns:
            Tupla con (historial_entrenamiento, métricas, tiempo_entrenamiento).
        """
        set_seed(self.config)
        self.logger.info("=" * 60)
        self.logger.info("INICIANDO ENTRENAMIENTO MLP")
        self.logger.info("=" * 60)
        
        # Construir modelo
        self.build_model(X_train.shape[1:])
        
        # Obtener callbacks
        callbacks = get_callbacks(self.config, "mlp")
        
        # Entrenar
        start_time = time.time()
        history = self.model.fit(
            X_train, y_train,
            epochs=self.config.EPOCHS,
            batch_size=self.config.BATCH_SIZE,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        end_time = time.time()
        training_time = end_time - start_time
        
        # Cargar el mejor modelo
        self.model = tf.keras.models.load_model(self.config.MLP_MODEL_PATH)
        
        # Evaluar
        self.logger.info("Evaluando modelo MLP...")
        y_pred = self.model.predict(X_val, verbose=0).flatten()
        metrics = calculate_metrics(y_val, y_pred)
        metrics['loss'] = float(self.model.evaluate(X_val, y_val, verbose=0)[0])
        metrics['val_loss'] = float(history.history['val_loss'][-1])
        metrics['training_time'] = float(training_time)
        metrics['epochs_run'] = int(len(history.epoch))
        
        # Guardar
        save_history(history, self.config.MLP_HISTORY_PATH)
        save_metrics(metrics, self.config.MLP_METRICS_PATH)
        plot_training_history(history.history, self.config.FIGURES_DIR, "mlp")
        
        self.logger.info(f"MLP entrenado en {training_time:.2f} segundos")
        self.logger.info(f"Métricas: {metrics}")
        self.logger.info("=" * 60)
        
        return history.history, metrics, training_time


def run_mlp(config: AIConfig):
    """
    Función principal para ejecutar el entrenamiento del MLP.
    
    Args:
        config: Objeto de configuración.
    """
    # Cargar datos
    train_df, val_df, test_df = load_datasets(config)
    
    # Separar características y objetivo
    X_train, y_train = separate_features_target(train_df, config.TARGET_VARIABLE)
    X_val, y_val = separate_features_target(val_df, config.TARGET_VARIABLE)
    X_test, y_test = separate_features_target(test_df, config.TARGET_VARIABLE)
    
    # Entrenar
    mlp = MLPModel(config)
    history, metrics, training_time = mlp.train(X_train, y_train, X_val, y_val)
    
    return metrics
