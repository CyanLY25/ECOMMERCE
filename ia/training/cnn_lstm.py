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
    create_sequences,
    get_callbacks,
    calculate_metrics,
    save_metrics,
    save_history,
    plot_training_history,
    get_optimizer
)
from ia.utils.logger import setup_logger


class CNNLSTMModel:
    """
    Clase para implementar y entrenar un modelo CNN-LSTM híbrido para predicción de demanda.
    """
    
    def __init__(self, config: AIConfig):
        """
        Inicializa el modelo CNN-LSTM.
        
        Args:
            config: Objeto de configuración.
        """
        self.config = config
        self.logger = setup_logger("cnn_lstm", config.CNN_LSTM_LOG_PATH)
        self.model: tf.keras.Model = None
        
    def build_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """
        Construye la arquitectura del modelo CNN-LSTM.
        
        Args:
            input_shape: Forma de los datos de entrada (window_size, num_features).
            
        Returns:
            Modelo Keras compilado.
        """
        # Limpiar sesión Keras
        tf.keras.backend.clear_session()
        
        inputs = tf.keras.Input(shape=input_shape)
        x = inputs
        
        # Capa Convolucional 1D
        x = tf.keras.layers.Conv1D(
            filters=self.config.CNN_LSTM_FILTERS,
            kernel_size=self.config.CNN_LSTM_KERNEL_SIZE,
            activation='relu'
        )(x)
        
        # Capa Max Pooling 1D
        x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
        
        # Capa LSTM
        x = tf.keras.layers.LSTM(units=self.config.CNN_LSTM_UNITS)(x)
        x = tf.keras.layers.Dropout(self.config.CNN_LSTM_DROPOUT)(x)
        
        # Capa Dense
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        
        # Capa de salida (regresión)
        outputs = tf.keras.layers.Dense(1, activation='linear')(x)
        
        # Construir y compilar el modelo
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        optimizer = get_optimizer(self.config)
        optimizer.learning_rate = self.config.CNN_LSTM_LEARNING_RATE
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'], run_eagerly=False)
        
        self.model = model
        self.logger.info("Arquitectura CNN-LSTM construida exitosamente")
        model.summary(print_fn=self.logger.info)
        
        return model
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, float], float]:
        """
        Entrena el modelo CNN-LSTM.
        
        Args:
            X_train: Datos de entrenamiento (secuencias).
            y_train: Etiquetas de entrenamiento.
            X_val: Datos de validación (secuencias).
            y_val: Etiquetas de validación.
            
        Returns:
            Tupla con (historial_entrenamiento, métricas, tiempo_entrenamiento).
        """
        set_seed(self.config)
        self.logger.info("=" * 60)
        self.logger.info("INICIANDO ENTRENAMIENTO CNN-LSTM")
        self.logger.info("=" * 60)
        
        # Construir modelo
        self.build_model((X_train.shape[1], X_train.shape[2]))
        
        # Obtener callbacks
        callbacks = get_callbacks(self.config, "cnn_lstm")
        
        # Entrenar
        start_time = time.time()
        history = self.model.fit(
            X_train, y_train,
            epochs=self.config.CNN_LSTM_EPOCHS,
            batch_size=self.config.CNN_LSTM_BATCH_SIZE,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        end_time = time.time()
        training_time = end_time - start_time
        
        # Cargar el mejor modelo
        self.model = tf.keras.models.load_model(self.config.CNN_LSTM_MODEL_PATH)
        
        # Evaluar
        self.logger.info("Evaluando modelo CNN-LSTM...")
        y_pred = self.model.predict(X_val, verbose=0).flatten()
        metrics = calculate_metrics(y_val, y_pred)
        metrics['loss'] = float(self.model.evaluate(X_val, y_val, verbose=0)[0])
        metrics['val_loss'] = float(history.history['val_loss'][-1])
        metrics['training_time'] = float(training_time)
        metrics['epochs_run'] = int(len(history.epoch))
        
        # Guardar
        save_history(history, self.config.CNN_LSTM_HISTORY_PATH)
        save_metrics(metrics, self.config.CNN_LSTM_METRICS_PATH)
        plot_training_history(history.history, self.config.FIGURES_DIR, "cnn_lstm")
        
        self.logger.info(f"CNN-LSTM entrenado en {training_time:.2f} segundos")
        self.logger.info(f"Métricas: {metrics}")
        self.logger.info("=" * 60)
        
        return history.history, metrics, training_time
