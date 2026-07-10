"""
Módulo de Hyperparameter Tuning usando keras_tuner.
"""
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Tuple
from abc import ABC, abstractmethod

# Verificar si keras_tuner está disponible
try:
    import keras_tuner
    KERAS_TUNER_AVAILABLE = True
except ImportError:
    KERAS_TUNER_AVAILABLE = False

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from ia.config.config import AIConfig
from ia.training.common import (
    set_seed,
    load_datasets,
    separate_features_target,
    create_sequences,
    get_optimizer
)
from ia.utils.logger import setup_logger

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100


class HyperparameterTuner:
    """
    Clase genérica para ejecutar Hyperparameter Tuning usando keras_tuner.
    Compatible con todos los modelos del proyecto.
    """
    
    def __init__(self, config: AIConfig):
        """
        Inicializa el tuner.
        
        Args:
            config: Objeto de configuración del proyecto.
        """
        self.config = config
        self.logger = setup_logger("hyperparameter_tuning", config.TUNING_LOG_PATH)
        self.all_results = {}
        self.trials_data = []
        
        # Verificar dependencias
        if not KERAS_TUNER_AVAILABLE:
            error_msg = (
                "keras_tuner no está instalado. "
                "Por favor, instálalo con: pip install keras-tuner"
            )
            self.logger.error(error_msg)
            print(error_msg)
            sys.exit(1)
    
    def _build_mlp_model(self, hp):
        """
        Construye un modelo MLP con hiperparámetros variables.
        """
        inputs = tf.keras.Input(shape=(self.input_dim,))
        x = inputs
        
        # Número de neuronas (una capa oculta, optimizada)
        units = hp.Choice('neurons', values=[32, 64, 128, 256])
        x = tf.keras.layers.Dense(units)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        dropout = hp.Choice('dropout', values=[0.2, 0.3, 0.4, 0.5])
        x = tf.keras.layers.Dropout(dropout)(x)
        
        # Capa de salida
        outputs = tf.keras.layers.Dense(1)(x)
        
        # Compilar
        learning_rate = hp.Choice('learning_rate', values=[0.01, 0.001, 0.0001])
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _build_lstm_model(self, hp):
        """
        Construye un modelo LSTM con hiperparámetros variables.
        """
        inputs = tf.keras.Input(shape=(self.window_size, self.input_dim))
        x = inputs
        
        # LSTM
        units = hp.Choice('lstm_units', values=[32, 64, 128])
        x = tf.keras.layers.LSTM(units, return_sequences=False)(x)
        dropout = hp.Choice('dropout', values=[0.2, 0.3, 0.5])
        x = tf.keras.layers.Dropout(dropout)(x)
        
        # Capa de salida
        outputs = tf.keras.layers.Dense(1)(x)
        
        # Compilar
        learning_rate = hp.Choice('learning_rate', values=[0.001, 0.0001])
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _build_gru_model(self, hp):
        """
        Construye un modelo GRU con hiperparámetros variables.
        """
        inputs = tf.keras.Input(shape=(self.window_size, self.input_dim))
        x = inputs
        
        # GRU
        units = hp.Choice('gru_units', values=[32, 64, 128])
        x = tf.keras.layers.GRU(units, return_sequences=False)(x)
        dropout = hp.Choice('dropout', values=[0.2, 0.3, 0.5])
        x = tf.keras.layers.Dropout(dropout)(x)
        
        # Capa de salida
        outputs = tf.keras.layers.Dense(1)(x)
        
        # Compilar
        learning_rate = hp.Choice('learning_rate', values=[0.001, 0.0001])
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _build_cnn_lstm_model(self, hp):
        """
        Construye un modelo CNN-LSTM con hiperparámetros variables.
        """
        inputs = tf.keras.Input(shape=(self.window_size, self.input_dim))
        x = inputs
        
        # CNN
        filters = hp.Choice('cnn_filters', values=[32, 64])
        kernel_size = hp.Choice('kernel_size', values=[2, 3, 5])
        x = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu')(x)
        x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
        
        # LSTM
        lstm_units = hp.Choice('lstm_units', values=[32, 64, 128])
        x = tf.keras.layers.LSTM(lstm_units)(x)
        dropout = hp.Choice('dropout', values=[0.2, 0.3, 0.5])
        x = tf.keras.layers.Dropout(dropout)(x)
        
        # Capa de salida
        outputs = tf.keras.layers.Dense(1)(x)
        
        # Compilar
        learning_rate = hp.Choice('learning_rate', values=[0.001, 0.0001])
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _build_cnn_gru_model(self, hp):
        """
        Construye un modelo CNN-GRU con hiperparámetros variables.
        """
        inputs = tf.keras.Input(shape=(self.window_size, self.input_dim))
        x = inputs
        
        # CNN
        filters = hp.Choice('cnn_filters', values=[32, 64])
        kernel_size = hp.Choice('kernel_size', values=[2, 3, 5])
        x = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu')(x)
        x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
        
        # GRU
        gru_units = hp.Choice('gru_units', values=[32, 64, 128])
        x = tf.keras.layers.GRU(gru_units)(x)
        dropout = hp.Choice('dropout', values=[0.2, 0.3, 0.5])
        x = tf.keras.layers.Dropout(dropout)(x)
        
        # Capa de salida
        outputs = tf.keras.layers.Dense(1)(x)
        
        # Compilar
        learning_rate = hp.Choice('learning_rate', values=[0.001, 0.0001])
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _get_model_builder(self, model_name: str):
        """
        Obtiene la función de construcción del modelo.
        """
        builders = {
            'mlp': self._build_mlp_model,
            'lstm': self._build_lstm_model,
            'gru': self._build_gru_model,
            'cnn_lstm': self._build_cnn_lstm_model,
            'cnn_gru': self._build_cnn_gru_model
        }
        
        if model_name not in builders:
            raise ValueError(f"Modelo '{model_name}' no reconocido")
            
        return builders[model_name]
    
    def tune(self, model_name: str, X_train, y_train, X_val, y_val):
        """
        Ejecuta el tuning de hiperparámetros para un modelo específico.
        
        Args:
            model_name: Nombre del modelo.
            X_train: Datos de entrenamiento.
            y_train: Etiquetas de entrenamiento.
            X_val: Datos de validación.
            y_val: Etiquetas de validación.
            
        Returns:
            Diccionario con los mejores parámetros y score.
        """
        set_seed(self.config)
        start_time = time.time()
        
        self.logger.info("=" * 80)
        self.logger.info(f"INICIANDO TUNING PARA MODELO: {model_name.upper()}")
        self.logger.info("=" * 80)
        
        print("\n" + "=" * 80)
        print(f"Modelo: {model_name.upper()}")
        print("=" * 80)
        
        # Preparar datos
        self.input_dim = X_train.shape[1] if model_name == 'mlp' else X_train.shape[2]
        self.window_size = (
            self.config.LSTM_SEQUENCE_LENGTH if model_name == 'lstm'
            else self.config.GRU_SEQUENCE_LENGTH if model_name == 'gru'
            else self.config.CNN_LSTM_SEQUENCE_LENGTH if model_name == 'cnn_lstm'
            else 10
        )
        
        # Obtener builder
        model_builder = self._get_model_builder(model_name)
        
        # Directorio para el tuner
        project_dir = self.config.TUNING_DIR / model_name
        
        # Crear tuner (RandomSearch)
        tuner = keras_tuner.RandomSearch(
            model_builder,
            objective=self.config.OBJECTIVE,
            max_trials=self.config.MAX_TRIALS,
            executions_per_trial=self.config.EXECUTIONS_PER_TRIAL,
            directory=str(project_dir),
            project_name=f"{model_name}_tuning",
            overwrite=self.config.OVERWRITE_TUNING,
            seed=self.config.TUNING_RANDOM_SEED
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=0)
        ]
        
        # Ejecutar búsqueda
        print("\nComenzando búsqueda...")
        for trial_idx in range(self.config.MAX_TRIALS):
            print(f"Trial {trial_idx + 1}/{self.config.MAX_TRIALS}")
        
        tuner.search(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config.EPOCHS,
            batch_size=32,
            callbacks=callbacks,
            verbose=0
        )
        
        # Obtener mejores resultados
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        best_model = tuner.get_best_models(num_models=1)[0]
        
        # Evaluar mejor modelo
        val_loss, val_mae = best_model.evaluate(X_val, y_val, verbose=0)
        
        # Preparar resultados
        result = {
            'best_parameters': best_hps.values,
            'best_score': val_loss
        }
        
        self.all_results[model_name.upper()] = result
        
        # Guardar datos de trials para visualización
        for trial in tuner.oracle.get_best_trials(self.config.MAX_TRIALS):
            trial_entry = {
                'model': model_name,
                'trial_id': trial.trial_id,
                'score': trial.score,
                **trial.hyperparameters.values
            }
            self.trials_data.append(trial_entry)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Mostrar resultados
        self.logger.info(f"\nMejor configuración encontrada para {model_name.upper()}:")
        self.logger.info(json.dumps(best_hps.values, indent=4))
        self.logger.info(f"Mejor Loss: {val_loss:.6f}")
        self.logger.info(f"Tiempo total: {total_time:.2f} segundos")
        
        print(f"\nMejor configuración encontrada:")
        print(json.dumps(best_hps.values, indent=4))
        print(f"Mejor Loss: {val_loss:.6f}")
        print(f"Tiempo total: {total_time:.2f} segundos")
        
        return result
    
    def save_results(self):
        """
        Guarda los resultados del tuning en archivos JSON y CSV.
        """
        # Guardar JSON
        with open(self.config.TUNING_RESULTS_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.all_results, f, indent=4, ensure_ascii=False)
        
        # Guardar CSV de trials
        if self.trials_data:
            trials_df = pd.DataFrame(self.trials_data)
            trials_df.to_csv(self.config.TUNING_RESULTS_CSV_PATH, index=False, encoding='utf-8')
        
        # Guardar archivo con las mejores configuraciones
        best_configs = {k: v['best_parameters'] for k, v in self.all_results.items()}
        with open(self.config.TUNING_BEST_CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(best_configs, f, indent=4, ensure_ascii=False)
        
        self.logger.info("Resultados guardados exitosamente!")
    
    def generate_plots(self):
        """
        Genera gráficas de los resultados del tuning.
        """
        if not self.trials_data:
            self.logger.warning("No hay datos de trials para generar gráficas")
            return
        
        df = pd.DataFrame(self.trials_data)
        
        # Gráfica 1: Loss por trial para cada modelo
        plt.figure(figsize=(14, 7))
        for model in df['model'].unique():
            model_df = df[df['model'] == model].sort_values('trial_id')
            plt.plot(model_df['trial_id'], model_df['score'], 
                    marker='o', label=model, linewidth=2, markersize=6)
        
        plt.title('Evolución del Loss durante el Tuning', fontsize=14)
        plt.xlabel('Trial', fontsize=12)
        plt.ylabel('Validation Loss', fontsize=12)
        plt.legend(title='Modelo', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.config.FIGURES_DIR / 'tuning_loss.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Gráfica 2: Comparación de mejor loss por modelo
        plt.figure(figsize=(12, 6))
        best_scores = []
        model_names = []
        for model_name, result in self.all_results.items():
            best_scores.append(result['best_score'])
            model_names.append(model_name)
        
        sns.barplot(x=model_names, y=best_scores, palette='viridis', alpha=0.8)
        plt.title('Comparación del Mejor Loss por Modelo', fontsize=14)
        plt.xlabel('Modelo', fontsize=12)
        plt.ylabel('Mejor Validation Loss', fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(self.config.FIGURES_DIR / 'tuning_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("Gráficas generadas exitosamente!")
