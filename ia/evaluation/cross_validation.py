import time
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import tensorflow as tf
from tensorflow.keras.models import clone_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from ia.config.config import AIConfig
from ia.training.common import (
    set_seed, load_datasets, separate_features_target, get_optimizer,
    get_activation, calculate_metrics, save_metrics
)
from ia.training.lstm import create_sequences
from ia.utils.logger import setup_logger

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100


class CrossValidationRunner:
    """
    Clase para ejecutar Cross Validation para todos los modelos.
    """
    
    def __init__(self, config: AIConfig):
        """
        Inicializa el ejecutor de Cross Validation.
        
        Args:
            config: Objeto de configuración.
        """
        self.config = config
        self.logger = setup_logger("cross_validation", config.CV_LOG_PATH)
        self.all_results: List[Dict[str, Any]] = []
        
        # Asegurar directorios existan
        config.ensure_directories_exist()
        
    def build_mlp_model(self, input_dim: int, params: Dict[str, Any]) -> tf.keras.Model:
        """
        Construye un modelo MLP con parámetros dados.
        
        Args:
            input_dim: Dimensión de entrada.
            params: Diccionario con hiperparámetros.
            
        Returns:
            Modelo MLP compilado.
        """
        inputs = tf.keras.Input(shape=(input_dim,))
        x = inputs
        
        for units in params.get("layers", self.config.MLP_LAYERS):
            x = tf.keras.layers.Dense(units)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            activation = params.get("activation", self.config.ACTIVATION_FUNCTION)
            if activation == self.config.ACTIVATION_FUNCTION.LEAKY_RELU:
                x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
            elif activation == self.config.ACTIVATION_FUNCTION.ELU:
                x = tf.keras.layers.ELU(alpha=1.0)(x)
            else:
                x = tf.keras.layers.Activation(activation.value)(x)
            x = tf.keras.layers.Dropout(params.get("dropout", self.config.DROPOUT_RATE))(x)
            
        outputs = tf.keras.layers.Dense(1)(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        optimizer = get_optimizer(self.config)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
        
        return model
        
    def build_lstm_model(self, input_shape: Tuple[int, int], params: Dict[str, Any]) -> tf.keras.Model:
        """
        Construye un modelo LSTM.
        
        Args:
            input_shape: Forma de entrada (seq_len, n_features).
            params: Diccionario con hiperparámetros.
            
        Returns:
            Modelo LSTM compilado.
        """
        inputs = tf.keras.Input(shape=input_shape)
        x = inputs
        
        lstm_units = params.get("lstm_units", self.config.LSTM_UNITS)
        for i, units in enumerate(lstm_units):
            return_sequences = i < len(lstm_units) - 1
            x = tf.keras.layers.LSTM(
                units, 
                return_sequences=return_sequences, 
                dropout=params.get("dropout", self.config.DROPOUT_RATE)
            )(x)
            
        outputs = tf.keras.layers.Dense(1)(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        optimizer = get_optimizer(self.config)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
        
        return model
        
    def build_gru_model(self, input_shape: Tuple[int, int], params: Dict[str, Any]) -> tf.keras.Model:
        """
        Construye un modelo GRU.
        
        Args:
            input_shape: Forma de entrada (seq_len, n_features).
            params: Diccionario con hiperparámetros.
            
        Returns:
            Modelo GRU compilado.
        """
        inputs = tf.keras.Input(shape=input_shape)
        x = inputs
        
        gru_units = params.get("gru_units", self.config.GRU_UNITS)
        for i, units in enumerate(gru_units):
            return_sequences = i < len(gru_units) - 1
            x = tf.keras.layers.GRU(
                units, 
                return_sequences=return_sequences, 
                dropout=params.get("dropout", self.config.DROPOUT_RATE)
            )(x)
            
        outputs = tf.keras.layers.Dense(1)(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        optimizer = get_optimizer(self.config)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
        
        return model
        
    def build_cnn_lstm_model(self, input_shape: Tuple[int, int], params: Dict[str, Any]) -> tf.keras.Model:
        """
        Construye un modelo CNN-LSTM.
        
        Args:
            input_shape: Forma de entrada (seq_len, n_features).
            params: Diccionario con hiperparámetros.
            
        Returns:
            Modelo CNN-LSTM compilado.
        """
        inputs = tf.keras.Input(shape=input_shape)
        x = inputs
        
        filters = params.get("filters", self.config.CNN_LSTM_FILTERS)
        kernel_size = params.get("kernel_size", self.config.CNN_LSTM_KERNEL_SIZE)
        pool_size = params.get("pool_size", self.config.CNN_LSTM_POOL_SIZE)
        lstm_units = params.get("lstm_units", self.config.CNN_LSTM_LSTM_UNITS)
        
        for f in filters:
            x = tf.keras.layers.Conv1D(filters=f, kernel_size=kernel_size, padding="same")(x)
            activation = params.get("activation", self.config.ACTIVATION_FUNCTION)
            if activation == self.config.ACTIVATION_FUNCTION.LEAKY_RELU:
                x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
            elif activation == self.config.ACTIVATION_FUNCTION.ELU:
                x = tf.keras.layers.ELU(alpha=1.0)(x)
            else:
                x = tf.keras.layers.Activation(activation.value)(x)
            x = tf.keras.layers.MaxPooling1D(pool_size=pool_size, padding="same")(x)
            
        x = tf.keras.layers.LSTM(lstm_units, dropout=params.get("dropout", self.config.DROPOUT_RATE))(x)
        outputs = tf.keras.layers.Dense(1)(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        optimizer = get_optimizer(self.config)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
        
        return model
        
    def build_cnn_gru_model(self, input_shape: Tuple[int, int], params: Dict[str, Any]) -> tf.keras.Model:
        """
        Construye un modelo CNN-GRU.
        
        Args:
            input_shape: Forma de entrada (seq_len, n_features).
            params: Diccionario con hiperparámetros.
            
        Returns:
            Modelo CNN-GRU compilado.
        """
        inputs = tf.keras.Input(shape=input_shape)
        x = inputs
        
        filters = params.get("filters", self.config.CNN_GRU_FILTERS)
        kernel_size = params.get("kernel_size", self.config.CNN_GRU_KERNEL_SIZE)
        pool_size = params.get("pool_size", self.config.CNN_GRU_POOL_SIZE)
        gru_units = params.get("gru_units", self.config.CNN_GRU_GRU_UNITS)
        
        for f in filters:
            x = tf.keras.layers.Conv1D(filters=f, kernel_size=kernel_size, padding="same")(x)
            activation = params.get("activation", self.config.ACTIVATION_FUNCTION)
            if activation == self.config.ACTIVATION_FUNCTION.LEAKY_RELU:
                x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
            elif activation == self.config.ACTIVATION_FUNCTION.ELU:
                x = tf.keras.layers.ELU(alpha=1.0)(x)
            else:
                x = tf.keras.layers.Activation(activation.value)(x)
            x = tf.keras.layers.MaxPooling1D(pool_size=pool_size, padding="same")(x)
            
        x = tf.keras.layers.GRU(gru_units, dropout=params.get("dropout", self.config.DROPOUT_RATE))(x)
        outputs = tf.keras.layers.Dense(1)(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        optimizer = get_optimizer(self.config)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
        
        return model
        
    def run_single_model_cv(
        self, 
        model_name: str, 
        X: np.ndarray, 
        y: np.ndarray, 
        params: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Ejecuta Cross Validation para un solo modelo.
        
        Args:
            model_name: Nombre del modelo.
            X: Datos de entrada.
            y: Valores objetivo.
            params: Diccionario con hiperparámetros.
            
        Returns:
            Lista de resultados por fold.
        """
        if params is None:
            params = {}
            
        self.logger.info(f"=== Iniciando Cross Validation para {model_name} ===")
        set_seed(self.config)
        
        kf = KFold(n_splits=self.config.CV_NUM_FOLDS, shuffle=True, random_state=self.config.RANDOM_SEED)
        fold_results = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            self.logger.info(f"--- Fold {fold_idx + 1}/{self.config.CV_NUM_FOLDS} ---")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Preparar datos para modelos secuenciales
            if model_name in ["lstm", "gru", "cnn_lstm", "cnn_gru"]:
                seq_len = params.get("seq_length", 
                                     self.config.LSTM_SEQUENCE_LENGTH if model_name == "lstm" else
                                     self.config.GRU_SEQUENCE_LENGTH if model_name == "gru" else
                                     self.config.CNN_LSTM_SEQUENCE_LENGTH if model_name == "cnn_lstm" else
                                     self.config.CNN_GRU_SEQUENCE_LENGTH)
                X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_len)
                X_val_seq, y_val_seq = create_sequences(X_val, y_val, seq_len)
            else:
                X_train_seq, y_train_seq = X_train, y_train
                X_val_seq, y_val_seq = X_val, y_val
                
            # Construir modelo
            start_time = time.time()
            if model_name == "mlp":
                model = self.build_mlp_model(X_train_seq.shape[1], params)
            elif model_name == "lstm":
                model = self.build_lstm_model(X_train_seq.shape[1:], params)
            elif model_name == "gru":
                model = self.build_gru_model(X_train_seq.shape[1:], params)
            elif model_name == "cnn_lstm":
                model = self.build_cnn_lstm_model(X_train_seq.shape[1:], params)
            elif model_name == "cnn_gru":
                model = self.build_cnn_gru_model(X_train_seq.shape[1:], params)
                
            # Callbacks
            callbacks = [
                EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=0),
                ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=0)
            ]
            
            # Entrenar
            history = model.fit(
                X_train_seq, y_train_seq,
                epochs=params.get("epochs", self.config.EPOCHS),
                batch_size=params.get("batch_size", self.config.BATCH_SIZE),
                validation_data=(X_val_seq, y_val_seq),
                callbacks=callbacks,
                verbose=0
            )
            
            # Evaluar
            y_pred = model.predict(X_val_seq, verbose=0).flatten()
            val_loss = history.history["val_loss"][-1]
            train_loss = history.history["loss"][-1]
            
            metrics = calculate_metrics(y_val_seq, y_pred)
            end_time = time.time()
            fold_time = end_time - start_time
            
            # Guardar resultados
            fold_result = {
                "model": model_name,
                "fold": fold_idx + 1,
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "mape": metrics["mape"],
                "r2": metrics["r2"],
                "loss": train_loss,
                "val_loss": val_loss,
                "time": fold_time,
                "epochs_used": len(history.history["loss"])
            }
            fold_results.append(fold_result)
            self.all_results.append(fold_result)
            self.logger.info(f"Fold {fold_idx+1}: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}, Tiempo={fold_time:.2f}s")
            
        return fold_results
        
    def calculate_statistics(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula estadísticas agregadas por modelo.
        
        Args:
            results_df: DataFrame con resultados por fold.
            
        Returns:
            DataFrame con estadísticas.
        """
        stats_list = []
        
        for model_name in results_df["model"].unique():
            model_data = results_df[results_df["model"] == model_name]
            
            for metric in ["mae", "rmse", "mape", "r2", "loss", "time"]:
                metric_values = model_data[metric].values
                mean_val = np.mean(metric_values)
                std_val = np.std(metric_values)
                min_val = np.min(metric_values)
                max_val = np.max(metric_values)
                
                # Intervalo de confianza del 95%
                ci_low, ci_high = stats.t.interval(
                    confidence=0.95,
                    df=len(metric_values)-1,
                    loc=mean_val,
                    scale=stats.sem(metric_values)
                )
                
                stats_list.append({
                    "model": model_name,
                    "metric": metric,
                    "mean": mean_val,
                    "std": std_val,
                    "min": min_val,
                    "max": max_val,
                    "ci_low": ci_low,
                    "ci_high": ci_high
                })
                
        stats_df = pd.DataFrame(stats_list)
        return stats_df
        
    def plot_results(self, results_df: pd.DataFrame):
        """
        Genera gráficos de resultados.
        
        Args:
            results_df: DataFrame con resultados por fold.
        """
        # Boxplot RMSE
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=results_df, x="model", y="rmse", palette="viridis")
        plt.title("Distribución de RMSE por Modelo (Cross Validation)", fontsize=14)
        plt.xlabel("Modelo", fontsize=12)
        plt.ylabel("RMSE", fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.config.FIGURES_DIR / "cv_rmse_boxplot.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        # Boxplot MAE
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=results_df, x="model", y="mae", palette="magma")
        plt.title("Distribución de MAE por Modelo (Cross Validation)", fontsize=14)
        plt.xlabel("Modelo", fontsize=12)
        plt.ylabel("MAE", fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.config.FIGURES_DIR / "cv_mae_boxplot.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        # Boxplot R²
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=results_df, x="model", y="r2", palette="coolwarm")
        plt.title("Distribución de R² por Modelo (Cross Validation)", fontsize=14)
        plt.xlabel("Modelo", fontsize=12)
        plt.ylabel("R²", fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.config.FIGURES_DIR / "cv_r2_boxplot.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        # Boxplot Tiempo
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=results_df, x="model", y="time", palette="plasma")
        plt.title("Distribución de Tiempo de Entrenamiento por Modelo (Cross Validation)", fontsize=14)
        plt.xlabel("Modelo", fontsize=12)
        plt.ylabel("Tiempo (segundos)", fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.config.FIGURES_DIR / "cv_time_boxplot.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        # Comparación por Fold
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        sns.lineplot(data=results_df, x="fold", y="rmse", hue="model", marker="o", ax=axes[0,0])
        axes[0,0].set_title("RMSE por Fold", fontsize=12)
        axes[0,0].legend(bbox_to_anchor=(1.05,1), loc='upper left')
        
        sns.lineplot(data=results_df, x="fold", y="mae", hue="model", marker="o", ax=axes[0,1])
        axes[0,1].set_title("MAE por Fold", fontsize=12)
        axes[0,1].legend(bbox_to_anchor=(1.05,1), loc='upper left')
        
        sns.lineplot(data=results_df, x="fold", y="r2", hue="model", marker="o", ax=axes[1,0])
        axes[1,0].set_title("R² por Fold", fontsize=12)
        axes[1,0].legend(bbox_to_anchor=(1.05,1), loc='upper left')
        
        sns.lineplot(data=results_df, x="fold", y="time", hue="model", marker="o", ax=axes[1,1])
        axes[1,1].set_title("Tiempo por Fold", fontsize=12)
        axes[1,1].legend(bbox_to_anchor=(1.05,1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(self.config.FIGURES_DIR / "cv_comparison_by_fold.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        self.logger.info("Gráficos de Cross Validation guardados")
        
    def run_all(self):
        """
        Ejecuta Cross Validation para todos los modelos.
        """
        self.logger.info("="*80)
        self.logger.info("INICIANDO CROSS VALIDATION COMPLETO")
        self.logger.info("="*80)
        
        # Cargar datos
        self.logger.info("Cargando datos...")
        train_df, _, _ = load_datasets(self.config)
        X, y = separate_features_target(train_df, self.config.TARGET_VARIABLE)
        
        # Ejecutar CV para cada modelo
        model_names = ["mlp", "lstm", "gru", "cnn_lstm", "cnn_gru"]
        for model_name in model_names:
            self.run_single_model_cv(model_name, X, y)
            
        # Guardar resultados
        results_df = pd.DataFrame(self.all_results)
        results_df.to_csv(self.config.CV_RESULTS_PATH, index=False)
        self.logger.info(f"Resultados de Cross Validation guardados en {self.config.CV_RESULTS_PATH}")
        
        # Calcular y guardar estadísticas
        stats_df = self.calculate_statistics(results_df)
        stats_df.to_csv(self.config.CV_STATISTICS_PATH, index=False)
        self.logger.info(f"Estadísticas de Cross Validation guardadas en {self.config.CV_STATISTICS_PATH}")
        
        # Generar gráficos
        self.plot_results(results_df)
        
        self.logger.info("="*80)
        self.logger.info("CROSS VALIDATION COMPLETADO")
        self.logger.info("="*80)
        
        return results_df, stats_df
