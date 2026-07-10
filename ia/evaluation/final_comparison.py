import time
import json
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Tuple
from datetime import datetime

import tensorflow as tf

from ia.config.config import AIConfig
from ia.utils.logger import setup_logger
from ia.training.common import (
    set_seed, load_datasets, separate_features_target, calculate_metrics
)
from ia.evaluation.cross_validation import CrossValidationRunner
from ia.utils.hyperparameter_tuner import HyperparameterTuner


sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100


class FinalComparator:
    """
    Clase para realizar la comparación final de modelos, seleccionar el mejor
    y prepararlo para producción.
    """
    
    def __init__(self, config: AIConfig):
        """
        Inicializa el comparador final.
        
        Args:
            config: Objeto de configuración.
        """
        self.config = config
        self.logger = setup_logger("final_comparison", config.LOGS_DIR / "final_comparison.log")
        self.cv_runner = CrossValidationRunner(config)
        self.tuner = HyperparameterTuner(config)
        
        config.ensure_directories_exist()
        
    def load_tuning_results(self) -> pd.DataFrame:
        """
        Carga los resultados del Hyperparameter Tuning.
        
        Returns:
            DataFrame con resultados de tuning.
        """
        if not self.config.TUNING_RESULTS_PATH.exists():
            raise FileNotFoundError(f"Resultados de tuning no encontrados: {self.config.TUNING_RESULTS_PATH}")
            
        return pd.read_csv(self.config.TUNING_RESULTS_PATH)
        
    def extract_best_from_tuning(self, tuning_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extrae las mejores métricas por modelo del tuning.
        
        Args:
            tuning_df: DataFrame con resultados de tuning.
            
        Returns:
            DataFrame con una fila por modelo, con las mejores métricas.
        """
        best_rows = []
        
        for model_name in tuning_df["model"].unique():
            model_data = tuning_df[tuning_df["model"] == model_name]
            best_idx = model_data["avg_rmse"].idxmin()
            best_row = model_data.loc[best_idx].copy()
            
            best_rows.append({
                "model": model_name,
                "mae": best_row["avg_mae"],
                "rmse": best_row["avg_rmse"],
                "mape": best_row["avg_mape"],
                "r2": best_row["avg_r2"],
                "time": best_row["time"]
            })
            
        comparison_df = pd.DataFrame(best_rows)
        return comparison_df
        
    def plot_final_comparison(self, comparison_df: pd.DataFrame):
        """
        Genera gráficos de comparación final.
        
        Args:
            comparison_df: DataFrame con comparación de modelos.
        """
        # MAE
        plt.figure(figsize=(12,6))
        sns.barplot(data=comparison_df, x="model", y="mae", palette="viridis")
        plt.title("Comparación de MAE entre Modelos", fontsize=14)
        plt.xlabel("Modelo", fontsize=12)
        plt.ylabel("MAE", fontsize=12)
        plt.tight_layout()
        plt.savefig(self.config.FIGURES_DIR / "final_comparison_mae.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        # RMSE
        plt.figure(figsize=(12,6))
        sns.barplot(data=comparison_df, x="model", y="rmse", palette="magma")
        plt.title("Comparación de RMSE entre Modelos", fontsize=14)
        plt.xlabel("Modelo", fontsize=12)
        plt.ylabel("RMSE", fontsize=12)
        plt.tight_layout()
        plt.savefig(self.config.FIGURES_DIR / "final_comparison_rmse.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        # MAPE
        plt.figure(figsize=(12,6))
        sns.barplot(data=comparison_df, x="model", y="mape", palette="coolwarm")
        plt.title("Comparación de MAPE entre Modelos", fontsize=14)
        plt.xlabel("Modelo", fontsize=12)
        plt.ylabel("MAPE (%)", fontsize=12)
        plt.tight_layout()
        plt.savefig(self.config.FIGURES_DIR / "final_comparison_mape.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        # R²
        plt.figure(figsize=(12,6))
        sns.barplot(data=comparison_df, x="model", y="r2", palette="plasma")
        plt.title("Comparación de R² entre Modelos", fontsize=14)
        plt.xlabel("Modelo", fontsize=12)
        plt.ylabel("R²", fontsize=12)
        plt.tight_layout()
        plt.savefig(self.config.FIGURES_DIR / "final_comparison_r2.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        # Tiempo
        plt.figure(figsize=(12,6))
        sns.barplot(data=comparison_df, x="model", y="time", palette="Set2")
        plt.title("Comparación de Tiempo de Entrenamiento entre Modelos", fontsize=14)
        plt.xlabel("Modelo", fontsize=12)
        plt.ylabel("Tiempo (segundos)", fontsize=12)
        plt.tight_layout()
        plt.savefig(self.config.FIGURES_DIR / "final_comparison_time.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        self.logger.info("Gráficos de comparación final guardados")
        
    def select_best_model(self, comparison_df: pd.DataFrame) -> Tuple[str, pd.Series]:
        """
        Selecciona el mejor modelo usando los criterios definidos.
        
        Args:
            comparison_df: DataFrame con comparación de modelos.
            
        Returns:
            Tupla con (nombre del mejor modelo, serie con sus métricas).
        """
        # Ordenar por: 1. R² (desc), 2. RMSE (asc), 3. MAE (asc), 4. MAPE (asc), 5. Tiempo (asc)
        sorted_df = comparison_df.sort_values(
            by=["r2", "rmse", "mae", "mape", "time"],
            ascending=[False, True, True, True, True]
        )
        
        best_model_name = sorted_df.iloc[0]["model"]
        best_metrics = sorted_df.iloc[0]
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"MEJOR MODELO SELECCIONADO: {best_model_name.upper()}")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Métricas:\n{best_metrics}")
        
        return best_model_name, best_metrics
        
    def train_final_best_model(
        self, 
        model_name: str, 
        best_params: Dict[str, Any],
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray
    ) -> Tuple[tf.keras.Model, Dict[str, Any]]:
        """
        Entrena el mejor modelo con todos los datos de entrenamiento y validación.
        
        Args:
            model_name: Nombre del modelo.
            best_params: Mejores hiperparámetros.
            X_train, y_train: Datos de entrenamiento.
            X_val, y_val: Datos de validación.
            
        Returns:
            Tupla con (modelo entrenado, métricas finales).
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"ENTRENANDO MODELO FINAL: {model_name.upper()}")
        self.logger.info(f"{'='*60}")
        
        # Configurar hiperparámetros
        original_lr = self.config.LEARNING_RATE
        original_batch_size = self.config.BATCH_SIZE
        original_epochs = self.config.EPOCHS
        original_dropout = self.config.DROPOUT_RATE
        original_optimizer = self.config.OPTIMIZER
        original_activation = self.config.ACTIVATION_FUNCTION
        
        self.config.LEARNING_RATE = best_params["lr"]
        self.config.BATCH_SIZE = best_params["batch_size"]
        self.config.EPOCHS = best_params["epochs"]
        self.config.DROPOUT_RATE = best_params["dropout"]
        self.config.OPTIMIZER = best_params["optimizer"]
        self.config.ACTIVATION_FUNCTION = best_params["activation"]
        
        # Preparar datos (combinar train + val)
        X_full = np.concatenate([X_train, X_val])
        y_full = np.concatenate([y_train, y_val])
        
        if model_name in ["lstm", "gru", "cnn_lstm", "cnn_gru"]:
            seq_len = best_params["seq_length"]
            from ia.training.lstm import create_sequences
            X_full_seq, y_full_seq = create_sequences(X_full, y_full, seq_len)
        else:
            X_full_seq, y_full_seq = X_full, y_full
            
        # Construir y entrenar modelo
        set_seed(self.config)
        
        if model_name == "mlp":
            model = self.cv_runner.build_mlp_model(X_full_seq.shape[1], best_params)
        elif model_name == "lstm":
            model = self.cv_runner.build_lstm_model(X_full_seq.shape[1:], best_params)
        elif model_name == "gru":
            model = self.cv_runner.build_gru_model(X_full_seq.shape[1:], best_params)
        elif model_name == "cnn_lstm":
            model = self.cv_runner.build_cnn_lstm_model(X_full_seq.shape[1:], best_params)
        elif model_name == "cnn_gru":
            model = self.cv_runner.build_cnn_gru_model(X_full_seq.shape[1:], best_params)
            
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="loss", patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=5, min_lr=1e-7),
            tf.keras.callbacks.ModelCheckpoint(
                self.config.MODELS_DIR / f"{model_name}_final.keras",
                monitor="loss",
                save_best_only=True
            )
        ]
        
        start_time = time.time()
        history = model.fit(
            X_full_seq, y_full_seq,
            epochs=self.config.EPOCHS,
            batch_size=self.config.BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        end_time = time.time()
        training_time = end_time - start_time
        
        # Evaluar
        y_pred = model.predict(X_full_seq, verbose=0).flatten()
        final_metrics = calculate_metrics(y_full_seq, y_pred)
        final_metrics["training_time"] = training_time
        final_metrics["epochs_used"] = len(history.history["loss"])
        
        # Restaurar config
        self.config.LEARNING_RATE = original_lr
        self.config.BATCH_SIZE = original_batch_size
        self.config.EPOCHS = original_epochs
        self.config.DROPOUT_RATE = original_dropout
        self.config.OPTIMIZER = original_optimizer
        self.config.ACTIVATION_FUNCTION = original_activation
        
        self.logger.info(f"Modelo final entrenado en {training_time:.2f} segundos")
        self.logger.info(f"Métricas finales: {final_metrics}")
        
        return model, final_metrics
        
    def prepare_for_production(
        self, 
        model_name: str, 
        model: tf.keras.Model, 
        best_params: Dict[str, Any],
        final_metrics: Dict[str, Any]
    ):
        """
        Prepara el modelo para producción: guarda en backend/model/ y genera JSON.
        
        Args:
            model_name: Nombre del modelo.
            model: Modelo entrenado.
            best_params: Mejores hiperparámetros.
            final_metrics: Métricas finales.
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"PREPARANDO MODELO PARA PRODUCCIÓN")
        self.logger.info(f"{'='*60}")
        
        # Crear directorio de backend/model si no existe
        self.config.BACKEND_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        
        # Guardar modelo en backend/model/best_model.keras
        backend_model_path = self.config.BACKEND_MODEL_DIR / "best_model.keras"
        model.save(backend_model_path)
        self.logger.info(f"Modelo guardado en: {backend_model_path}")
        
        # Generar model_info.json
        model_info = {
            "model_name": model_name,
            "training_date": datetime.now().isoformat(),
            "selection_date": datetime.now().isoformat(),
            "hyperparameters": best_params,
            "metrics": final_metrics,
            "config": {
                "cv_folds": self.config.CV_NUM_FOLDS,
                "tuning_method": self.config.TUNING_METHOD.value
            }
        }
        
        model_info_path = self.config.BACKEND_MODEL_INFO_PATH
        with open(model_info_path, "w") as f:
            json.dump(model_info, f, indent=4, default=str)
            
        self.logger.info(f"Información del modelo guardada en: {model_info_path}")
        
        # También guardar una copia en ia/reports/
        shutil.copy(backend_model_path, self.config.REPORTS_DIR / "best_model.keras")
        shutil.copy(model_info_path, self.config.REPORTS_DIR / "best_model_info.json")
        
        self.logger.info(f"Copia del modelo y JSON guardados en: {self.config.REPORTS_DIR}")
        
    def run(self):
        """
        Ejecuta el flujo completo de comparación final y preparación para producción.
        """
        self.logger.info("="*80)
        self.logger.info("INICIANDO COMPARACIÓN FINAL DE MODELOS")
        self.logger.info("="*80)
        
        # 1. Cargar datos
        train_df, val_df, test_df = load_datasets(self.config)
        X_train, y_train = separate_features_target(train_df, self.config.TARGET_VARIABLE)
        X_val, y_val = separate_features_target(val_df, self.config.TARGET_VARIABLE)
        
        # 2. Cargar y procesar resultados de tuning
        try:
            tuning_df = self.load_tuning_results()
        except FileNotFoundError:
            self.logger.warning("Resultados de tuning no encontrados. Ejecutando Cross Validation básica...")
            self.cv_runner.run_all()
            cv_df = pd.read_csv(self.config.CV_RESULTS_PATH)
            
            # Simular resultados de tuning con los resultados de CV
            tuning_df = cv_df.groupby("model").agg(
                avg_rmse=("rmse", "mean"),
                avg_r2=("r2", "mean"),
                avg_mae=("mae", "mean"),
                avg_mape=("mape", "mean"),
                time=("time", "mean")
            ).reset_index()
            # Agregar columnas dummy de params para compatibilidad
            for col in ["lr", "batch_size", "epochs", "dropout", "optimizer", "activation"]:
                tuning_df[col] = "N/A"
        
        # 3. Extraer mejores métricas
        comparison_df = self.extract_best_from_tuning(tuning_df)
        comparison_df.to_csv(self.config.FINAL_COMPARISON_PATH, index=False)
        self.logger.info(f"Comparación final guardada en: {self.config.FINAL_COMPARISON_PATH}")
        
        # 4. Generar gráficos
        self.plot_final_comparison(comparison_df)
        
        # 5. Seleccionar mejor modelo
        best_model_name, best_metrics = self.select_best_model(comparison_df)
        
        # 6. Obtener mejores parámetros para el mejor modelo
        if "iteration" in tuning_df.columns:
            best_model_tuning = tuning_df[tuning_df["model"] == best_model_name]
            best_idx = best_model_tuning["avg_rmse"].idxmin()
            best_params = tuning_df.loc[best_idx].to_dict()
            # Eliminar columnas no necesarias
            for col in ["model", "iteration", "avg_rmse", "avg_r2", "avg_mae", "avg_mape", "time"]:
                if col in best_params:
                    del best_params[col]
        else:
            # Usar params por defecto
            best_params = {
                "lr": self.config.LEARNING_RATE,
                "batch_size": self.config.BATCH_SIZE,
                "epochs": self.config.EPOCHS,
                "dropout": self.config.DROPOUT_RATE,
                "optimizer": self.config.OPTIMIZER,
                "activation": self.config.ACTIVATION_FUNCTION
            }
            if best_model_name in ["lstm", "gru", "cnn_lstm", "cnn_gru"]:
                best_params["seq_length"] = 10
        
        # 7. Entrenar modelo final
        final_model, final_metrics = self.train_final_best_model(
            best_model_name, best_params, X_train, y_train, X_val, y_val
        )
        
        # 8. Preparar para producción
        self.prepare_for_production(best_model_name, final_model, best_params, final_metrics)
        
        # 9. Guardar información del mejor modelo en JSON
        final_info = {
            "model_name": best_model_name,
            "metrics": final_metrics,
            "hyperparameters": best_params,
            "training_date": datetime.now().isoformat()
        }
        with open(self.config.FINAL_BEST_MODEL_PATH, "w") as f:
            json.dump(final_info, f, indent=4, default=str)
            
        self.logger.info("\n" + "="*80)
        self.logger.info("COMPARACIÓN FINAL COMPLETADA EXITOSAMENTE")
        self.logger.info(f"Mejor modelo listo para producción en: {self.config.BACKEND_BEST_MODEL_PATH}")
        self.logger.info("="*80)
        
        return comparison_df, best_model_name, final_model
