"""
Módulo de validación cruzada genérica para modelos de TensorFlow/Keras.
"""
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Tuple, Callable
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import clone_model

from ia.config.config import AIConfig
from ia.training.common import (
    set_seed,
    separate_features_target,
    calculate_metrics
)
from ia.utils.logger import setup_logger

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100


class CrossValidator:
    """
    Clase genérica para ejecutar validación temporal en modelos de TensorFlow/Keras.
    Diseñada para ser reutilizada por cualquier modelo compatible y para Hyperparameter Tuning.
    """
    
    def __init__(self, config: AIConfig):
        """
        Inicializa el validador cruzado.
        
        Args:
            config: Objeto de configuración del proyecto.
        """
        self.config = config
        self.logger = setup_logger("cross_validation", config.CV_LOG_PATH)
        self.figures_dir = config.FIGURES_DIR
        
    def _get_model_builder(self, model_name: str) -> Callable:
        """
        Obtiene la función de construcción del modelo según su nombre.
        
        Args:
            model_name: Nombre del modelo ("mlp", "lstm", "gru", "cnn_lstm", "cnn_gru").
            
        Returns:
            Función que construye y compila el modelo.
        """
        from ia.training.mlp import MLPModel
        from ia.training.lstm import LSTMModel
        from ia.training.gru import GRUModel
        from ia.training.cnn_lstm import CNNLSTMModel
        from ia.training.cnn_gru import CNNGRUModel
        from ia.training.tft import TFTModel
        
        model_classes = {
            "mlp": MLPModel,
            "lstm": LSTMModel,
            "gru": GRUModel,
            "cnn_lstm": CNNLSTMModel,
            "cnn_gru": CNNGRUModel,
            "tft": TFTModel
        }
        
        if model_name not in model_classes:
            raise ValueError(f"Modelo '{model_name}' no reconocido. Opciones: {list(model_classes.keys())}")
            
        return model_classes[model_name]
        
    def _prepare_data(self, model_name: str, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Prepara los datos según el tipo de modelo (secuencial o no secuencial).
        
        Args:
            model_name: Nombre del modelo.
            X: Características de entrada.
            y: Variable objetivo.
            
        Returns:
            Tupla con (X_preparado, y_preparado, window_size).
            window_size es 0 para modelos no secuenciales.
        """
        from ia.training.common import create_sequences
        
        if model_name in ["lstm", "gru", "cnn_lstm", "cnn_gru", "tft"]:
            window_size = getattr(self.config, f"{model_name.upper().replace('_', '')}_WINDOW_SIZE", 
                                  getattr(self.config, f"{model_name.upper()}_SEQUENCE_LENGTH", 10))
            X_seq, y_seq = create_sequences(X, y, window_size)
            return X_seq, y_seq, window_size
        else:
            return X, y, 0
            
    def _get_callbacks(self, model_name: str) -> List[tf.keras.callbacks.Callback]:
        """
        Obtiene los callbacks para el entrenamiento, pero sin ModelCheckpoint
        ya que en CV no queremos sobrescribir el modelo principal.
        
        Args:
            model_name: Nombre del modelo.
            
        Returns:
            Lista de callbacks.
        """
        callbacks = []
        
        # Early Stopping
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            mode='min',
            verbose=0
        ))
        
        # Reduce LR on Plateau
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            mode='min',
            verbose=0
        ))
        
        return callbacks
        
    def _train_and_evaluate_fold(self, model_name: str, 
                                X_train: np.ndarray, y_train: np.ndarray,
                                X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """
        Entrena y evalúa un modelo en un fold específico.
        
        Args:
            model_name: Nombre del modelo.
            X_train: Datos de entrenamiento del fold.
            y_train: Etiquetas de entrenamiento del fold.
            X_val: Datos de validación del fold.
            y_val: Etiquetas de validación del fold.
            
        Returns:
            Diccionario con métricas y resultados del fold.
        """
        set_seed(self.config)
        
        # Construir el modelo
        ModelClass = self._get_model_builder(model_name)
        model_instance = ModelClass(self.config)
        
        # Preparar datos para modelos secuenciales
        if model_name in ["lstm", "gru", "cnn_lstm", "cnn_gru", "tft"]:
            X_train_prep, y_train_prep, _ = self._prepare_data(model_name, X_train, y_train)
            X_val_prep, y_val_prep, _ = self._prepare_data(model_name, X_val, y_val)
            
            if len(X_train_prep) == 0 or len(X_val_prep) == 0:
                raise ValueError(f"Datos insuficientes para crear secuencias para {model_name}")
                
            model = model_instance.build_model((X_train_prep.shape[1], X_train_prep.shape[2]))
            train_data = (X_train_prep, y_train_prep)
            val_data = (X_val_prep, y_val_prep)
        else:
            model = model_instance.build_model(X_train.shape[1:])
            train_data = (X_train, y_train)
            val_data = (X_val, y_val)
            
        # Callbacks
        callbacks = self._get_callbacks(model_name)
        
        # Parámetros de entrenamiento específicos del modelo
        epochs = getattr(self.config, f"{model_name.upper()}_EPOCHS", self.config.EPOCHS)
        batch_size = getattr(self.config, f"{model_name.upper()}_BATCH_SIZE", self.config.BATCH_SIZE)
        
        # Entrenar
        start_time = time.time()
        history = model.fit(
            train_data[0], train_data[1],
            epochs=epochs,
            batch_size=batch_size,
            validation_data=val_data,
            callbacks=callbacks,
            verbose=0
        )
        end_time = time.time()
        training_time = end_time - start_time
        
        # Evaluar
        y_pred = model.predict(val_data[0], verbose=0).flatten()
        final_metrics = calculate_metrics(val_data[1], y_pred)
        
        # Añadir métricas adicionales
        final_metrics['loss'] = float(history.history['loss'][-1])
        final_metrics['val_loss'] = float(history.history['val_loss'][-1])
        final_metrics['training_time'] = training_time
        final_metrics['epochs_run'] = len(history.epoch)
        
        return final_metrics
        
    def run(self, model_name: str, X: np.ndarray, y: np.ndarray) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Ejecuta la validación cruzada K-Fold para un modelo específico.
        
        Args:
            model_name: Nombre del modelo ("mlp", "lstm", "gru", "cnn_lstm", "cnn_gru").
            X: Características de entrada completas.
            y: Variable objetivo completa.
            
        Returns:
            Tupla con (lista de resultados por fold, diccionario de resumen).
        """
        self.logger.info("=" * 80)
        self.logger.info(f"INICIANDO VALIDACIÓN CRUZADA PARA MODELO: {model_name.upper()}")
        self.logger.info("=" * 80)
        
        print("\n" + "=" * 80)
        print(f"Modelo: {model_name.upper()}")
        print("=" * 80)
        
        # Los CSV están ordenados cronológicamente. TimeSeriesSplit garantiza
        # que cada validación sea posterior a sus datos de entrenamiento y
        # evita filtrar observaciones futuras hacia TFT.
        splitter = TimeSeriesSplit(n_splits=self.config.CV_FOLDS)
        
        fold_results = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
            self.logger.info(f"Fold {fold_idx + 1}/{self.config.CV_FOLDS}")
            print(f"\nFold {fold_idx + 1}/{self.config.CV_FOLDS}")
            
            # Dividir datos
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            try:
                # Entrenar y evaluar
                fold_metrics = self._train_and_evaluate_fold(
                    model_name, X_train, y_train, X_val, y_val
                )
                fold_metrics['fold'] = fold_idx + 1
                fold_results.append(fold_metrics)
                
                # Mostrar resultados del fold
                self.logger.info(f"  MAE: {fold_metrics['mae']:.4f}")
                self.logger.info(f"  RMSE: {fold_metrics['rmse']:.4f}")
                self.logger.info(f"  R²: {fold_metrics['r2']:.4f}")
                self.logger.info(f"  MAPE: {fold_metrics['mape']:.4f}")
                self.logger.info(f"  Tiempo: {fold_metrics['training_time']:.2f}s")
                self.logger.info(f"  Épocas: {fold_metrics['epochs_run']}")
                
                print(f"  MAE: {fold_metrics['mae']:.4f}")
                print(f"  RMSE: {fold_metrics['rmse']:.4f}")
                print(f"  R²: {fold_metrics['r2']:.4f}")
                print(f"  MAPE: {fold_metrics['mape']:.4f}")
                print(f"  Tiempo: {fold_metrics['training_time']:.2f}s")
                print(f"  Épocas: {fold_metrics['epochs_run']}")
                
            except Exception as e:
                self.logger.error(f"Error en fold {fold_idx + 1}: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                raise
                
        # Calcular resumen
        summary = self._calculate_summary(fold_results)
        
        # Mostrar resumen
        self._print_summary(model_name, summary)
        
        return fold_results, summary
        
    def _calculate_summary(self, fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calcula el resumen estadístico de los resultados de los folds.
        
        Args:
            fold_results: Lista de diccionarios con resultados por fold.
            
        Returns:
            Diccionario con promedio, desviación estándar, mínimo y máximo por métrica.
        """
        metrics_list = ['mae', 'mse', 'rmse', 'mape', 'r2', 'loss', 'val_loss', 'training_time', 'epochs_run']
        summary = {}
        
        for metric in metrics_list:
            values = [r[metric] for r in fold_results if metric in r]
            if values:
                summary[f'mean_{metric}'] = float(np.mean(values))
                summary[f'std_{metric}'] = float(np.std(values))
                summary[f'min_{metric}'] = float(np.min(values))
                summary[f'max_{metric}'] = float(np.max(values))
                
        return summary
        
    def _print_summary(self, model_name: str, summary: Dict[str, Any]):
        """
        Imprime el resumen en consola y en el logger.
        
        Args:
            model_name: Nombre del modelo.
            summary: Diccionario con el resumen estadístico.
        """
        self.logger.info("\n" + "-" * 80)
        self.logger.info(f"RESUMEN DE VALIDACIÓN CRUZADA - {model_name.upper()}")
        self.logger.info("-" * 80)
        
        print("\n" + "-" * 80)
        print(f"RESUMEN DE VALIDACIÓN CRUZADA - {model_name.upper()}")
        print("-" * 80)
        
        metrics_map = {
            'mae': 'MAE',
            'rmse': 'RMSE',
            'r2': 'R²',
            'mape': 'MAPE',
            'training_time': 'Tiempo'
        }
        
        for metric_key, metric_name in metrics_map.items():
            mean = summary.get(f'mean_{metric_key}', 0)
            std = summary.get(f'std_{metric_key}', 0)
            self.logger.info(f"{metric_name}: {mean:.4f} ± {std:.4f}")
            print(f"{metric_name}: {mean:.4f} ± {std:.4f}")
            
        self.logger.info("=" * 80 + "\n")
        print("=" * 80 + "\n")


def plot_cross_validation_results(all_results: Dict[str, Dict], config: AIConfig):
    """
    Genera gráficos de comparación de resultados de validación cruzada.
    
    Args:
        all_results: Diccionario con resultados por modelo.
        config: Objeto de configuración.
    """
    figures_dir = config.FIGURES_DIR
    
    # Preparar datos para gráficos
    data = []
    for model_name, result in all_results.items():
        for fold in result['folds']:
            row = fold.copy()
            row['model'] = model_name
            data.append(row)
            
    df = pd.DataFrame(data)
    
    # 1. Boxplot de todas las métricas
    metrics = ['mae', 'rmse', 'r2']
    metric_names = ['MAE', 'RMSE', 'R²']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for ax, metric, name in zip(axes, metrics, metric_names):
        sns.boxplot(data=df, x='model', y=metric, ax=ax, palette='viridis')
        ax.set_title(f'Distribución de {name} por Modelo', fontsize=14)
        ax.set_xlabel('Modelo', fontsize=12)
        ax.set_ylabel(name, fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        
    plt.tight_layout()
    plt.savefig(figures_dir / 'cross_validation_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Barplot de promedios
    summary_data = []
    for model_name, result in all_results.items():
        row = {'model': model_name}
        for key, value in result['summary'].items():
            row[key] = value
        summary_data.append(row)
        
    summary_df = pd.DataFrame(summary_data)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for ax, metric, name in zip(axes, metrics, metric_names):
        sns.barplot(data=summary_df, x='model', y=f'mean_{metric}', ax=ax, palette='viridis', alpha=0.8)
        ax.errorbar(data=summary_df, x='model', y=f'mean_{metric}', yerr=f'std_{metric}', 
                   fmt='none', c='black', capsize=5)
        ax.set_title(f'Promedio de {name} ± Desviación Estándar', fontsize=14)
        ax.set_xlabel('Modelo', fontsize=12)
        ax.set_ylabel(name, fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        
    plt.tight_layout()
    plt.savefig(figures_dir / 'cross_validation_barplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Gráfico de RMSE
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='model', y='rmse', palette='magma')
    plt.title('Distribución de RMSE por Modelo (Cross Validation)', fontsize=14)
    plt.xlabel('Modelo', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(figures_dir / 'cross_validation_rmse.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Gráfico de MAE
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='model', y='mae', palette='magma')
    plt.title('Distribución de MAE por Modelo (Cross Validation)', fontsize=14)
    plt.xlabel('Modelo', fontsize=12)
    plt.ylabel('MAE', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(figures_dir / 'cross_validation_mae.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Gráfico de R²
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='model', y='r2', palette='coolwarm')
    plt.title('Distribución de R² por Modelo (Cross Validation)', fontsize=14)
    plt.xlabel('Modelo', fontsize=12)
    plt.ylabel('R²', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(figures_dir / 'cross_validation_r2.png', dpi=300, bbox_inches='tight')
    plt.close()


def save_results(all_results: Dict[str, Dict], config: AIConfig):
    """
    Guarda los resultados de validación cruzada en archivos JSON y CSV.
    
    Args:
        all_results: Diccionario con resultados por modelo.
        config: Objeto de configuración.
    """
    # Guardar JSON
    with open(config.CV_RESULTS_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
        
    # Guardar CSV detallado
    csv_data = []
    for model_name, result in all_results.items():
        for fold in result['folds']:
            row = fold.copy()
            row['model'] = model_name
            csv_data.append(row)
            
    pd.DataFrame(csv_data).to_csv(config.CV_RESULTS_CSV_PATH, index=False, encoding='utf-8')
    
    # Guardar CSV de resumen
    summary_data = []
    for model_name, result in all_results.items():
        row = {'model': model_name}
        for key, value in result['summary'].items():
            row[key] = value
        summary_data.append(row)
        
    pd.DataFrame(summary_data).to_csv(config.CV_SUMMARY_CSV_PATH, index=False, encoding='utf-8')
