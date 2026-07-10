import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List
from ia.config.config import AIConfig
from ia.training.common import load_metrics
from ia.utils.logger import setup_logger
from ia.training.mlp import run_mlp
from ia.training.lstm import run_lstm
from ia.training.gru import run_gru
from ia.training.cnn_lstm import run_cnn_lstm
from ia.training.cnn_gru import run_cnn_gru


class ModelTrainer:
    """
    Clase para orquestar el entrenamiento de todos los modelos y comparar sus resultados.
    """
    
    def __init__(self, config: AIConfig):
        """
        Inicializa el entrenador de modelos.
        
        Args:
            config: Objeto de configuración.
        """
        self.config = config
        self.logger = setup_logger("trainer")
        self.all_metrics: Dict[str, Dict[str, float]] = {}
        
        # Mapeo de nombres de modelos a sus funciones de ejecución
        self.model_runners = {
            "mlp": run_mlp,
            "lstm": run_lstm,
            "gru": run_gru,
            "cnn_lstm": run_cnn_lstm,
            "cnn_gru": run_cnn_gru
        }
        
    def train_all(self):
        """
        Entrena todos los modelos.
        """
        self.logger.info("=" * 60)
        self.logger.info("ENTRENANDO TODOS LOS MODELOS")
        self.logger.info("=" * 60)
        
        for model_name, runner in self.model_runners.items():
            try:
                self.logger.info(f"\nEntrenando modelo: {model_name.upper()}")
                metrics = runner(self.config)
                self.all_metrics[model_name] = metrics
            except Exception as e:
                self.logger.error(f"Error entrenando {model_name}: {e}", exc_info=True)
                
        self.logger.info("\nTodos los modelos han sido entrenados!")
        
    def compare_models(self):
        """
        Compara los resultados de todos los modelos.
        
        Returns:
            DataFrame con la comparación de métricas.
        """
        self.logger.info("Generando comparación de modelos...")
        
        # Crear DataFrame
        comparison_data = []
        for model_name, metrics in self.all_metrics.items():
            row = {
            "model": model_name,
            "mae": metrics.get("mae"),
            "mse": metrics.get("mse"),
            "rmse": metrics.get("rmse"),
            "r2": metrics.get("r2"),
            "mape": metrics.get("mape"),
            "loss": metrics.get("loss"),
            "training_time": metrics.get("training_time")
        }
        comparison_data.append(row)
            
        comparison_df = pd.DataFrame(comparison_data)
        
        # Guardar
        comparison_df.to_csv(self.config.MODEL_COMPARISON_PATH, index=False)
        self.logger.info(f"Comparación guardada en {self.config.MODEL_COMPARISON_PATH}")
        
        # Mostrar
        self.logger.info("\n" + comparison_df.to_string(index=False))
        
        return comparison_df
        
    def select_best_model(self, comparison_df: pd.DataFrame) -> str:
        """
        Selecciona el mejor modelo según las métricas.
        
        Args:
            comparison_df: DataFrame con la comparación de modelos.
            
        Returns:
            Nombre del mejor modelo.
        """
        self.logger.info("\nSeleccionando el mejor modelo...")
        
        # Ordenar por criterios:
        # 1. Menor RMSE
        # 2. Mayor R²
        # 3. Menor MAE
        # 4. Menor tiempo de entrenamiento (si hay empate)
        
        comparison_df_sorted = comparison_df.sort_values(
            by=["rmse", "r2", "mae", "training_time"],
            ascending=[True, False, True, True]
        )
        
        best_model = comparison_df_sorted.iloc[0]["model"]
        
        # Guardar
        with open(self.config.BEST_MODEL_PATH, 'w') as f:
            json.dump({"best_model": best_model, "metrics": comparison_df_sorted.iloc[0].to_dict()}, f, indent=4)
            
        self.logger.info(f"Mejor modelo seleccionado: {best_model.upper()}")
        
        return best_model
        
    def plot_comparisons(self, comparison_df: pd.DataFrame):
        """
        Genera gráficos de comparación de modelos.
        
        Args:
            comparison_df: DataFrame con la comparación.
        """
        self.logger.info("Generando gráficos de comparación...")
        
        # Configurar estilo
        sns.set_style("whitegrid")
        plt.rcParams["figure.dpi"] = 100
        
        # 1. Comparación de MAE
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x="model", y="mae", data=comparison_df, ax=ax, palette="viridis")
        ax.set_title("Comparación de MAE entre Modelos", fontsize=14)
        ax.set_xlabel("Modelo", fontsize=12)
        ax.set_ylabel("MAE", fontsize=12)
        fig.savefig(self.config.FIGURES_DIR / "comparison_mae.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        
        # 2. Comparación de RMSE
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x="model", y="rmse", data=comparison_df, ax=ax, palette="magma")
        ax.set_title("Comparación de RMSE entre Modelos", fontsize=14)
        ax.set_xlabel("Modelo", fontsize=12)
        ax.set_ylabel("RMSE", fontsize=12)
        fig.savefig(self.config.FIGURES_DIR / "comparison_rmse.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        
        # 3. Comparación de R²
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x="model", y="r2", data=comparison_df, ax=ax, palette="coolwarm")
        ax.set_title("Comparación de R² entre Modelos", fontsize=14)
        ax.set_xlabel("Modelo", fontsize=12)
        ax.set_ylabel("R²", fontsize=12)
        fig.savefig(self.config.FIGURES_DIR / "comparison_r2.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        
        # 4. Comparación de Tiempo de Entrenamiento
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x="model", y="training_time", data=comparison_df, ax=ax, palette="plasma")
        ax.set_title("Comparación de Tiempo de Entrenamiento entre Modelos", fontsize=14)
        ax.set_xlabel("Modelo", fontsize=12)
        ax.set_ylabel("Tiempo (segundos)", fontsize=12)
        fig.savefig(self.config.FIGURES_DIR / "comparison_training_time.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        
        self.logger.info("Gráficos de comparación guardados!")
        
    def run(self):
        """
        Ejecuta todo el pipeline: entrenamiento, comparación y selección.
        """
        # Asegurar directorios
        self.config.ensure_directories_exist()
        
        # Entrenar modelos
        self.train_all()
        
        # Comparar
        comparison_df = self.compare_models()
        
        # Seleccionar mejor
        best_model = self.select_best_model(comparison_df)
        
        # Generar gráficos de comparación
        self.plot_comparisons(comparison_df)
        
        return best_model
