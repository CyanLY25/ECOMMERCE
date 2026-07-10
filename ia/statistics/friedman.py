"""
Implementación de la Prueba de Friedman para Comparación de Múltiples Modelos.
"""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, Tuple, List
from pathlib import Path


class FriedmanTest:
    """
    Clase para ejecutar la Prueba de Friedman y analizar resultados.
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Inicializa el test de Friedman.
        
        Args:
            alpha: Nivel de significancia.
        """
        self.alpha = alpha
        self.statistic: float = None
        self.p_value: float = None
        self.ranking: pd.DataFrame = None
        self.average_ranks: Dict[str, float] = None
        self.conclusion: str = None
        
    def calculate_ranks(self, pivot_df: pd.DataFrame, higher_is_better: bool = False) -> pd.DataFrame:
        """
        Calcula los rangos de los modelos para cada fold.
        
        Args:
            pivot_df: DataFrame pivotado (folds x modelos).
            higher_is_better: Si True, valores mayores son mejores (ej: R²).
            
        Returns:
            DataFrame con rangos.
        """
        # Para métricas como RMSE/MAE, menor es mejor: invertimos el ranking
        if higher_is_better:
            rank_df = pivot_df.rank(axis=1, ascending=False)
        else:
            rank_df = pivot_df.rank(axis=1, ascending=True)
        return rank_df
        
    def run(self, pivot_df: pd.DataFrame, higher_is_better: bool = False) -> Dict[str, Any]:
        """
        Ejecuta la prueba de Friedman.
        
        Args:
            pivot_df: DataFrame pivotado (folds x modelos).
            higher_is_better: Si True, valores mayores son mejores.
            
        Returns:
            Diccionario con resultados.
        """
        # Preparar datos para scipy.stats.friedmanchisquare
        model_data = [pivot_df[col].values for col in pivot_df.columns]
        
        # Ejecutar test
        self.statistic, self.p_value = stats.friedmanchisquare(*model_data)
        
        # Calcular rangos
        rank_df = self.calculate_ranks(pivot_df, higher_is_better)
        self.ranking = rank_df
        
        # Calcular rangos promedio
        self.average_ranks = rank_df.mean().to_dict()
        
        # Generar conclusión
        if self.p_value < self.alpha:
            self.conclusion = (
                f"Se rechaza la hipótesis nula (p = {self.p_value:.6f} < α = {self.alpha}). "
                f"Existen diferencias estadísticamente significativas entre al menos dos modelos."
            )
        else:
            self.conclusion = (
                f"No se puede rechazar la hipótesis nula (p = {self.p_value:.6f} ≥ α = {self.alpha}). "
                f"No hay evidencia suficiente para afirmar que existen diferencias significativas entre los modelos."
            )
            
        return {
            "statistic": self.statistic,
            "p_value": self.p_value,
            "alpha": self.alpha,
            "average_ranks": self.average_ranks,
            "ranking_df": self.ranking,
            "conclusion": self.conclusion,
            "significant": self.p_value < self.alpha
        }
        
    def save_results(self, save_path: Path, results: Dict[str, Any]):
        """
        Guarda resultados en CSV.
        
        Args:
            save_path: Ruta donde guardar el CSV.
            results: Diccionario con resultados del test.
        """
        # Guardar resultados principales
        main_data = {
            "test": ["Friedman"],
            "statistic": [results["statistic"]],
            "p_value": [results["p_value"]],
            "alpha": [results["alpha"]],
            "significant": [results["significant"]],
            "conclusion": [results["conclusion"]]
        }
        main_df = pd.DataFrame(main_data)
        main_df.to_csv(save_path, index=False)
        
        # Guardar ranking promedio
        rank_save_path = save_path.parent / "friedman_ranking.csv"
        rank_data = []
        for model, avg_rank in results["average_ranks"].items():
            rank_data.append({"model": model, "average_rank": avg_rank})
        rank_df = pd.DataFrame(rank_data).sort_values("average_rank")
        rank_df.to_csv(rank_save_path, index=False)
