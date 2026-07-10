"""
Módulo de análisis principal: cálculo de intervalos de confianza, ranking, etc.
"""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, List
from pathlib import Path


class ModelAnalyzer:
    """
    Clase para análisis de modelos.
    """
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        
    def calculate_confidence_intervals(
        self, results_df: pd.DataFrame, metric: str
    ) -> pd.DataFrame:
        """
        Calcula intervalos de confianza para cada modelo.
        """
        ci_list = []
        
        for model in results_df["model"].unique():
            model_data = results_df[results_df["model"] == model][metric].values
            mean_val = np.mean(model_data)
            std_val = np.std(model_data, ddof=1)
            n = len(model_data)
            sem = stats.sem(model_data)
            
            ci_low, ci_high = stats.t.interval(
                confidence=1 - self.alpha,
                df=n - 1,
                loc=mean_val,
                scale=sem
            )
            
            ci_list.append({
                "model": model,
                "mean": mean_val,
                "std": std_val,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n": n
            })
            
        ci_df = pd.DataFrame(ci_list)
        return ci_df
        
    def generate_ranking(
        self, results_df: pd.DataFrame, 
        metric: str, 
        higher_is_better: bool = False
    ) -> pd.DataFrame:
        """
        Genera ranking de modelos basado en la métrica.
        """
        # Calcular media por modelo
        model_means = results_df.groupby("model")[metric].mean().reset_index()
        
        # Ordenar
        if higher_is_better:
            model_means = model_means.sort_values(metric, ascending=False)
        else:
            model_means = model_means.sort_values(metric, ascending=True)
            
        # Añadir posición
        model_means["rank"] = range(1, len(model_means) + 1)
        
        return model_means
        
    def get_best_model(self, ranking_df: pd.DataFrame) -> str:
        """
        Obtiene el modelo con mejor rendimiento.
        """
        return ranking_df.iloc[0]["model"]
