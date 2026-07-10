"""
Implementación del Test Post-Hoc de Nemenyi.
"""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, List
from pathlib import Path


class NemenyiTest:
    """
    Clase para ejecutar el Test Post-Hoc de Nemenyi.
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Inicializa el test de Nemenyi.
        
        Args:
            alpha: Nivel de significancia.
        """
        self.alpha = alpha
        self.results_df: pd.DataFrame = None
        self.critical_difference: float = None
        self.significant_pairs: List[Tuple[str, str]] = []
        
    def calculate_critical_difference(
        self, 
        num_models: int, 
        num_folds: int
    ) -> float:
        """
        Calcula la Diferencia Crítica (CD) para el test de Nemenyi.
        
        Args:
            num_models: Número de modelos a comparar.
            num_folds: Número de folds.
            
        Returns:
            Valor de la Diferencia Crítica.
        """
        # Valores críticos de la distribución de studentizada de Student
        # Para α=0.05 y valores comunes (basado en tablas de Demšar)
        q_values = {
            (2, 0.05): 1.960,
            (3, 0.05): 2.344,
            (4, 0.05): 2.569,
            (5, 0.05): 2.728,
            (6, 0.05): 2.850,
            (7, 0.05): 2.949,
            (8, 0.05): 3.031,
            (9, 0.05): 3.102,
            (10, 0.05): 3.164
        }
        
        # Obtener q_valor (si no está, usar aproximación
        key = (num_models, self.alpha)
        if key in q_values:
            q = q_values[key]
        else:
            q = stats.norm.ppf(1 - self.alpha / 2)
            
        cd = q * np.sqrt(num_models * (num_models + 1) / (6 * num_folds))
        return cd
        
    def run(
        self, average_ranks: Dict[str, float], num_folds: int) -> Dict[str, Any]:
        """
        Ejecuta el test de Nemenyi.
        
        Args:
            average_ranks: Diccionario con rangos promedio por modelo.
            num_folds: Número de folds.
            
        Returns:
            Diccionario con resultados.
        """
        model_names = list(average_ranks.keys())
        num_models = len(model_names)
        
        # Calcular diferencia crítica
        self.critical_difference = self.calculate_critical_difference(num_models, num_folds)
        
        results = []
        
        # Comparar todos los pares
        for i in range(num_models):
            for j in range(i + 1, num_models):
                model_a = model_names[i]
                model_b = model_names[j]
                rank_a = average_ranks[model_a]
                rank_b = average_ranks[model_b]
                diff = abs(rank_a - rank_b)
                significant = diff > self.critical_difference
                
                results.append({
                    "model_a": model_a,
                    "model_b": model_b,
                    "rank_a": rank_a,
                    "rank_b": rank_b,
                    "rank_diff": diff,
                    "critical_diff": self.critical_difference,
                    "significant": significant
                })
                
                if significant:
                    self.significant_pairs.append((model_a, model_b))
                    
        self.results_df = pd.DataFrame(results)
        
        return {
            "critical_difference": self.critical_difference,
            "results_df": self.results_df,
            "significant_pairs": self.significant_pairs,
            "average_ranks": average_ranks
        }
        
    def save_results(self, save_path: Path):
        """
        Guarda resultados en CSV.
        """
        self.results_df.to_csv(save_path, index=False)
