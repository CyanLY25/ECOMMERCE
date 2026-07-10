"""
Implementación del Test de Wilcoxon Signed-Rank con Corrección de Bonferroni.
"""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, List
from pathlib import Path
from .utils import interpret_cohens_d, interpret_cliffs_delta, safe_divide


class WilcoxonTest:
    """
    Clase para ejecutar Test de Wilcoxon y calcular tamaños del efecto.
    """

    def __init__(self, alpha: float = 0.05):
        """
        Inicializa el test de Wilcoxon.
        """
        self.alpha = alpha
        self.results_df: pd.DataFrame = None
        
    def calculate_cohens_d(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calcula Cohen's d para datos pareados.
        
        Args:
            a: Datos del modelo A.
            b: Datos del modelo B.
            
        Returns:
            Valor de Cohen's d.
        """
        diff = a - b
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1)
        return safe_divide(mean_diff, std_diff)
        
    def calculate_cliffs_delta(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calcula Cliff's Delta (tamaño del efecto no paramétrico).
        
        Args:
            a: Datos del modelo A.
            b: Datos del modelo B.
            
        Returns:
            Valor de Cliff's Delta.
        """
        n = len(a)
        wins = 0
        losses = 0
        
        for ai, bi in zip(a, b):
            if ai > bi:
                wins += 1
            elif ai < bi:
                losses += 1
                
        return (wins - losses) / (n * n)
        
    def run_pair(
        self, data_a: np.ndarray, data_b: np.ndarray, 
        model_a: str, model_b: str,
        higher_is_better: bool = False
    ) -> Dict[str, Any]:
        """
        Ejecuta Wilcoxon para un par de modelos.
        
        Args:
            data_a: Datos del modelo A.
            data_b: Datos del modelo B.
            model_a: Nombre del modelo A.
            model_b: Nombre del modelo B.
            higher_is_better: Si valores mayores son mejores.
            
        Returns:
            Diccionario con resultados.
        """
        # Wilcoxon Signed-Rank Test
        # Por defecto, alternativa 'two-sided'
        statistic, p_value = stats.wilcoxon(
            data_a, data_b, 
            zero_method='pratt',
            alternative='two-sided'
        )
        
        # Tamaños del efecto
        cohens_d = self.calculate_cohens_d(data_a, data_b)
        cliffs_delta = self.calculate_cliffs_delta(data_a, data_b)
        
        # Interpretación
        cohens_interpretation = interpret_cohens_d(cohens_d)
        cliffs_interpretation = interpret_cliffs_delta(cliffs_delta)
        
        return {
            "model_a": model_a,
            "model_b": model_b,
            "statistic": statistic,
            "p_value": p_value,
            "cohens_d": cohens_d,
            "cohens_interpretation": cohens_interpretation,
            "cliffs_delta": cliffs_delta,
            "cliffs_interpretation": cliffs_interpretation,
            "mean_a": np.mean(data_a),
            "mean_b": np.mean(data_b),
            "mean_diff": np.mean(data_a - data_b)
        }
        
    def run_all(
        self, pivot_df: pd.DataFrame, 
        higher_is_better: bool = False
    ) -> Dict[str, Any]:
        """
        Ejecuta Wilcoxon para todos los pares de modelos con corrección de Bonferroni.
        
        Args:
            pivot_df: DataFrame pivotado (folds x modelos).
            higher_is_better: Si True, valores mayores son mejores.
            
        Returns:
            Diccionario con resultados.
        """
        model_names = pivot_df.columns.tolist()
        all_results = []
        
        num_comparisons = len(model_names) * (len(model_names) - 1) // 2
        
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                model_a = model_names[i]
                model_b = model_names[j]
                data_a = pivot_df[model_a].values
                data_b = pivot_df[model_b].values
                
                result = self.run_pair(
                    data_a, data_b, model_a, model_b, higher_is_better
                )
                
                all_results.append(result)
                
        self.results_df = pd.DataFrame(all_results)
        
        # Aplicar corrección de Bonferroni
        self.results_df["p_value_bonferroni"] = self.results_df["p_value"] * num_comparisons
        self.results_df["p_value_bonferroni"] = self.results_df["p_value_bonferroni"].clip(upper=1.0)
        
        # Determinar significancia
        self.results_df["significant"] = self.results_df["p_value_bonferroni"] < self.alpha
        
        # Generar interpretaciones
        interpretations = []
        for idx, row in self.results_df.iterrows():
            row_a = row["model_a"]
            row_b = row["model_b"]
            if row["significant"]:
                if higher_is_better:
                    better = row_a if row["mean_a"] > row["mean_b"] else row_b
                    worse = row_b if row["mean_a"] > row["mean_b"] else row_a
                else:
                    better = row_a if row["mean_a"] < row["mean_b"] else row_b
                    worse = row_b if row["mean_a"] < row["mean_b"] else row_a
                interpretations.append(
                    f"El modelo {better} es estadísticamente superior a {worse} (p={row['p_value_bonferroni']:.6f})"
                )
            else:
                interpretations.append(
                    f"No hay diferencias estadísticamente significativas entre {row['model_a']} y {row['model_b']}"
                )
                
        self.results_df["interpretation"] = interpretations
        
        return {
            "results_df": self.results_df,
            "num_comparisons": num_comparisons,
            "alpha": self.alpha
        }
        
    def save_results(self, save_path: Path):
        """
        Guarda resultados en CSV.
        """
        self.results_df.to_csv(save_path, index=False)
