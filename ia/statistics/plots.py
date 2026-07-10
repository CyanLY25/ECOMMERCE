"""
Generación de gráficos para la validación estadística.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List


class PlotGenerator:
    """
    Clase para generar todos los gráficos requeridos.
    """
    
    def __init__(self, figures_dir: Path, dpi: int = 300):
        """
        Inicializa el generador de gráficos.
        """
        self.figures_dir = figures_dir
        self.dpi = dpi
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        sns.set_style("whitegrid")
        plt.rcParams.update({'figure.dpi': dpi, 'font.size': 10})
        
    def plot_critical_difference(
        self, average_ranks: Dict[str, float], 
        critical_difference: float,
        save_name: str = "critical_difference.png"
    ):
        """
        Genera el gráfico de Diferencia Crítica (CD-diagram).
        """
        # Ordenar modelos por rango promedio
        sorted_models = sorted(average_ranks.items(), key=lambda x: x[1])
        model_names = [item[0] for item in sorted_models]
        ranks = np.array([item[1] for item in sorted_models])
        num_models = len(model_names)
        
        fig, ax = plt.subplots(figsize=(12, 3))
        
        # Dibujar líneas
        ax.plot(ranks, np.zeros_like(ranks), 'ko', markersize=8)
        
        # Añadir nombres de modelos
        for i, (model, rank) in enumerate(sorted_models):
            ax.text(rank, 0.1, model, ha='center', va='bottom', rotation=45, fontsize=10)
            
        # Dibujar barras de CD
        y_pos = -0.1
        current_start = 0
        i = 0
        while i < num_models:
            j = i
            while j < num_models and ranks[j] - ranks[i] <= critical_difference:
                j += 1
            if j > i + 1:
                ax.plot([ranks[i], ranks[j-1]], [y_pos, y_pos], 'k-', linewidth=3)
            i = j
            
        # Ajustar límites
        ax.set_xlim(0.5, num_models + 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_xlabel("Ranking Promedio")
        ax.set_title(f"Diagrama de Diferencia Crítica (CD = {critical_difference:.3f})")
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / save_name, bbox_inches='tight', dpi=self.dpi)
        plt.close()
        
    def plot_boxplot(
        self, results_df: pd.DataFrame, metric: str,
        save_name: str = "boxplot_metric.png"
    ):
        """
        Genera boxplot de la métrica por modelo.
        """
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='model', y=metric, data=results_df, palette='viridis')
        plt.title(f"Distribución de {metric} por Modelo")
        plt.xlabel("Modelo")
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.figures_dir / save_name, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
    def plot_violinplot(
        self, results_df: pd.DataFrame, metric: str,
        save_name: str = "violinplot_metric.png"
    ):
        """
        Genera violinplot de la métrica por modelo.
        """
        plt.figure(figsize=(10, 6))
        sns.violinplot(x='model', y=metric, data=results_df, palette='magma')
        plt.title(f"Distribución de {metric} por Modelo (Violin Plot)")
        plt.xlabel("Modelo")
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.figures_dir / save_name, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
    def plot_ranking(
        self, average_ranks: Dict[str, float],
        save_name: str = "ranking_plot.png"
    ):
        """
        Genera gráfico de barras del ranking promedio.
        """
        sorted_items = sorted(average_ranks.items(), key=lambda x: x[1])
        models = [item[0] for item in sorted_items]
        ranks = [item[1] for item in sorted_items]
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=models, y=ranks, palette='coolwarm')
        plt.title("Ranking Promedio de Modelos")
        plt.xlabel("Modelo")
        plt.ylabel("Ranking Promedio (menor es mejor)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.figures_dir / save_name, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
    def plot_pvalues_heatmap(
        self, wilcoxon_df: pd.DataFrame,
        save_name: str = "pvalues_heatmap.png"
    ):
        """
        Genera heatmap de p-values.
        """
        # Preparar datos
        models = pd.unique(wilcoxon_df[['model_a', 'model_b']].values.ravel('K'))
        num_models = len(models)
        p_matrix = np.ones((num_models, num_models))
        
        model_to_idx = {model: i for i, model in enumerate(models)}
        
        for _, row in wilcoxon_df.iterrows():
            i = model_to_idx[row['model_a']]
            j = model_to_idx[row['model_b']]
            p = row['p_value_bonferroni']
            p_matrix[i, j] = p
            p_matrix[j, i] = p
            
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            p_matrix,
            xticklabels=models,
            yticklabels=models,
            annot=True,
            fmt=".4f",
            cmap="viridis_r",
            cbar_kws={'label': 'p-value (Bonferroni)'}
        )
        plt.title("Heatmap de p-values (Bonferroni)")
        plt.tight_layout()
        plt.savefig(self.figures_dir / save_name, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
    def plot_significance_heatmap(
        self, wilcoxon_df: pd.DataFrame,
        save_name: str = "significance_heatmap.png"
    ):
        """
        Genera heatmap de significancia.
        """
        models = pd.unique(wilcoxon_df[['model_a', 'model_b']].values.ravel('K'))
        num_models = len(models)
        sig_matrix = np.zeros((num_models, num_models), dtype=int)
        
        model_to_idx = {model: i for i, model in enumerate(models)}
        
        for _, row in wilcoxon_df.iterrows():
            i = model_to_idx[row['model_a']]
            j = model_to_idx[row['model_b']]
            sig = 1 if row['significant'] else 0
            sig_matrix[i, j] = sig
            sig_matrix[j, i] = sig
            
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            sig_matrix,
            xticklabels=models,
            yticklabels=models,
            annot=True,
            fmt="d",
            cmap="RdYlGn",
            cbar_kws={'label': 'Significancia (1=significativo)'},
            vmin=0, vmax=1
        )
        plt.title("Heatmap de Significancia Estadística")
        plt.tight_layout()
        plt.savefig(self.figures_dir / save_name, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
    def plot_confidence_intervals(
        self, ci_df: pd.DataFrame, metric: str,
        save_name: str = "confidence_intervals.png"
    ):
        """
        Genera gráfico de intervalos de confianza.
        """
        plt.figure(figsize=(12, 6))
        
        # Ordenar por mean
        ci_df_sorted = ci_df.sort_values('mean')
        
        y_pos = np.arange(len(ci_df_sorted))
        
        plt.errorbar(
            x=ci_df_sorted['mean'],
            y=y_pos,
            xerr=[ci_df_sorted['mean'] - ci_df_sorted['ci_low'], 
                  ci_df_sorted['ci_high'] - ci_df_sorted['mean']],
            fmt='o',
            capsize=5,
            ecolor='darkblue',
            color='darkblue'
        )
        
        plt.yticks(y_pos, ci_df_sorted['model'])
        plt.xlabel(f"{metric} (media ± IC 95%)")
        plt.title("Intervalos de Confianza del 95%")
        plt.tight_layout()
        plt.savefig(self.figures_dir / save_name, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
    def generate_all_plots(
        self, results_df: pd.DataFrame,
        average_ranks: Dict[str, float],
        critical_difference: float,
        wilcoxon_df: pd.DataFrame,
        ci_df: pd.DataFrame,
        metric: str
    ):
        """
        Genera todos los gráficos.
        """
        self.plot_critical_difference(average_ranks, critical_difference)
        self.plot_boxplot(results_df, metric)
        self.plot_violinplot(results_df, metric)
        self.plot_ranking(average_ranks)
        self.plot_pvalues_heatmap(wilcoxon_df)
        self.plot_significance_heatmap(wilcoxon_df)
        self.plot_confidence_intervals(ci_df, metric)
