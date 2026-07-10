#!/usr/bin/env python3
"""
Módulo Principal de Validación Estadística para Comparación de Modelos.
"""
import sys
from pathlib import Path

# Añadir el directorio raíz del proyecto al path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ia.config.config import AIConfig
from ia.utils.logger import setup_logger
from .utils import load_cv_results, pivot_cv_results
from .friedman import FriedmanTest
from .nemenyi import NemenyiTest
from .wilcoxon import WilcoxonTest
from .plots import PlotGenerator
from .analysis import ModelAnalyzer
from .report import ReportGenerator


class StatisticalValidator:
    """
    Clase principal que coordina toda la validación estadística.
    """
    
    def __init__(self, config: AIConfig):
        """
        Inicializa el validador.
        """
        self.config = config
        self.logger = setup_logger("statistics", config.STATISTICS_LOG_PATH)
        self.alpha = config.STATISTICS_ALPHA
        self.metric = config.STATISTICS_METRIC
        self.higher_is_better = self.metric in ["r2", "accuracy"]
        
        # Inicializar componentes
        self.friedman = FriedmanTest(self.alpha)
        self.nemenyi = NemenyiTest(self.alpha)
        self.wilcoxon = WilcoxonTest(self.alpha)
        self.analyzer = ModelAnalyzer(self.alpha)
        self.plotter = PlotGenerator(config.FIGURES_DIR, dpi=300)
        self.reporter = ReportGenerator(config.REPORTS_DIR)
        
    def run(self):
        """
        Ejecuta todo el flujo de validación.
        """
        self.logger.info("=" * 80)
        self.logger.info("INICIANDO VALIDACIÓN ESTADÍSTICA")
        self.logger.info("=" * 80)
        
        # Paso 1: Cargar datos de CV
        self.logger.info("Cargando resultados de Cross Validation...")
        try:
            results_df = load_cv_results(self.config.CV_RESULTS_CSV_PATH)
            self.logger.info(f"Datos cargados: {len(results_df)} filas")
        except FileNotFoundError:
            self.logger.error("Archivo de Cross Validation no encontrado.")
            self.logger.error(f"Por favor, ejecuta primero la fase de Cross Validation.")
            return None
            
        # Paso 2: Preparar datos pivotados
        pivot_df = pivot_cv_results(results_df, self.metric)
        
        # Paso 3: Prueba de Friedman
        self.logger.info("Ejecutando Prueba de Friedman...")
        friedman_results = self.friedman.run(pivot_df, self.higher_is_better)
        self.logger.info(f"Resultado Friedman: statistic={friedman_results['statistic']:.4f}, p={friedman_results['p_value']:.6f}")
        self.friedman.save_results(self.config.FRIEDMAN_RESULTS_PATH, friedman_results)
        
        # Paso 4: Nemenyi (si Friedman es significativo)
        nemenyi_results = None
        if friedman_results["significant"]:
            self.logger.info("Friedman significativo - ejecutando Nemenyi...")
            num_folds = len(pivot_df)
            nemenyi_results = self.nemenyi.run(friedman_results["average_ranks"], num_folds)
            self.nemenyi.save_results(self.config.NEMENYI_RESULTS_PATH)
            self.logger.info(f"Nemenyi completado - CD: {nemenyi_results['critical_difference']:.3f}")
        else:
            self.logger.info("Friedman no significativo - saltando Nemenyi.")
            
        # Paso 5: Wilcoxon para todos los pares
        self.logger.info("Ejecutando Test de Wilcoxon con Bonferroni...")
        wilcoxon_results = self.wilcoxon.run_all(pivot_df, self.higher_is_better)
        self.wilcoxon.save_results(self.config.WILCOXON_RESULTS_PATH)
        self.logger.info(f"Wilcoxon completado - {wilcoxon_results['num_comparisons']} comparaciones")
        
        # Paso 6: Análisis (intervalos de confianza, ranking)
        self.logger.info("Calculando intervalos de confianza y ranking...")
        ci_df = self.analyzer.calculate_confidence_intervals(results_df, self.metric)
        ci_df.to_csv(self.config.CONFIDENCE_INTERVALS_PATH, index=False)
        
        ranking_df = self.analyzer.generate_ranking(results_df, self.metric, self.higher_is_better)
        ranking_df.to_csv(self.config.RANKING_RESULTS_PATH, index=False)
        
        best_model = self.analyzer.get_best_model(ranking_df)
        self.logger.info(f"Mejor modelo: {best_model}")
        
        # Paso 7: Generar gráficos
        self.logger.info("Generando gráficos...")
        self.plotter.generate_all_plots(
            results_df=results_df,
            average_ranks=friedman_results["average_ranks"],
            critical_difference=nemenyi_results["critical_difference"] if nemenyi_results else 0.0,
            wilcoxon_df=wilcoxon_results["results_df"],
            ci_df=ci_df,
            metric=self.metric
        )
        
        # Paso 8: Generar reportes
        self.logger.info("Generando reportes...")
        self.reporter.generate_markdown(
            friedman_results=friedman_results,
            nemenyi_results=nemenyi_results if nemenyi_results else {"critical_difference": 0, "significant_pairs": [], "average_ranks": {}},
            wilcoxon_results=wilcoxon_results,
            ci_df=ci_df,
            ranking_df=ranking_df,
            best_model=best_model,
            metric=self.metric,
            alpha=self.alpha,
            save_path=self.config.STATISTICS_REPORT_MD
        )
        
        self.reporter.generate_html(
            md_path=self.config.STATISTICS_REPORT_MD,
            save_path=self.config.STATISTICS_REPORT_HTML
        )
        
        self.reporter.generate_conclusions(
            friedman_results=friedman_results,
            ranking_df=ranking_df,
            best_model=best_model,
            metric=self.metric,
            save_path=self.config.STATISTICS_CONCLUSIONS
        )
        
        # Guardar resultados combinados
        combined = {
            "friedman": friedman_results,
            "nemenyi": nemenyi_results,
            "wilcoxon": wilcoxon_results,
            "ci": ci_df,
            "ranking": ranking_df,
            "best_model": best_model
        }
        
        self.logger.info("=" * 80)
        self.logger.info("VALIDACIÓN ESTADÍSTICA COMPLETADA")
        self.logger.info("=" * 80)
        
        print(f"\n✅ Validación completada exitosamente!")
        print(f"📊 Mejor modelo: {best_model}")
        print(f"📄 Reportes guardados en: {self.config.REPORTS_DIR}")
        
        return combined


def main():
    """
    Función principal para ejecutar desde línea de comandos.
    """
    config = AIConfig()
    config.ensure_directories_exist()
    
    validator = StatisticalValidator(config)
    validator.run()


if __name__ == "__main__":
    main()
