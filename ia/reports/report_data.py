"""
Utilidades para cargar todos los artefactos generados por el pipeline
(EDA, entrenamiento, cross validation, tuning, pruebas estadísticas)
en una sola estructura de datos lista para alimentar los reportes
PDF / Word / Excel.
"""
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from ia.config.config import AIConfig
from ia.utils.logger import setup_logger

logger = setup_logger("report_data")


def _read_csv(path: Path) -> Optional[pd.DataFrame]:
    if path and Path(path).exists():
        try:
            return pd.read_csv(path)
        except Exception as e:
            logger.warning(f"No se pudo leer {path}: {e}")
    return None


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if path and Path(path).exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"No se pudo leer {path}: {e}")
    return None


def _read_text(path: Path) -> Optional[str]:
    if path and Path(path).exists():
        try:
            return Path(path).read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"No se pudo leer {path}: {e}")
    return None


def load_report_data(config: AIConfig) -> Dict[str, Any]:
    """
    Recolecta en un diccionario todos los resultados numéricos y rutas de
    figuras que necesitan los reportes finales (PDF, Word, Excel).

    No lanza excepción si falta algún archivo: cada sección faltante
    queda como None / DataFrame vacío y los generadores de reporte lo
    señalan explícitamente en el documento, en vez de fallar en silencio.
    """
    data: Dict[str, Any] = {}

    # --- EDA ---
    data["eda_statistics"] = _read_csv(config.STATISTICS_PATH)

    # --- Comparación de modelos (entrenamiento inicial) ---
    data["model_comparison"] = _read_csv(config.MODEL_COMPARISON_PATH)
    data["best_model"] = _read_json(config.BEST_MODEL_PATH)

    # --- Cross Validation ---
    data["cv_summary"] = _read_csv(config.CV_SUMMARY_CSV_PATH)
    data["cv_results"] = _read_csv(config.CV_RESULTS_CSV_PATH)

    # --- Hyperparameter Tuning ---
    data["best_hyperparameters"] = _read_json(config.TUNING_BEST_CONFIG_PATH)
    data["tuning_results"] = _read_csv(config.TUNING_RESULTS_CSV_PATH)

    # --- Pruebas Estadísticas ---
    data["friedman"] = _read_csv(config.FRIEDMAN_RESULTS_PATH)
    data["wilcoxon"] = _read_csv(config.WILCOXON_RESULTS_PATH)
    data["nemenyi"] = _read_csv(config.NEMENYI_RESULTS_PATH)
    data["ranking"] = _read_csv(config.RANKING_RESULTS_PATH)
    data["confidence_intervals"] = _read_csv(config.CONFIDENCE_INTERVALS_PATH)
    data["statistical_conclusions"] = _read_text(config.STATISTICS_CONCLUSIONS)

    # --- Figuras clave para incrustar en los reportes ---
    figures_dir = config.FIGURES_DIR
    key_figures = [
        "5_correlation_heatmap.png",
        "13_time_series_sales.png",
        "9_top_products.png",
        "11_country_distribution.png",
        "cross_validation_boxplot.png",
        "cross_validation_rmse.png",
        "significance_heatmap.png",
        "critical_difference.png",
        "ranking_plot.png",
        "tuning_comparison.png",
    ]
    data["figures"] = {
        name: (figures_dir / name)
        for name in key_figures
        if (figures_dir / name).exists()
    }

    data["generated_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    return data
