"""
Script principal para ejecutar validación cruzada en todos los modelos.
"""
import argparse
import json
import sys
from pathlib import Path

# Añadir el directorio raíz del proyecto al path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ia.config.config import AIConfig
from ia.training.common import load_datasets, separate_features_target
from ia.validation.cross_validation import (
    CrossValidator,
    plot_cross_validation_results,
    save_results
)
from ia.utils.logger import setup_logger


SUPPORTED_MODELS = ["mlp", "lstm", "gru", "cnn_lstm", "cnn_gru", "tft"]


def run_all_models_cross_validation(model_names=None):
    """
    Ejecuta validación cruzada K-Fold para todos los modelos del proyecto.
    """
    # Inicializar configuración y logger
    config = AIConfig()
    config.ensure_directories_exist()
    
    logger = setup_logger("cross_validation_runner", config.CV_LOG_PATH)
    
    logger.info("=" * 80)
    logger.info("INICIANDO VALIDACIÓN CRUZADA COMPLETA PARA TODOS LOS MODELOS")
    logger.info("=" * 80)
    
    try:
        # Cargar datos
        logger.info("Cargando datasets...")
        train_df, _, _ = load_datasets(config)
        X, y = separate_features_target(train_df, config.TARGET_VARIABLE)
        logger.info(f"Datos cargados: {X.shape[0]} muestras, {X.shape[1]} características")
        
        # Lista de modelos
        selected_models = list(model_names) if model_names else list(SUPPORTED_MODELS)
        invalid_models = sorted(set(selected_models) - set(SUPPORTED_MODELS))
        if invalid_models:
            raise ValueError(f"Modelos no reconocidos: {invalid_models}")
        all_results = {}
        failures = {}
        
        # Inicializar validador
        validator = CrossValidator(config)
        
        # Ejecutar CV para cada modelo
        for model_name in selected_models:
            try:
                fold_results, summary = validator.run(model_name, X, y)
                all_results[model_name] = {
                    "folds": fold_results,
                    "summary": summary
                }
            except Exception as e:
                logger.error(f"Error al ejecutar CV para {model_name}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                failures[model_name] = str(e)

        if failures:
            details = "; ".join(f"{model}: {error}" for model, error in failures.items())
            raise RuntimeError(f"La validación cruzada quedó incompleta: {details}")

        # Una ejecución aislada conserva los resultados anteriores de los
        # demás modelos para mantener completa la comparación estadística.
        if set(selected_models) != set(SUPPORTED_MODELS) and config.CV_RESULTS_JSON_PATH.exists():
            with open(config.CV_RESULTS_JSON_PATH, "r", encoding="utf-8") as file:
                previous_results = json.load(file)
            previous_results.update(all_results)
            all_results = previous_results
                
        # Guardar resultados
        logger.info("Guardando resultados...")
        save_results(all_results, config)
        logger.info(f"Resultados JSON guardados en: {config.CV_RESULTS_JSON_PATH}")
        logger.info(f"Resultados CSV guardados en: {config.CV_RESULTS_CSV_PATH}")
        logger.info(f"Resumen CSV guardado en: {config.CV_SUMMARY_CSV_PATH}")
        
        # Generar gráficos
        logger.info("Generando gráficos...")
        plot_cross_validation_results(all_results, config)
        logger.info(f"Gráficos guardados en: {config.FIGURES_DIR}")
        
        logger.info("=" * 80)
        logger.info("VALIDACIÓN CRUZADA COMPLETA FINALIZADA EXITOSAMENTE")
        logger.info("=" * 80)
        
        print("\n" + "=" * 80)
        print("VALIDACIÓN CRUZADA COMPLETA FINALIZADA EXITOSAMENTE")
        print("=" * 80)
        
        return all_results
        
    except Exception as e:
        logger.error(f"ERROR GENERAL: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Falló la validación cruzada: {e}") from e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validación temporal de modelos")
    parser.add_argument("--models", nargs="+", choices=SUPPORTED_MODELS)
    cli_args = parser.parse_args()
    run_all_models_cross_validation(cli_args.models)
