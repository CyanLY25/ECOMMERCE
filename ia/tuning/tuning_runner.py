"""
Script principal para ejecutar Hyperparameter Tuning en todos los modelos.
"""
import argparse
import json
import sys
from pathlib import Path
import pandas as pd

# Añadir el directorio raíz del proyecto al path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ia.config.config import AIConfig
from ia.training.common import (
    load_datasets,
    separate_features_target,
    create_sequences
)
from ia.tuning.hyperparameter_tuner import HyperparameterTuner
from ia.utils.logger import setup_logger


SUPPORTED_MODELS = ['mlp', 'lstm', 'gru', 'cnn_lstm', 'cnn_gru', 'tft']


def run_all_models_tuning(model_names=None):
    """
    Ejecuta Hyperparameter Tuning para todos los modelos del proyecto.
    """
    # Inicializar configuración
    config = AIConfig()
    config.ensure_directories_exist()
    
    logger = setup_logger("tuning_runner", config.TUNING_LOG_PATH)
    
    logger.info("=" * 80)
    logger.info("INICIANDO HYPERPARAMETER TUNING PARA TODOS LOS MODELOS")
    logger.info("=" * 80)
    
    try:
        # Cargar datos
        logger.info("Cargando datasets...")
        train_df, val_df, _ = load_datasets(config)
        
        # Separar features y target
        X_train, y_train = separate_features_target(train_df, config.TARGET_VARIABLE)
        X_val, y_val = separate_features_target(val_df, config.TARGET_VARIABLE)
        
        # Inicializar tuner
        tuner = HyperparameterTuner(config)
        
        # Lista de modelos
        models = list(model_names) if model_names else list(SUPPORTED_MODELS)
        invalid_models = sorted(set(models) - set(SUPPORTED_MODELS))
        if invalid_models:
            raise ValueError(f"Modelos no reconocidos: {invalid_models}")

        # Una ejecución parcial conserva los mejores resultados y trials de
        # los modelos que ya fueron ajustados anteriormente.
        if set(models) != set(SUPPORTED_MODELS):
            if config.TUNING_RESULTS_JSON_PATH.exists():
                with open(config.TUNING_RESULTS_JSON_PATH, "r", encoding="utf-8") as file:
                    tuner.all_results.update(json.load(file))
            if config.TUNING_RESULTS_CSV_PATH.exists():
                previous_trials = pd.read_csv(config.TUNING_RESULTS_CSV_PATH)
                previous_trials = previous_trials[
                    ~previous_trials["model"].str.lower().isin(models)
                ]
                tuner.trials_data.extend(previous_trials.to_dict(orient="records"))
        failures = {}
        
        # Preparar datos para cada tipo de modelo
        for model_name in models:
            if model_name == 'mlp':
                # Usar datos normales para MLP
                X_t = X_train
                X_v = X_val
                y_t = y_train
                y_v = y_val
            else:
                # Preparar secuencias para modelos recurrentes
                window_size = (
                    config.LSTM_SEQUENCE_LENGTH if model_name == 'lstm'
                    else config.GRU_SEQUENCE_LENGTH if model_name == 'gru'
                    else config.CNN_LSTM_SEQUENCE_LENGTH if model_name == 'cnn_lstm'
                    else config.TFT_WINDOW_SIZE if model_name == 'tft'
                    else config.CNN_GRU_WINDOW_SIZE
                )
                X_t, y_t = create_sequences(X_train, y_train, window_size)
                X_v, y_v = create_sequences(X_val, y_val, window_size)
            
            # Ejecutar tuning
            try:
                tuner.tune(model_name, X_t, y_t, X_v, y_v)
            except Exception as e:
                logger.error(f"Error en tuning de {model_name}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                failures[model_name] = str(e)

        if failures:
            details = "; ".join(f"{model}: {error}" for model, error in failures.items())
            raise RuntimeError(f"El tuning quedó incompleto: {details}")
        
        # Guardar y visualizar resultados
        tuner.save_results()
        tuner.generate_plots()
        
        # Resumen final
        logger.info("=" * 80)
        logger.info("RESUMEN DE MEJORES CONFIGURACIONES")
        logger.info("=" * 80)
        
        print("\n" + "=" * 80)
        print("RESUMEN DE MEJORES CONFIGURACIONES")
        print("=" * 80)
        
        for model_name, result in tuner.all_results.items():
            logger.info(f"\n{model_name}:")
            logger.info(f"  Mejor Loss: {result['best_score']:.6f}")
            logger.info(f"  Parámetros: {result['best_parameters']}")
            
            print(f"\n{model_name}:")
            print(f"  Mejor Loss: {result['best_score']:.6f}")
            print(f"  Parámetros: {result['best_parameters']}")
        
        logger.info("=" * 80)
        logger.info("HYPERPARAMETER TUNING COMPLETADO EXITOSAMENTE")
        logger.info("=" * 80)
        
        print("\n" + "=" * 80)
        print("HYPERPARAMETER TUNING COMPLETADO EXITOSAMENTE")
        print("=" * 80)
        
        return tuner.all_results
        
    except Exception as e:
        logger.error(f"ERROR GENERAL: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Falló el hyperparameter tuning: {e}") from e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ajuste de hiperparámetros por modelo")
    parser.add_argument("--models", nargs="+", choices=SUPPORTED_MODELS)
    cli_args = parser.parse_args()
    run_all_models_tuning(cli_args.models)
