"""
Script principal para ejecutar Hyperparameter Tuning en todos los modelos.
"""
import sys
from pathlib import Path

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


def run_all_models_tuning():
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
        models = ['mlp', 'lstm', 'gru', 'cnn_lstm', 'cnn_gru']
        
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
                continue
        
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
        sys.exit(1)


if __name__ == "__main__":
    run_all_models_tuning()
