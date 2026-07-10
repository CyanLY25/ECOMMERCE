#!/usr/bin/env python3
"""
Script de entrenamiento exclusivo para el modelo MLP (Multilayer Perceptron).
"""
import sys
import os
from pathlib import Path

# Deshabilitar TensorBoard y reducir logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Mock tf.summary.scalar para evitar errores
import tensorflow as tf
def dummy_scalar(*args, **kwargs):
    pass
tf.summary.scalar = dummy_scalar
tf.summary.histogram = dummy_scalar
tf.summary.image = dummy_scalar

# Añadir el directorio raíz del proyecto al path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ia.config.config import AIConfig
from ia.training.mlp import MLPModel
from ia.training.common import load_datasets, separate_features_target, calculate_metrics
from ia.utils.logger import setup_logger


def main():
    """
    Función principal para ejecutar el entrenamiento del modelo MLP.
    """
    # Cargar configuración y asegurar directorios
    config = AIConfig()
    config.ensure_directories_exist()
    
    # Configurar logger
    logger = setup_logger("mlp_training", config.LOGS_DIR / "mlp_training.log")
    
    logger.info("=" * 80)
    logger.info("INICIANDO ENTRENAMIENTO DEL MODELO MLP")
    logger.info("=" * 80)
    
    try:
        # Cargar datasets
        logger.info("Cargando datasets...")
        train_df, val_df, test_df = load_datasets(config)
        logger.info(f"Datasets cargados: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        
        # Separar características y objetivo
        logger.info("Separando características y variable objetivo...")
        X_train, y_train = separate_features_target(train_df, config.TARGET_VARIABLE)
        X_val, y_val = separate_features_target(val_df, config.TARGET_VARIABLE)
        X_test, y_test = separate_features_target(test_df, config.TARGET_VARIABLE)
        logger.info(f"Características: {X_train.shape[1]}")
        
        # Crear y entrenar el modelo
        logger.info("Inicializando modelo MLP...")
        mlp = MLPModel(config)
        history, val_metrics, training_time = mlp.train(X_train, y_train, X_val, y_val)
        
        # Evaluar en conjunto de prueba
        logger.info("Evaluando modelo en conjunto de prueba...")
        y_test_pred = mlp.model.predict(X_test, verbose=0).flatten()
        test_metrics = calculate_metrics(y_test, y_test_pred)
        
        # Guardar métricas de prueba (combinadas con las de validación)
        final_metrics = {
            "mae": test_metrics["mae"],
            "mse": test_metrics["mse"],
            "rmse": test_metrics["rmse"],
            "mape": test_metrics["mape"],
            "r2": test_metrics["r2"],
            "loss": val_metrics["loss"],
            "val_loss": val_metrics["val_loss"],
            "training_time": training_time,
            "epochs_run": val_metrics["epochs_run"]
        }
        
        # Guardar métricas en ia/reports/mlp_metrics.json (como pide el usuario)
        import json
        reports_metrics_path = config.REPORTS_DIR / "mlp_metrics.json"
        with open(reports_metrics_path, "w", encoding="utf-8") as f:
            json.dump(final_metrics, f, indent=4)
        logger.info(f"Métricas guardadas en {reports_metrics_path}")
        
        # Mostrar resultados por consola
        print("\n" + "=" * 80)
        print("RESULTADOS DEL ENTRENAMIENTO MLP")
        print("=" * 80)
        print(f"MAE: {final_metrics['mae']:.4f}")
        print(f"RMSE: {final_metrics['rmse']:.4f}")
        print(f"MAPE: {final_metrics['mape']:.4f}")
        print(f"R²: {final_metrics['r2']:.4f}")
        print(f"Tiempo de entrenamiento: {final_metrics['training_time']:.2f} segundos")
        print(f"Epochs ejecutadas: {final_metrics['epochs_run']}")
        print("=" * 80 + "\n")
        
        # Verificar que existan los archivos esperados
        logger.info("Verificando archivos generados...")
        expected_files = [
            config.MLP_MODEL_PATH,
            config.MLP_HISTORY_PATH,
            reports_metrics_path,
            config.LOGS_DIR / "mlp_training.log"
        ]
        for file_path in expected_files:
            if file_path.exists():
                logger.info(f"✅ Archivo encontrado: {file_path}")
            else:
                logger.error(f"❌ Archivo NO encontrado: {file_path}")
        
        # Verificar que el modelo se puede cargar
        logger.info("Verificando que el modelo se puede cargar...")
        import tensorflow as tf
        loaded_model = tf.keras.models.load_model(config.MLP_MODEL_PATH)
        logger.info("✅ Modelo cargado exitosamente desde .keras")
        
        logger.info("=" * 80)
        logger.info("ENTRENAMIENTO DEL MLP COMPLETADO EXITOSAMENTE")
        logger.info("=" * 80)
        
        return final_metrics
        
    except Exception as e:
        logger.error(f"ERROR durante el entrenamiento: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
