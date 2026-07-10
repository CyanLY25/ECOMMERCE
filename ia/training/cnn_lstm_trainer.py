#!/usr/bin/env python3
"""
Script para ejecutar el entrenamiento del modelo CNN-LSTM.
"""
import sys
from pathlib import Path

# Añadir el directorio raíz del proyecto al path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ia.config.config import AIConfig
from ia.training.cnn_lstm import CNNLSTMModel
from ia.training.common import (
    load_datasets,
    separate_features_target,
    create_sequences
)


def main():
    # Cargar configuración y asegurar directorios existan
    config = AIConfig()
    config.ensure_directories_exist()
    
    # Cargar datos
    train_df, val_df, test_df = load_datasets(config)
    
    # Separar características y objetivo
    X_train, y_train = separate_features_target(train_df, config.TARGET_VARIABLE)
    X_val, y_val = separate_features_target(val_df, config.TARGET_VARIABLE)
    X_test, y_test = separate_features_target(test_df, config.TARGET_VARIABLE)
    
    # Generar secuencias
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, config.CNN_LSTM_WINDOW_SIZE)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, config.CNN_LSTM_WINDOW_SIZE)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, config.CNN_LSTM_WINDOW_SIZE)
    
    # Entrenar
    cnn_lstm = CNNLSTMModel(config)
    history, metrics, training_time = cnn_lstm.train(X_train_seq, y_train_seq, X_val_seq, y_val_seq)
    
    # Evaluar en test
    print("\n" + "="*60)
    print("RESULTADOS DEL MODELO CNN-LSTM")
    print("="*60)
    y_test_pred = cnn_lstm.model.predict(X_test_seq, verbose=0).flatten()
    
    from ia.training.common import calculate_metrics
    test_metrics = calculate_metrics(y_test_seq, y_test_pred)
    
    print(f"MAE: {test_metrics['mae']:.4f}")
    print(f"RMSE: {test_metrics['rmse']:.4f}")
    print(f"MAPE: {test_metrics['mape']:.4f}")
    print(f"R²: {test_metrics['r2']:.4f}")
    print(f"Tiempo de entrenamiento: {training_time:.2f} s")
    print(f"Épocas ejecutadas: {metrics['epochs_run']}")
    print("="*60)
    
    # Verificar archivos generados
    print("\nVerificando archivos generados:")
    expected_files = [
        config.CNN_LSTM_MODEL_PATH,
        config.CNN_LSTM_HISTORY_PATH,
        config.CNN_LSTM_METRICS_PATH,
        config.CNN_LSTM_LOG_PATH,
        config.FIGURES_DIR / "cnn_lstm_loss.png",
        config.FIGURES_DIR / "cnn_lstm_mae.png"
    ]
    
    all_ok = True
    for file in expected_files:
        if file.exists():
            print(f"  OK: {file}")
        else:
            print(f"  MISSING: {file}")
            all_ok = False
    
    # Verificar que el modelo se pueda cargar
    import tensorflow as tf
    try:
        loaded_model = tf.keras.models.load_model(config.CNN_LSTM_MODEL_PATH)
        print("\n✅ El modelo se cargó exitosamente!")
    except Exception as e:
        print(f"\n❌ Error al cargar el modelo: {e}")
        all_ok = False
        sys.exit(1)
    
    # Guardar métricas de test en la ruta correcta (ia/reports/)
    from ia.training.common import save_metrics
    final_metrics = {**test_metrics, **metrics}
    save_metrics(final_metrics, config.REPORTS_DIR / "cnn_lstm_metrics.json")
    
    return final_metrics


if __name__ == "__main__":
    main()
