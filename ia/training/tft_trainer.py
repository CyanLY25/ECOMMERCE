#!/usr/bin/env python3
"""Entrenamiento y prueba aislada del Temporal Fusion Transformer (TFT)."""
import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

# Añadir el directorio raíz del proyecto al path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ia.config.config import AIConfig
from ia.training.tft import TFTModel, load_tft_model
from ia.training.common import (
    calculate_metrics,
    create_sequences,
    load_datasets,
    save_metrics,
    separate_features_target,
)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena y verifica únicamente el modelo TFT")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Ejecuta 1 época con una muestra pequeña y guarda artefactos separados",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Sobrescribe temporalmente TFT_EPOCHS para esta ejecución",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        help="Máximo de secuencias por split; útil para una comprobación rápida",
    )
    return parser.parse_args(argv)


def _configure_smoke_paths(config: AIConfig) -> None:
    """Aísla los artefactos de prueba para no sobrescribir un TFT definitivo."""
    smoke_dir = config.IA_DIR / "smoke_test" / "tft"
    models_dir = smoke_dir / "models"
    reports_dir = smoke_dir / "reports"
    figures_dir = reports_dir / "figures"
    logs_dir = smoke_dir / "logs"

    for directory in (models_dir, reports_dir, figures_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    config.MODELS_DIR = models_dir
    config.REPORTS_DIR = reports_dir
    config.FIGURES_DIR = figures_dir
    config.LOGS_DIR = logs_dir
    config.TFT_MODEL_PATH = models_dir / "tft.keras"
    config.TFT_HISTORY_PATH = models_dir / "tft_history.pkl"
    config.TFT_METRICS_PATH = models_dir / "tft_metrics.json"
    config.TFT_LOG_PATH = logs_dir / "tft.log"


def _validate_processed_schema(dataframe, config: AIConfig, split_name: str) -> None:
    expected = [config.MODEL_FEATURES[0], config.TARGET_VARIABLE, *config.MODEL_FEATURES[1:]]
    actual = list(dataframe.columns)
    if actual != expected:
        raise ValueError(
            f"El esquema de {split_name} no coincide con el esperado. "
            f"Esperado: {expected}. Recibido: {actual}"
        )


def _limit_sequences(
    X: np.ndarray,
    y: np.ndarray,
    max_sequences: Optional[int],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if max_sequences is None or len(X) <= max_sequences:
        return X, y
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(X), size=max_sequences, replace=False))
    return X[indices], y[indices]


def main(argv: Optional[Sequence[str]] = None):
    args = _parse_args(argv)
    config = AIConfig()
    config.ensure_directories_exist()

    if args.epochs is not None and args.epochs <= 0:
        raise ValueError("--epochs debe ser mayor que cero")
    if args.max_sequences is not None and args.max_sequences <= 0:
        raise ValueError("--max-sequences debe ser mayor que cero")

    if args.smoke_test:
        _configure_smoke_paths(config)
        config.TFT_EPOCHS = args.epochs or 1
        max_sequences = args.max_sequences or 1024
        print("Modo smoke test: los artefactos definitivos no serán sobrescritos.")
    else:
        if args.epochs is not None:
            config.TFT_EPOCHS = args.epochs
        max_sequences = args.max_sequences

    train_df, val_df, test_df = load_datasets(config)
    for split_name, dataframe in (
        ("train", train_df),
        ("validation", val_df),
        ("test", test_df),
    ):
        _validate_processed_schema(dataframe, config, split_name)

    X_train, y_train = separate_features_target(train_df, config.TARGET_VARIABLE)
    X_val, y_val = separate_features_target(val_df, config.TARGET_VARIABLE)
    X_test, y_test = separate_features_target(test_df, config.TARGET_VARIABLE)

    X_train_seq, y_train_seq = create_sequences(X_train, y_train, config.TFT_WINDOW_SIZE)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, config.TFT_WINDOW_SIZE)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, config.TFT_WINDOW_SIZE)

    X_train_seq, y_train_seq = _limit_sequences(
        X_train_seq, y_train_seq, max_sequences, config.RANDOM_SEED
    )
    X_val_seq, y_val_seq = _limit_sequences(
        X_val_seq, y_val_seq, max_sequences, config.RANDOM_SEED + 1
    )
    X_test_seq, y_test_seq = _limit_sequences(
        X_test_seq, y_test_seq, max_sequences, config.RANDOM_SEED + 2
    )

    print(
        "Secuencias TFT: "
        f"train={X_train_seq.shape}, validation={X_val_seq.shape}, test={X_test_seq.shape}"
    )

    tft = TFTModel(config)
    _, validation_metrics, training_time = tft.train(
        X_train_seq, y_train_seq, X_val_seq, y_val_seq
    )

    print("\n" + "=" * 60)
    print("RESULTADOS DEL MODELO TFT")
    print("=" * 60)
    y_test_pred = tft.model.predict(X_test_seq, verbose=0).flatten()
    test_metrics = calculate_metrics(y_test_seq, y_test_pred)

    print(f"MAE: {test_metrics['mae']:.4f}")
    print(f"RMSE: {test_metrics['rmse']:.4f}")
    print(f"MAPE: {test_metrics['mape']:.4f}")
    print(f"R²: {test_metrics['r2']:.4f}")
    print(f"Tiempo de entrenamiento: {training_time:.2f} s")
    print(f"Épocas ejecutadas: {validation_metrics['epochs_run']}")
    print("=" * 60)

    expected_files = [
        config.TFT_MODEL_PATH,
        config.TFT_HISTORY_PATH,
        config.TFT_METRICS_PATH,
        config.TFT_LOG_PATH,
        config.FIGURES_DIR / "tft_loss.png",
        config.FIGURES_DIR / "tft_mae.png",
    ]
    missing_files = [path for path in expected_files if not path.exists()]
    if missing_files:
        raise RuntimeError(f"No se generaron los artefactos TFT: {missing_files}")

    # Esta carga explícita prueba la serialización usando los custom objects.
    loaded_model = load_tft_model(config.TFT_MODEL_PATH)
    reloaded_predictions = loaded_model.predict(X_test_seq[:8], verbose=0).flatten()
    if not np.allclose(y_test_pred[:8], reloaded_predictions, rtol=1e-5, atol=1e-5):
        raise RuntimeError("Las predicciones cambiaron después de volver a cargar el TFT")
    print(f"Modelo TFT guardado y recargado correctamente: {config.TFT_MODEL_PATH}")

    final_metrics = {
        **validation_metrics,
        **test_metrics,
        "validation_mae": validation_metrics["mae"],
        "validation_rmse": validation_metrics["rmse"],
        "validation_mape": validation_metrics["mape"],
        "validation_r2": validation_metrics["r2"],
        "training_time": training_time,
    }
    report_metrics_path = config.REPORTS_DIR / "tft_metrics.json"
    save_metrics(final_metrics, report_metrics_path)
    print(f"Métricas de test guardadas en: {report_metrics_path}")
    return final_metrics


if __name__ == "__main__":
    main()
