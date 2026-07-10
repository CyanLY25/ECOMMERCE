#!/usr/bin/env python3
"""
Módulo para comparar y seleccionar el mejor modelo de predicción de demanda.
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

# Añadir el directorio raíz del proyecto al path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ia.config.config import AIConfig
from ia.utils.logger import setup_logger


def get_model_metrics(config: AIConfig, model_name: str) -> Dict[str, float]:
    """
    Lee las métricas de un modelo desde su archivo JSON.
    """
    # Mapear nombres de modelo a nombres de archivo (usa underscores en lugar de guiones)
    model_file_name = model_name.lower().replace("-", "_")
    metrics_path = config.REPORTS_DIR / f"{model_file_name}_metrics.json"
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    
    return {
        "Modelo": model_name,
        "MAE": float(metrics.get("mae", 0)),
        "RMSE": float(metrics.get("rmse", 0)),
        "MSE": float(metrics.get("mse", 0)),
        "MAPE": float(metrics.get("mape", 0)),
        "R²": float(metrics.get("r2", 0)),
        "Tiempo": float(metrics.get("training_time", 0)),
        "Épocas": int(metrics.get("epochs_run", 0))
    }


def main():
    """
    Función principal para comparar y seleccionar el mejor modelo.
    """
    config = AIConfig()
    config.ensure_directories_exist()
    logger = setup_logger("model_selector", config.LOGS_DIR / "model_selector.log")
    
    logger.info("=" * 80)
    logger.info("INICIANDO SELECCIÓN DE MEJOR MODELO")
    logger.info("=" * 80)
    
    # Lista de modelos a evaluar
    model_names = ["MLP", "LSTM", "GRU", "CNN-LSTM", "CNN-GRU"]
    model_paths = {
        "MLP": config.MLP_MODEL_PATH,
        "LSTM": config.LSTM_MODEL_PATH,
        "GRU": config.GRU_MODEL_PATH,
        "CNN-LSTM": config.CNN_LSTM_MODEL_PATH,
        "CNN-GRU": config.CNN_GRU_MODEL_PATH
    }
    
    # Cargar métricas de cada modelo
    all_metrics = []
    for model_name in model_names:
        try:
            metrics = get_model_metrics(config, model_name)
            all_metrics.append(metrics)
            logger.info(f"Cargadas métricas para {model_name}")
        except Exception as e:
            logger.error(f"Error al leer métricas de {model_name}: {e}")
            raise RuntimeError(f"Faltan métricas para {model_name}") from e
    
    # Crear DataFrame y ordenar por RMSE, luego MAE, luego R²
    df = pd.DataFrame(all_metrics)
    df = df.sort_values(by=["RMSE", "MAE", "R²"], ascending=[True, True, False])
    df = df.reset_index(drop=True)
    
    # Obtener el mejor modelo
    best_model = df.iloc[0]
    best_model_name = best_model["Modelo"]
    best_model_path = str(model_paths[best_model_name])
    
    # Guardar comparación CSV
    comparison_path = config.MODEL_COMPARISON_PATH
    df.to_csv(comparison_path, index=False, encoding="utf-8")
    logger.info(f"Comparación guardada en {comparison_path}")
    
    # Guardar best_model.json
    best_model_data = {
        "model": best_model_name,
        "path": best_model_path,
        "rmse": best_model["RMSE"],
        "mae": best_model["MAE"],
        "mape": best_model["MAPE"],
        "r2": best_model["R²"]
    }
    best_model_path_json = config.BEST_MODEL_PATH
    with open(best_model_path_json, "w", encoding="utf-8") as f:
        json.dump(best_model_data, f, indent=4, ensure_ascii=False)
    logger.info(f"Mejor modelo guardado en {best_model_path_json}")
    
    # Guardar ranking
    ranking_path = config.REPORTS_DIR / "model_ranking.txt"
    with open(ranking_path, "w", encoding="utf-8") as f:
        f.write("RANKING DE MODELOS (por RMSE)\n")
        f.write("=" * 40 + "\n")
        for idx, row in df.iterrows():
            f.write(f"{idx + 1}. {row['Modelo']}\n")
    logger.info(f"Ranking guardado en {ranking_path}")
    
    # Mostrar resultados por consola
    print("\n" + "=" * 50)
    print("COMPARACIÓN DE MODELOS")
    print("=" * 50)
    print(df[["Modelo", "RMSE", "MAE", "R²"]].to_string(index=False))
    print("\n" + "-" * 50)
    print("MEJOR MODELO:")
    print(f"{best_model_name}")
    print(f"\nRMSE: {best_model['RMSE']:.4f}")
    print(f"MAE: {best_model['MAE']:.4f}")
    print(f"R²: {best_model['R²']:.4f}")
    print(f"\nArchivo: {best_model_path}")
    print("=" * 50 + "\n")
    
    logger.info("=" * 80)
    logger.info("SELECCIÓN DE MEJOR MODELO COMPLETADA")
    logger.info(f"Mejor modelo: {best_model_name}")
    logger.info(f"RMSE: {best_model['RMSE']:.4f}")
    logger.info("=" * 80)
    
    # Verificar archivos generados
    expected_files = [
        config.MODEL_COMPARISON_PATH,
        config.BEST_MODEL_PATH,
        config.REPORTS_DIR / "model_ranking.txt"
    ]
    logger.info("Verificando archivos generados:")
    for file in expected_files:
        if file.exists():
            logger.info(f"  ✅ {file}")
        else:
            logger.error(f"  ❌ {file}")
            raise RuntimeError(f"Falta el archivo {file}")
    
    logger.info("Todos los archivos generados correctamente!")


if __name__ == "__main__":
    main()
