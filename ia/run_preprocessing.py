#!/usr/bin/env python3
"""
Script para ejecutar el pipeline de preprocesamiento mejorado.
"""
import sys
from pathlib import Path

# Añadir el directorio raíz del proyecto al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ia.config.config import AIConfig
from ia.preprocessing.data_preprocessor import ImprovedDataPreprocessor

def main():
    # Cargar configuración y asegurar directorios existan
    config = AIConfig()
    config.ensure_directories_exist()
    
    # Crear preprocesador y ejecutar pipeline
    preprocessor = ImprovedDataPreprocessor(config)
    train, val, test = preprocessor.full_preprocessing_pipeline()
    
    # Verificar que se crearon todos los archivos
    expected_files = [
        config.TRAIN_DATA_PATH,
        config.VALIDATION_DATA_PATH,
        config.TEST_DATA_PATH,
        config.MODELS_DIR / "scaler.pkl",
        config.MODELS_DIR / "stockcode_encoder.pkl",
        config.MODELS_DIR / "country_encoder.pkl",
        config.MODELS_DIR / "customerid_encoder.pkl",
        config.MODELS_DIR / "mesnombre_encoder.pkl",
        config.REPORTS_DIR / "preprocessing_report.html",
        Path("ia/logs/preprocessing.log")
    ]
    
    print("\nVerificando archivos generados...")
    all_exist = True
    for file in expected_files:
        if file.exists():
            print(f"  OK {file}")
        else:
            print(f"  MISSING {file}")
            all_exist = False
            
    if all_exist:
        print("\n🎉 TODOS LOS ARCHIVOS SE GENERARON EXITOSAMENTE!")
    else:
        print("\n⚠️ Algunos archivos no se generaron!")
        sys.exit(1)
        
if __name__ == "__main__":
    main()
