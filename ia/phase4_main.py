#!/usr/bin/env python3
"""
Fase 4: Validación Cruzada, Ajuste de Hiperparámetros y Selección de Mejor Modelo
"""

import sys
import argparse
from pathlib import Path

# Añadir el directorio del proyecto al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ia.config.config import AIConfig
from ia.evaluation.cross_validation import CrossValidationRunner
from ia.utils.hyperparameter_tuner import HyperparameterTuner
from ia.evaluation.final_comparison import FinalComparator


def main():
    parser = argparse.ArgumentParser(
        description="Fase 4: Validación Cruzada, Tuning de Hiperparámetros y Comparación de Modelos"
    )
    parser.add_argument(
        "--step",
        type=str,
        default="all",
        choices=["cv", "tuning", "compare", "all"],
        help="Paso a ejecutar: cv (solo Cross Validation), tuning (solo Hyperparameter Tuning), compare (solo comparación final), o all (todos los pasos)"
    )
    
    args = parser.parse_args()
    
    # Cargar configuración
    config = AIConfig()
    config.ensure_directories_exist()
    
    print("="*80)
    print("FASE 4: VALIDACIÓN CRUZADA, TUNING DE HIPERPARÁMETROS Y SELECCIÓN DE MEJOR MODELO")
    print("="*80)
    
    try:
        if args.step == "cv":
            # Ejecutar solo Cross Validation
            print("\n>>> EJECUTANDO CROSS VALIDATION")
            cv_runner = CrossValidationRunner(config)
            cv_results, cv_stats = cv_runner.run_all()
            print("\n✅ Cross Validation completado!")
            
        elif args.step == "tuning":
            # Ejecutar solo Hyperparameter Tuning
            print("\n>>> EJECUTANDO HYPERPARAMETER TUNING")
            tuner = HyperparameterTuner(config)
            all_tuning_results = tuner.tune_all_models()
            print("\n✅ Hyperparameter Tuning completado!")
            
        elif args.step == "compare":
            # Ejecutar solo Comparación Final
            print("\n>>> EJECUTANDO COMPARACIÓN FINAL")
            comparator = FinalComparator(config)
            comparison_df, best_model, final_model = comparator.run()
            print("\n✅ Comparación final completada!")
            
        else:  # all
            # Ejecutar todos los pasos
            print("\n>>> PASO 1: CROSS VALIDATION")
            cv_runner = CrossValidationRunner(config)
            cv_results, cv_stats = cv_runner.run_all()
            
            print("\n>>> PASO 2: HYPERPARAMETER TUNING")
            tuner = HyperparameterTuner(config)
            all_tuning_results = tuner.tune_all_models()
            
            print("\n>>> PASO 3: COMPARACIÓN FINAL Y SELECCIÓN")
            comparator = FinalComparator(config)
            comparison_df, best_model, final_model = comparator.run()
            
        print("\n" + "="*80)
        print("FASE 4 COMPLETADA EXITOSAMENTE!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
