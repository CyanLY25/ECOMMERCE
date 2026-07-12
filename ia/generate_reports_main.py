#!/usr/bin/env python3
"""
Punto de entrada para generar los reportes finales (PDF, Word, Excel)
a partir de todos los artefactos ya producidos por el pipeline
(EDA, entrenamiento, cross validation, tuning, pruebas estadísticas).

Uso:
    python -m ia.generate_reports_main
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ia.config.config import AIConfig
from ia.reports.report_generator import ReportGenerator
from ia.reports.report_data import load_report_data


def main():
    config = AIConfig()
    config.ensure_directories_exist()

    print("=" * 80)
    print("GENERACIÓN DE REPORTES FINALES (PDF, WORD, EXCEL)")
    print("=" * 80)

    data = load_report_data(config)
    generator = ReportGenerator(config)

    try:
        paths = generator.generate_all(data)
        print("\n✅ Reportes generados correctamente:")
        for fmt, path in paths.items():
            print(f"  - {fmt.upper()}: {path}")
    except Exception as e:
        print(f"\n❌ ERROR generando reportes: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("=" * 80)


if __name__ == "__main__":
    main()
