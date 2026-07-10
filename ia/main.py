#!/usr/bin/env python3
"""
Punto de entrada principal del pipeline maestro del proyecto de predicción de demanda.
"""
import sys
from pathlib import Path

# Añadir el directorio raíz del proyecto al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ia.config.config import AIConfig
from ia.pipeline.pipeline_manager import PipelineManager


def main():
    """
    Función principal para ejecutar el pipeline maestro.
    """
    config = AIConfig()
    manager = PipelineManager(config)
    manager.run()


if __name__ == "__main__":
    main()
