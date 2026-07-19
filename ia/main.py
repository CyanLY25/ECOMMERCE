#!/usr/bin/env python3
"""
Punto de entrada principal del pipeline maestro del proyecto de predicción de demanda.
"""
import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

# Al ejecutar `python ia/main.py`, Python añade `ia/` al sys.path. Esa ruta
# haría que `ia/statistics` ocultase al módulo estándar `statistics`, usado por
# seaborn. Se conserva únicamente la raíz del proyecto para importar `ia.*`.
script_dir = Path(__file__).parent.resolve()
sys.path = [
    entry for entry in sys.path
    if Path(entry or ".").resolve() != script_dir
]
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from ia.config.config import AIConfig


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline de predicción de demanda")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--tft-only",
        action="store_true",
        help="Entrena y evalúa únicamente TFT con la configuración definitiva",
    )
    mode.add_argument(
        "--tft-smoke-test",
        action="store_true",
        help="Prueba TFT durante 1 época sin sobrescribir artefactos definitivos",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    """
    Función principal para ejecutar el pipeline maestro.
    """
    args = _parse_args(argv)
    if args.tft_only or args.tft_smoke_test:
        from ia.training.tft_trainer import main as train_tft

        tft_args = ["--smoke-test"] if args.tft_smoke_test else []
        return train_tft(tft_args)

    from ia.pipeline.pipeline_manager import PipelineManager

    config = AIConfig()
    manager = PipelineManager(config)
    return manager.run()


if __name__ == "__main__":
    main()
