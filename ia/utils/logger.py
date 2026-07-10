import logging
from pathlib import Path
from typing import Optional
from ia.config.config import AIConfig


def setup_logger(
    name: str = "preprocessing",
    log_file: Optional[Path] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Configura y devuelve un logger con formato estándar.
    
    Args:
        name: Nombre del logger.
        log_file: Ruta del archivo de log. Si es None, usa la ruta de config.
        level: Nivel de logging.
        
    Returns:
        Logger configurado.
    """
    # Asegurar que existan los directorios
    AIConfig.ensure_directories_exist()
    
    if log_file is None:
        log_file = AIConfig.LOG_FILE_PATH
        
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # Evitar duplicación de logs
    
    # Eliminar handlers existentes
    if logger.handlers:
        logger.handlers.clear()
        
    # Formato del log
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Handler para archivo
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Handler para consola
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger
