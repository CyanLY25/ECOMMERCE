"""
Utilities para el módulo de Validación Estadística.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from pathlib import Path


def load_cv_results(cv_path: Path) -> pd.DataFrame:
    """
    Carga resultados de Cross Validation desde CSV.
    
    Args:
        cv_path: Ruta al archivo CSV de Cross Validation.
        
    Returns:
        DataFrame con resultados.
    """
    if not cv_path.exists():
        raise FileNotFoundError(f"Archivo de Cross Validation no encontrado: {cv_path}")
    return pd.read_csv(cv_path)


def pivot_cv_results(results_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Transforma resultados de CV a formato ancho (modelos como columnas, folds como filas).
    
    Args:
        results_df: DataFrame con resultados de CV (columnas: model, fold, metric, ...).
        metric: Nombre de la métrica a usar (ej: 'rmse').
        
    Returns:
        DataFrame pivotado.
    """
    pivot_df = results_df.pivot(index="fold", columns="model", values=metric)
    return pivot_df


def get_model_names(results_df: pd.DataFrame) -> List[str]:
    """
    Obtiene lista de nombres de modelos únicos.
    
    Args:
        results_df: DataFrame de resultados.
        
    Returns:
        Lista de nombres de modelos.
    """
    return sorted(results_df["model"].unique())


def interpret_cohens_d(d: float) -> str:
    """
    Interpreta el tamaño del efecto Cohen's d.
    
    Args:
        d: Valor de Cohen's d.
        
    Returns:
        Cadena con la interpretación.
    """
    abs_d = abs(d)
    if abs_d < 0.2:
        return "Despreciable"
    elif abs_d < 0.5:
        return "Pequeño"
    elif abs_d < 0.8:
        return "Mediano"
    else:
        return "Grande"


def interpret_cliffs_delta(delta: float) -> str:
    """
    Interpreta el tamaño del efecto Cliff's Delta.
    
    Args:
        delta: Valor de Cliff's Delta.
        
    Returns:
        Cadena con la interpretación.
    """
    abs_delta = abs(delta)
    if abs_delta < 0.147:
        return "Despreciable"
    elif abs_delta < 0.33:
        return "Pequeño"
    elif abs_delta < 0.474:
        return "Mediano"
    else:
        return "Grande"


def format_table_for_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    """
    Formatea un DataFrame como tabla LaTeX para artículos científicos.
    
    Args:
        df: DataFrame a formatear.
        caption: Título de la tabla.
        label: Etiqueta de referencia.
        
    Returns:
        Cadena con código LaTeX.
    """
    latex_table = df.to_latex(
        index=False,
        caption=caption,
        label=label,
        float_format="%.4f"
    )
    return latex_table


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """
    División segura, evita división por cero.
    
    Args:
        a: Numerador.
        b: Denominador.
        default: Valor por defecto si b es cero.
        
    Returns:
        Resultado de la división o valor por defecto.
    """
    if b == 0:
        return default
    return a / b
