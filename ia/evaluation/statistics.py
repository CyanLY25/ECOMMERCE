import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from ia.config.config import AIConfig
from ia.utils.logger import setup_logger


logger = setup_logger("statistics")


class StatisticsAnalyzer:
    """
    Clase para calcular y guardar estadísticas descriptivas del dataset.
    """
    
    def __init__(self, config: AIConfig):
        """
        Inicializa el analizador de estadísticas.
        
        Args:
            config: Objeto de configuración.
        """
        self.config = config
        
    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula estadísticas descriptivas para todas las columnas numéricas.
        
        Args:
            df: DataFrame a analizar.
            
        Returns:
            DataFrame con las estadísticas calculadas.
        """
        logger.info("Calculando estadísticas descriptivas...")
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        stats_list = []
        
        for col in numeric_columns:
            col_data = df[col].dropna()
            stats = {
                "Variable": col,
                "Media": col_data.mean(),
                "Mediana": col_data.median(),
                "Moda": col_data.mode().iloc[0] if not col_data.mode().empty else np.nan,
                "Varianza": col_data.var(),
                "DesviaciónEstándar": col_data.std(),
                "Mínimo": col_data.min(),
                "Máximo": col_data.max(),
                "Percentil25": col_data.quantile(0.25),
                "Percentil50": col_data.quantile(0.5),
                "Percentil75": col_data.quantile(0.75)
            }
            stats_list.append(stats)
            
        stats_df = pd.DataFrame(stats_list)
        logger.info(f"Estadísticas calculadas para {len(numeric_columns)} variables.")
        return stats_df
        
    def save_statistics(self, stats_df: pd.DataFrame, output_path: Optional[Path] = None) -> None:
        """
        Guarda las estadísticas en un archivo CSV.
        
        Args:
            stats_df: DataFrame con las estadísticas.
            output_path: Ruta de salida. Si es None, usa la configuración.
        """
        if output_path is None:
            output_path = self.config.STATISTICS_PATH
            
        stats_df.to_csv(output_path, index=False, encoding="utf-8")
        logger.info(f"Estadísticas guardadas en {output_path}")
