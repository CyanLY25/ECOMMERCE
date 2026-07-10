import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List
from ia.utils.logger import setup_logger

logger = setup_logger("outlier_detector")


class OutlierDetector:
    """
    Clase para detectar y eliminar outliers usando el método IQR.
    """
    
    def __init__(self):
        """Inicializa el detector de outliers."""
        self.outlier_stats: Dict[str, Dict[str, Any]] = {}
        
    def detect_iqr(self, df: pd.DataFrame, columns: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Detecta outliers usando el método IQR para múltiples columnas.
        
        Args:
            df: DataFrame con los datos.
            columns: Lista de columnas a analizar.
            
        Returns:
            Diccionario con estadísticas de outliers por columna.
        """
        outlier_info = {}
        
        for col in columns:
            if col not in df.columns:
                logger.warning(f"Columna {col} no encontrada en el DataFrame")
                continue
                
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            count_outliers = len(outliers)
            percentage_outliers = (count_outliers / len(df)) * 100
            
            outlier_info[col] = {
                "Q1": Q1,
                "Q3": Q3,
                "IQR": IQR,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "count_outliers": count_outliers,
                "percentage_outliers": percentage_outliers
            }
            
            logger.info(f"Columna {col}: {count_outliers} outliers ({percentage_outliers:.2f}%)")
            
        self.outlier_stats = outlier_info
        return outlier_info
        
    def remove_outliers(self, df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Elimina outliers de las columnas especificadas usando IQR.
        
        Args:
            df: DataFrame original.
            columns: Lista de columnas para eliminar outliers.
            
        Returns:
            Tupla con (DataFrame sin outliers, DataFrame con outliers eliminados).
        """
        logger.info("Eliminando outliers usando IQR...")
        
        df_clean = df.copy()
        outliers_mask = pd.Series([False] * len(df), index=df.index)
        
        for col in columns:
            if col not in df.columns:
                continue
                
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            col_outliers = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
            outliers_mask = outliers_mask | col_outliers
            
        df_outliers = df[outliers_mask]
        df_clean = df_clean[~outliers_mask]
        
        logger.info(f"Eliminados {len(df_outliers)} outliers.")
        logger.info(f"Registros restantes: {len(df_clean)}")
        
        return df_clean, df_outliers
