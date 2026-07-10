import numpy as np
from typing import Any, List, Dict, Tuple


class CrossValidator:
    """
    Clase para realizar validación cruzada de modelos de machine learning.
    """

    def __init__(self, config):
        """
        Inicializa el validador cruzado con la configuración.
        
        Args:
            config: Objeto de configuración con el número de folds.
        """
        self.config = config
        self.num_folds = config.NUM_FOLDS

    def cross_validation(self, model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, List[float]]:
        """
        Realiza validación cruzada K-fold en el modelo.
        
        Args:
            model: Modelo a evaluar. Debe tener métodos fit() y predict().
            X: Datos de entrada.
            y: Etiquetas/valores objetivo.
            
        Returns:
            Diccionario con las métricas de cada fold.
        """
        raise NotImplementedError("Método cross_validation() no implementado aún.")
