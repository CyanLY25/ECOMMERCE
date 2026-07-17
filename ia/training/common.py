import os
import random
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from typing import Tuple, Dict, Any, List
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from ia.config.config import AIConfig
from ia.utils.logger import setup_logger


def set_seed(config: AIConfig):
    """
    Fija todas las semillas aleatorias para garantizar reproducibilidad.
    
    Args:
        config: Objeto de configuración con la semilla aleatoria.
    """
    seed = config.RANDOM_SEED
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


def load_datasets(config: AIConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Carga los datasets de entrenamiento, validación y prueba.
    
    Args:
        config: Objeto de configuración con las rutas de los datasets.
        
    Returns:
        Tupla con (train_df, val_df, test_df).
        
    Raises:
        FileNotFoundError: Si alguno de los archivos no existe.
    """
    # Verificar que existan los archivos
    for path in [config.TRAIN_DATA_PATH, config.VALIDATION_DATA_PATH, config.TEST_DATA_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"No se encontró el archivo: {path}")
            
    # Cargar datasets
    train_df = pd.read_csv(config.TRAIN_DATA_PATH)
    val_df = pd.read_csv(config.VALIDATION_DATA_PATH)
    test_df = pd.read_csv(config.TEST_DATA_PATH)
    
    return train_df, val_df, test_df


def separate_features_target(df: pd.DataFrame, target_col: str, drop_cols: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Separa las características de la variable objetivo.
    
    Args:
        df: DataFrame completo.
        target_col: Nombre de la columna objetivo.
        drop_cols: Lista de columnas a eliminar (opcional).
        
    Returns:
        Tupla con (X, y).
    """
    if drop_cols is None:
        drop_cols = []
        
    # Eliminar columnas no necesarias
    X = df.drop(columns=[target_col] + drop_cols, errors='ignore')
    # Eliminar columnas de tipo object (strings)
    X = X.select_dtypes(exclude=['object'])
    y = df[target_col].values
    
    return X.values.astype(np.float32), y.astype(np.float32)


def get_callbacks(config: AIConfig, model_name: str) -> List[tf.keras.callbacks.Callback]:
    """
    Obtiene la lista de callbacks para el entrenamiento.
    
    Args:
        config: Objeto de configuración.
        model_name: Nombre del modelo para nombrar los archivos.
        
    Returns:
        Lista de callbacks de Keras.
    """
    callbacks = []
    
    # Model Checkpoint
    model_path = getattr(config, f"{model_name.upper()}_MODEL_PATH")
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        filepath=model_path,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    ))
    
    # Early Stopping
    callbacks.append(tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        mode='min',
        verbose=1
    ))
    
    # Reduce LR on Plateau
    callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        mode='min',
        verbose=1
    ))
    
    return callbacks


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calcula todas las métricas de evaluación.
    
    Args:
        y_true: Valores reales.
        y_pred: Valores predichos.
        
    Returns:
        Diccionario con todas las métricas calculadas.
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    
    # Calcular MAPE (evitar división por cero)
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100)
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'mape': mape
    }


def save_metrics(metrics: Dict[str, float], metrics_path: Path):
    """
    Guarda las métricas en un archivo JSON.
    
    Args:
        metrics: Diccionario con las métricas.
        metrics_path: Ruta donde guardar el archivo.
    """
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)


def load_metrics(metrics_path: Path) -> Dict[str, float]:
    """
    Carga las métricas desde un archivo JSON.
    
    Args:
        metrics_path: Ruta del archivo JSON.
        
    Returns:
        Diccionario con las métricas.
    """
    with open(metrics_path, 'r') as f:
        return json.load(f)


def save_history(history: tf.keras.callbacks.History, history_path: Path):
    """
    Guarda el historial de entrenamiento en un archivo pickle.
    
    Args:
        history: Historial de entrenamiento.
        history_path: Ruta donde guardar el archivo.
    """
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)


def load_history(history_path: Path) -> Dict[str, Any]:
    """
    Carga el historial de entrenamiento desde un archivo pickle.
    
    Args:
        history_path: Ruta del archivo pickle.
        
    Returns:
        Diccionario con el historial.
    """
    with open(history_path, 'rb') as f:
        return pickle.load(f)


def plot_training_history(history: Dict[str, Any], save_dir: Path, model_name: str):
    """
    Genera gráficos del historial de entrenamiento.
    
    Args:
        history: Historial de entrenamiento.
        save_dir: Directorio donde guardar las gráficas.
        model_name: Nombre del modelo para nombrar los archivos.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("whitegrid")
    
    # Gráfico de Loss
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history['loss'], label='Training Loss')
    ax.plot(history['val_loss'], label='Validation Loss')
    ax.set_title(f'{model_name.upper()} - Loss', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    fig.savefig(save_dir / f'{model_name}_loss.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Gráfico de MAE
    fig, ax = plt.subplots(figsize=(10, 6))
    if 'mae' in history:
        ax.plot(history['mae'], label='Training MAE')
    if 'val_mae' in history:
        ax.plot(history['val_mae'], label='Validation MAE')
    ax.set_title(f'{model_name.upper()} - MAE', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAE')
    ax.legend()
    fig.savefig(save_dir / f'{model_name}_mae.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def get_optimizer(config: AIConfig):
    """
    Obtiene el optimizador según la configuración.
    
    Args:
        config: Objeto de configuración.
        
    Returns:
        Instancia del optimizador de Keras.
    """
    if config.OPTIMIZER == config.OPTIMIZER.ADAM:
        return tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    elif config.OPTIMIZER == config.OPTIMIZER.RMSPROP:
        return tf.keras.optimizers.RMSprop(learning_rate=config.LEARNING_RATE)
    elif config.OPTIMIZER == config.OPTIMIZER.SGD:
        return tf.keras.optimizers.SGD(learning_rate=config.LEARNING_RATE)
    else:
        raise ValueError(f"Optimizador no soportado: {config.OPTIMIZER}")


def create_sequences(X: np.ndarray, y: np.ndarray, window_size: int,
                      group_col_index: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transforma los datos en secuencias temporales para modelos LSTM/GRU/CNN-LSTM.

    IMPORTANTE: las ventanas se generan DENTRO de cada producto (agrupando
    por la columna de StockCode, por defecto la columna 0 de X), nunca
    cruzando de un producto a otro. La versión anterior cortaba filas
    consecutivas del array global (que puede mezclar productos distintos
    sin relación temporal entre sí), lo que hacía que LSTM/GRU/CNN-LSTM/
    CNN-GRU aprendieran de "secuencias" sin ninguna estructura temporal real.

    Args:
        X: Características de entrada (train_df/val_df/test_df ya convertido
           a numpy). La columna en `group_col_index` debe ser StockCode
           codificado (columna 0 en el pipeline actual, ya que el DataFrame
           conserva 'StockCode' como primera columna tras separar el target).
        y: Variable objetivo.
        window_size: Tamaño de la ventana de secuencia.
        group_col_index: Índice de la columna de X que identifica el producto.

    Returns:
        Tupla con (X_sequences, y_sequences).
    """
    X_seq = []
    y_seq = []

    groups = X[:, group_col_index]
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]  # conserva el orden cronológico dentro del grupo
        X_g = X[idx]
        y_g = y[idx]
        for i in range(len(X_g) - window_size):
            X_seq.append(X_g[i:i+window_size])
            y_seq.append(y_g[i+window_size])

    return np.array(X_seq), np.array(y_seq)


def get_activation(config: AIConfig):
    """
    Obtiene la función de activación según la configuración.
    
    Args:
        config: Objeto de configuración.
        
    Returns:
        String con el nombre de la función de activación.
    """
    if config.ACTIVATION_FUNCTION == config.ACTIVATION_FUNCTION.RELU:
        return 'relu'
    elif config.ACTIVATION_FUNCTION == config.ACTIVATION_FUNCTION.TANH:
        return 'tanh'
    elif config.ACTIVATION_FUNCTION == config.ACTIVATION_FUNCTION.LEAKY_RELU:
        return tf.keras.layers.LeakyReLU(alpha=0.1)
    else:
        raise ValueError(f"Función de activación no soportada: {config.ACTIVATION_FUNCTION}")