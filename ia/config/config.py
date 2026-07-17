import os
from pathlib import Path
from enum import Enum


class ScalerType(Enum):
    """Tipo de escalador a usar para normalización."""
    MINMAX = "minmax"
    STANDARD = "standard"


class OptimizerType(Enum):
    """Tipo de optimizador a usar para entrenamiento."""
    ADAM = "adam"
    RMSPROP = "rmsprop"
    SGD = "sgd"


class ActivationType(Enum):
    """Tipo de función de activación a usar en capas ocultas."""
    RELU = "relu"
    TANH = "tanh"
    LEAKY_RELU = "leaky_relu"
    ELU = "elu"


class TuningMethod(Enum):
    """Método de ajuste de hiperparámetros."""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"


class AIConfig:
    """
    Clase de configuración para el módulo de Inteligencia Artificial.
    Almacena todas las variables configurables del sistema de ML.
    """
    
    # Rutas del proyecto
    BASE_DIR = Path(__file__).parent.parent.parent.resolve()
    IA_DIR = BASE_DIR / "ia"
    DATASET_DIR = IA_DIR / "dataset"
    MODELS_DIR = IA_DIR / "models"
    REPORTS_DIR = IA_DIR / "reports"
    LOGS_DIR = IA_DIR / "logs"
    FIGURES_DIR = REPORTS_DIR / "figures"
    TENSORBOARD_DIR = REPORTS_DIR / "tensorboard"
    BACKEND_MODEL_DIR = BASE_DIR / "backend" / "model"
    STATISTICS_DIR = IA_DIR / "statistics"
    
    # Dataset
    DATASET_PATH = DATASET_DIR / "OnlineRetail.xlsx"
    
    # Parámetros generales de entrenamiento
    RANDOM_SEED = 42
    EPOCHS = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    DROPOUT_RATE = 0.3
    VALIDATION_SPLIT = 0.2
    OPTIMIZER = OptimizerType.ADAM
    ACTIVATION_FUNCTION = ActivationType.RELU
    
    # Parámetros específicos de modelos
    # MLP
    MLP_LAYERS = [128, 64, 32]
    # LSTM
    LSTM_WINDOW_SIZE = 10
    LSTM_EPOCHS = 100
    LSTM_BATCH_SIZE = 32
    LSTM_UNITS = [64, 32]
    LSTM_DROPOUT = 0.3
    LSTM_LEARNING_RATE = 0.001
    LSTM_SEQUENCE_LENGTH = 10
    # GRU
    GRU_WINDOW_SIZE = 10
    GRU_EPOCHS = 100
    GRU_BATCH_SIZE = 32
    GRU_UNITS = [64, 32]
    GRU_DROPOUT = 0.3
    GRU_LEARNING_RATE = 0.001
    GRU_SEQUENCE_LENGTH = 10
    # CNN-LSTM
    CNN_LSTM_WINDOW_SIZE = 10
    CNN_LSTM_FILTERS = 64
    CNN_LSTM_KERNEL_SIZE = 3
    CNN_LSTM_UNITS = 64
    CNN_LSTM_DROPOUT = 0.3
    CNN_LSTM_BATCH_SIZE = 32
    CNN_LSTM_EPOCHS = 100
    CNN_LSTM_LEARNING_RATE = 0.001
    CNN_LSTM_LSTM_UNITS = 64
    CNN_LSTM_SEQUENCE_LENGTH = 10
    # CNN-GRU
    CNN_GRU_WINDOW_SIZE = 10
    CNN_GRU_FILTERS = 64
    CNN_GRU_KERNEL_SIZE = 3
    CNN_GRU_UNITS = 64
    CNN_GRU_DROPOUT = 0.3
    CNN_GRU_BATCH_SIZE = 32
    CNN_GRU_EPOCHS = 100
    CNN_GRU_LEARNING_RATE = 0.001
    
    # Parámetros de preprocesamiento
    TEST_SIZE = 0.15
    VALIDATION_SIZE = 0.15
    SCALER_TYPE = ScalerType.MINMAX
    
    # Columnas requeridas
    REQUIRED_COLUMNS = [
        "InvoiceNo",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "UnitPrice",
        "CustomerID",
        "Country"
    ]
    
    # Variable objetivo
    TARGET_VARIABLE = "Quantity"
    
    # Rutas de archivos procesados
    PROCESSED_DATA_PATH = DATASET_DIR / "processed" / "processed_data.csv"
    TRAIN_DATA_PATH = DATASET_DIR / "processed" / "train.csv"
    TEST_DATA_PATH = DATASET_DIR / "processed" / "test.csv"
    VALIDATION_DATA_PATH = DATASET_DIR / "processed" / "validation.csv"
    STATISTICS_PATH = REPORTS_DIR / "statistics.csv"
    EDA_REPORT_PATH = REPORTS_DIR / "eda_report.html"

    # Rutas de los reportes finales (PDF, Word, Excel)
    REPORT_PDF_PATH = REPORTS_DIR / "reporte_final.pdf"
    REPORT_WORD_PATH = REPORTS_DIR / "reporte_final.docx"
    REPORT_EXCEL_PATH = REPORTS_DIR / "reporte_final.xlsx"
    LOG_FILE_PATH = LOGS_DIR / "preprocessing.log"
    
    # Rutas para modelos entrenados
    MLP_MODEL_PATH = MODELS_DIR / "mlp.keras"
    MLP_HISTORY_PATH = MODELS_DIR / "mlp_history.pkl"
    MLP_METRICS_PATH = MODELS_DIR / "mlp_metrics.json"
    MLP_LOG_PATH = LOGS_DIR / "mlp.log"
    
    LSTM_MODEL_PATH = MODELS_DIR / "lstm.keras"
    LSTM_HISTORY_PATH = MODELS_DIR / "lstm_history.pkl"
    LSTM_METRICS_PATH = MODELS_DIR / "lstm_metrics.json"
    LSTM_LOG_PATH = LOGS_DIR / "lstm.log"
    
    GRU_MODEL_PATH = MODELS_DIR / "gru.keras"
    GRU_HISTORY_PATH = MODELS_DIR / "gru_history.pkl"
    GRU_METRICS_PATH = MODELS_DIR / "gru_metrics.json"
    GRU_LOG_PATH = LOGS_DIR / "gru.log"
    
    CNN_LSTM_MODEL_PATH = MODELS_DIR / "cnn_lstm.keras"
    CNN_LSTM_HISTORY_PATH = MODELS_DIR / "cnn_lstm_history.pkl"
    CNN_LSTM_METRICS_PATH = MODELS_DIR / "cnn_lstm_metrics.json"
    CNN_LSTM_LOG_PATH = LOGS_DIR / "cnn_lstm.log"
    
    CNN_GRU_MODEL_PATH = MODELS_DIR / "cnn_gru.keras"
    CNN_GRU_HISTORY_PATH = MODELS_DIR / "cnn_gru_history.pkl"
    CNN_GRU_METRICS_PATH = MODELS_DIR / "cnn_gru_metrics.json"
    CNN_GRU_LOG_PATH = LOGS_DIR / "cnn_gru.log"
    
    # Rutas para reportes de comparación
    MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
    BEST_MODEL_PATH = REPORTS_DIR / "best_model.json"
    
    # Parámetros de Cross Validation
    CV_FOLDS = 5
    CV_SHUFFLE = True
    CV_LOG_PATH = LOGS_DIR / "cross_validation.log"
    CV_RESULTS_JSON_PATH = REPORTS_DIR / "cross_validation_results.json"
    CV_RESULTS_CSV_PATH = REPORTS_DIR / "cross_validation_results.csv"
    CV_SUMMARY_CSV_PATH = REPORTS_DIR / "cross_validation_summary.csv"
    
    # Parámetros de Hyperparameter Tuning
    ENABLE_HYPERPARAMETER_TUNING = True
    TUNING_RANDOM_SEED = RANDOM_SEED
    MAX_TRIALS = 5
    EXECUTIONS_PER_TRIAL = 1
    OBJECTIVE = "val_loss"
    OVERWRITE_TUNING = True
    TUNING_LOG_PATH = LOGS_DIR / "hyperparameter_tuning.log"
    TUNING_RESULTS_JSON_PATH = REPORTS_DIR / "hyperparameter_results.json"
    TUNING_RESULTS_CSV_PATH = REPORTS_DIR / "hyperparameter_results.csv"
    TUNING_BEST_CONFIG_PATH = REPORTS_DIR / "best_hyperparameters.json"
    TUNING_DIR = IA_DIR / "tuning"
    
    # Espacio de búsqueda para Hyperparameter Tuning
    TUNING_LR = [0.001, 0.0005, 0.0001]
    TUNING_BATCH_SIZE = [16, 32, 64]
    TUNING_EPOCHS = [50, 100, 150]
    TUNING_DROPOUT = [0.1, 0.2, 0.3]
    TUNING_OPTIMIZERS = [OptimizerType.ADAM, OptimizerType.RMSPROP, OptimizerType.SGD]
    TUNING_ACTIVATIONS = [ActivationType.RELU, ActivationType.LEAKY_RELU, ActivationType.ELU, ActivationType.TANH]
    # MLP
    TUNING_MLP_LAYERS = [[128, 64], [256, 128], [256, 128, 64]]
    # LSTM
    TUNING_LSTM_UNITS = [[32, 16], [64, 32], [128, 64]]
    # GRU
    TUNING_GRU_UNITS = [[32, 16], [64, 32], [128, 64]]
    # CNN-LSTM
    TUNING_CNN_LSTM_FILTERS = [[16, 32], [32, 64], [64, 128]]
    TUNING_CNN_LSTM_LSTM_UNITS = [32, 64, 128]
    # CNN-GRU
    TUNING_CNN_GRU_FILTERS = [[16, 32], [32, 64], [64, 128]]
    TUNING_CNN_GRU_GRU_UNITS = [32, 64, 128]
    
    # Rutas para reportes de la fase 4
    FINAL_COMPARISON_PATH = REPORTS_DIR / "final_model_comparison.csv"
    FINAL_BEST_MODEL_PATH = REPORTS_DIR / "final_best_model.json"
    BACKEND_BEST_MODEL_PATH = BACKEND_MODEL_DIR / "best_model.keras"
    BACKEND_MODEL_INFO_PATH = BACKEND_MODEL_DIR / "model_info.json"
    
    # Parámetros de Validación Estadística
    STATISTICS_ALPHA = 0.05
    STATISTICS_METRIC = "rmse"
    STATISTICS_LOG_PATH = LOGS_DIR / "statistics.log"
    STATISTICS_RESULTS_PATH = REPORTS_DIR / "statistics_results.csv"
    FRIEDMAN_RESULTS_PATH = REPORTS_DIR / "friedman_results.csv"
    WILCOXON_RESULTS_PATH = REPORTS_DIR / "wilcoxon_results.csv"
    NEMENYI_RESULTS_PATH = REPORTS_DIR / "nemenyi_results.csv"
    RANKING_RESULTS_PATH = REPORTS_DIR / "ranking_results.csv"
    PVALUES_RESULTS_PATH = REPORTS_DIR / "pvalues_results.csv"
    CONFIDENCE_INTERVALS_PATH = REPORTS_DIR / "confidence_intervals.csv"
    EFFECT_SIZE_PATH = REPORTS_DIR / "effect_size.csv"
    STATISTICS_REPORT_HTML = REPORTS_DIR / "statistics_report.html"
    STATISTICS_REPORT_MD = REPORTS_DIR / "statistics_report.md"
    STATISTICS_CONCLUSIONS = REPORTS_DIR / "statistical_conclusions.md"
    
    # Banderas de ejecución del pipeline maestro
    RUN_PREPROCESSING = True
    RUN_TRAINING = True
    RUN_MODEL_COMPARISON = True
    RUN_CROSS_VALIDATION = True
    RUN_HYPERPARAMETER_TUNING = True
    RUN_STATISTICAL_TESTS = True
    RUN_DEPLOYMENT = True
    
    # Rutas para reportes del pipeline maestro
    PROJECT_SUMMARY_JSON = REPORTS_DIR / "project_summary.json"
    PROJECT_SUMMARY_MD = REPORTS_DIR / "project_summary.md"
    PROJECT_SUMMARY_HTML = REPORTS_DIR / "project_summary.html"
    PROJECT_EXECUTION_LOG = LOGS_DIR / "project_execution.log"
    
    @classmethod
    def ensure_directories_exist(cls):
        """
        Asegura que todas las rutas de directorios existan.
        Crea los directorios si no existen.
        """
        directories = [
            cls.DATASET_DIR,
            cls.DATASET_DIR / "processed",
            cls.MODELS_DIR,
            cls.REPORTS_DIR,
            cls.LOGS_DIR,
            cls.FIGURES_DIR,
            cls.TENSORBOARD_DIR,
            cls.BACKEND_MODEL_DIR,
            cls.STATISTICS_DIR,
            cls.TUNING_DIR
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
