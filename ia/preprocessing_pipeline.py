"""
Pipeline de preprocesamiento y EDA para el dataset de demanda.
"""
import sys
from pathlib import Path

# Añadir el directorio raíz del proyecto al PATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ia.config.config import AIConfig
from ia.preprocessing import DataPreprocessor, OutlierDetector
from ia.evaluation.statistics import StatisticsAnalyzer
from ia.reports.visualizer import DataVisualizer
from ia.reports.report_generator import ReportGenerator
from ia.utils.logger import setup_logger

logger = setup_logger("preprocessing_pipeline")


def run_preprocessing():
    """
    Función principal que ejecuta todo el pipeline de preprocesamiento y EDA.
    """
    try:
        logger.info("=" * 60)
        logger.info("INICIANDO PIPELINE DE PREPROCESAMIENTO Y EDA")
        logger.info("=" * 60)
        
        # Inicializar configuración
        config = AIConfig()
        config.ensure_directories_exist()
        
        # Paso 1: Cargar y limpiar datos
        logger.info("\n--- Paso 1: Cargar y limpiar datos ---")
        preprocessor = DataPreprocessor(config)
        raw_data = preprocessor.load_data()
        cleaned_data = preprocessor.clean_data(raw_data)
        
        # Paso 2: Feature Engineering
        logger.info("\n--- Paso 2: Feature Engineering ---")
        data_with_features = preprocessor.feature_engineering(cleaned_data)
        
        # Paso 3: Detectar y eliminar outliers
        logger.info("\n--- Paso 3: Detección y eliminación de outliers ---")
        outlier_detector = OutlierDetector()
        outlier_stats = outlier_detector.detect_iqr(data_with_features, ["Quantity", "UnitPrice", "Ingresos"])
        data_no_outliers, _ = outlier_detector.remove_outliers(data_with_features, ["Quantity", "UnitPrice", "Ingresos"])
        
        # Paso 4: Generar visualizaciones
        logger.info("\n--- Paso 4: Generar visualizaciones ---")
        visualizer = DataVisualizer(config)
        visualizer.generate_all_visualizations(data_no_outliers)
        
        # Generar gráfico de comparación de outliers
        visualizer.plot_outlier_comparison(data_with_features, data_no_outliers, "Quantity")
        
        # Paso 5: Calcular estadísticas
        logger.info("\n--- Paso 5: Calcular estadísticas descriptivas ---")
        stats_analyzer = StatisticsAnalyzer(config)
        statistics = stats_analyzer.analyze(data_no_outliers)
        stats_analyzer.save_statistics(statistics)
        
        # Paso 6: Normalizar y codificar
        logger.info("\n--- Paso 6: Normalizar y codificar variables ---")
        # Eliminar DateOnly antes de normalizar y codificar
        data_for_ml = data_no_outliers.drop("DateOnly", axis=1, errors="ignore")
        normalized_data, scaler = preprocessor.normalize(data_for_ml)
        encoded_data = preprocessor.encode_categorical(normalized_data)
        
        # Paso 7: Dividir dataset
        logger.info("\n--- Paso 7: Dividir dataset en splits ---")
        train, val, test = preprocessor.split_dataset(encoded_data)
        
        # Paso 8: Guardar resultados
        logger.info("\n--- Paso 8: Guardar resultados ---")
        preprocessor.save_processed_dataset(encoded_data)
        preprocessor.save_splits(train, val, test)
        
        # Paso 9: Generar reporte HTML
        logger.info("\n--- Paso 9: Generar reporte EDA ---")
        report_generator = ReportGenerator(config)
        report_generator.generate_html_eda_report(data_no_outliers, statistics, outlier_stats)
        
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE DE PREPROCESAMIENTO COMPLETADO EXITOSAMENTE!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"ERROR EN EL PIPELINE DE PREPROCESAMIENTO: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    run_preprocessing()
