import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from ia.config.config import AIConfig
from ia.utils.logger import setup_logger

logger = setup_logger("report_generator")


class ReportGenerator:
    """
    Clase para generar reportes de los resultados de los modelos de ML.
    Soporta formatos PDF, Excel y Word.
    """

    def __init__(self, config: AIConfig):
        """
        Inicializa el generador de reportes con la configuración.
        
        Args:
            config: Objeto de configuración.
        """
        self.config = config

    def generate_html_eda_report(self, df: pd.DataFrame, statistics: pd.DataFrame, outlier_stats: Dict[str, Any]) -> None:
        """
        Genera un reporte HTML de EDA.
        
        Args:
            df: DataFrame procesado.
            statistics: DataFrame con estadísticas descriptivas.
            outlier_stats: Diccionario con estadísticas de outliers.
        """
        logger.info("Generando reporte HTML de EDA...")
        
        figures_dir = self.config.FIGURES_DIR
        figure_files = sorted(figures_dir.glob("*.png"))
        
        # Generar HTML
        html_content = f"""
        <!DOCTYPE html>
        <html lang="es">
        <head>
            <meta charset="UTF-8">
            <title>Reporte de Análisis Exploratorio de Datos</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #3498db; margin-top: 40px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .figure {{ margin: 30px 0; }}
                .figure img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <h1>Reporte de Análisis Exploratorio de Datos (EDA)</h1>
            
            <h2>Resumen del Dataset</h2>
            <p><strong>Número de registros:</strong> {len(df)}</p>
            <p><strong>Número de columnas:</strong> {len(df.columns)}</p>
            <p><strong>Columnas:</strong> {', '.join(df.columns)}</p>
            
            <h2>Valores Nulos</h2>
            <table>
                <tr><th>Columna</th><th>Valores Nulos</th><th>Porcentaje</th></tr>
                {''.join([f'<tr><td>{col}</td><td>{df[col].isna().sum()}</td><td>{(df[col].isna().sum()/len(df)*100):.2f}%</td></tr>' for col in df.columns])}
            </table>
            
            <h2>Estadísticas Descriptivas</h2>
            {statistics.to_html(index=False, classes='table')}
            
            <h2>Outliers</h2>
            <table>
                <tr><th>Columna</th><th>Cantidad de Outliers</th><th>Porcentaje</th></tr>
                {''.join([f'<tr><td>{col}</td><td>{stats["count_outliers"]}</td><td>{stats["percentage_outliers"]:.2f}%</td></tr>' for col, stats in outlier_stats.items()])}
            </table>
            
            <h2>Visualizaciones</h2>
            {''.join([f'<div class="figure"><h3>{fig.stem}</h3><img src="figures/{fig.name}" alt="{fig.stem}"></div>' for fig in figure_files])}
            
            <h2>Conclusiones Automáticas</h2>
            <ul>
                <li>El dataset contiene {len(df)} registros válidos después de la limpieza.</li>
                <li>Se detectaron outliers en las columnas: {', '.join(outlier_stats.keys())}.</li>
                <li>Las variables numéricas principales son: {', '.join(df.select_dtypes(include=[np.number]).columns.tolist())}.</li>
            </ul>
            
        </body>
        </html>
        """
        
        # Guardar HTML
        report_path = self.config.EDA_REPORT_PATH
        report_path.write_text(html_content, encoding="utf-8")
        logger.info(f"Reporte HTML guardado en {report_path}")

    def generate_pdf(self, data: Dict[str, Any], output_path: Optional[Path] = None) -> None:
        """
        Genera un reporte en formato PDF.
        
        Args:
            data: Datos a incluir en el reporte (métricas, gráficos, etc.).
            output_path: Ruta donde guardar el PDF. Si es None, usa config.
        """
        raise NotImplementedError("Método generate_pdf() no implementado aún.")

    def generate_excel(self, data: Dict[str, Any], output_path: Optional[Path] = None) -> None:
        """
        Genera un reporte en formato Excel (.xlsx).
        
        Args:
            data: Datos a incluir en el reporte.
            output_path: Ruta donde guardar el Excel. Si es None, usa config.
        """
        raise NotImplementedError("Método generate_excel() no implementado aún.")

    def generate_word(self, data: Dict[str, Any], output_path: Optional[Path] = None) -> None:
        """
        Genera un reporte en formato Word (.docx).
        
        Args:
            data: Datos a incluir en el reporte.
            output_path: Ruta donde guardar el Word. Si es None, usa config.
        """
        raise NotImplementedError("Método generate_word() no implementado aún.")
