import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, Any
from ia.config.config import AIConfig
from ia.utils.logger import setup_logger

logger = setup_logger("visualizer")
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100


class DataVisualizer:
    """
    Clase para generar visualizaciones del dataset.
    """
    
    def __init__(self, config: AIConfig):
        """
        Inicializa el visualizador.
        
        Args:
            config: Objeto de configuración.
        """
        self.config = config
        self.figures_dir = config.FIGURES_DIR
        
    def save_figure(self, fig: plt.Figure, filename: str) -> None:
        """
        Guarda una figura en disco.
        
        Args:
            fig: Objeto Figure de matplotlib.
            filename: Nombre del archivo.
        """
        filepath = self.figures_dir / filename
        fig.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Figura guardada en {filepath}")
        
    def plot_histograms(self, df: pd.DataFrame) -> None:
        """
        Genera histogramas para Quantity y UnitPrice.
        
        Args:
            df: DataFrame con los datos.
        """
        # Histograma Quantity
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df["Quantity"], bins=50, kde=True, ax=ax, color="skyblue")
        ax.set_title("Distribución de Quantity", fontsize=14)
        ax.set_xlabel("Quantity", fontsize=12)
        ax.set_ylabel("Frecuencia", fontsize=12)
        self.save_figure(fig, "1_histogram_quantity.png")
        
        # Histograma UnitPrice
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df["UnitPrice"], bins=50, kde=True, ax=ax, color="lightgreen")
        ax.set_title("Distribución de UnitPrice", fontsize=14)
        ax.set_xlabel("UnitPrice", fontsize=12)
        ax.set_ylabel("Frecuencia", fontsize=12)
        self.save_figure(fig, "2_histogram_unitprice.png")
        
    def plot_boxplots(self, df: pd.DataFrame) -> None:
        """
        Genera boxplots para Quantity y UnitPrice.
        
        Args:
            df: DataFrame con los datos.
        """
        # Boxplot Quantity
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(y=df["Quantity"], ax=ax, color="skyblue")
        ax.set_title("Boxplot de Quantity", fontsize=14)
        ax.set_ylabel("Quantity", fontsize=12)
        self.save_figure(fig, "3_boxplot_quantity.png")
        
        # Boxplot UnitPrice
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(y=df["UnitPrice"], ax=ax, color="lightgreen")
        ax.set_title("Boxplot de UnitPrice", fontsize=14)
        ax.set_ylabel("UnitPrice", fontsize=12)
        self.save_figure(fig, "4_boxplot_unitprice.png")
        
    def plot_correlation_heatmap(self, df: pd.DataFrame) -> None:
        """
        Genera heatmap de correlación.
        
        Args:
            df: DataFrame con los datos.
        """
        numeric_df = df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax, cbar=True)
        ax.set_title("Heatmap de Correlación", fontsize=14)
        self.save_figure(fig, "5_correlation_heatmap.png")
        
        # Matriz de correlación como texto
        corr.to_csv(self.figures_dir / "6_correlation_matrix.csv")
        logger.info("Matriz de correlación guardada en 6_correlation_matrix.csv")
        
    def plot_sales_over_time(self, df: pd.DataFrame) -> None:
        """
        Genera gráficos de ventas por mes, día y serie temporal.
        
        Args:
            df: DataFrame con los datos.
        """
        # Ventas por mes
        if "Mes" in df.columns and "Ingresos" in df.columns:
            monthly_sales = df.groupby("Mes")["Ingresos"].sum().reset_index()
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(x="Mes", y="Ingresos", data=monthly_sales, ax=ax, hue="Mes", palette="viridis", legend=False)
            ax.set_title("Ventas por Mes", fontsize=14)
            ax.set_xlabel("Mes", fontsize=12)
            ax.set_ylabel("Ingresos", fontsize=12)
            self.save_figure(fig, "7_sales_by_month.png")
            
        # Ventas por día de la semana
        if "DíaSemana" in df.columns and "Ingresos" in df.columns:
            weekday_sales = df.groupby("DíaSemana")["Ingresos"].sum().reset_index()
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(x="DíaSemana", y="Ingresos", data=weekday_sales, ax=ax, hue="DíaSemana", palette="magma", legend=False)
            ax.set_title("Ventas por Día de la Semana", fontsize=14)
            ax.set_xlabel("Día de la Semana", fontsize=12)
            ax.set_ylabel("Ingresos", fontsize=12)
            self.save_figure(fig, "8_sales_by_day.png")
            
        # Serie temporal de ventas
        if "InvoiceDate" in df.columns and "Ingresos" in df.columns:
            df["DateOnly"] = df["InvoiceDate"].dt.date
            daily_sales = df.groupby("DateOnly")["Ingresos"].sum().reset_index()
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(daily_sales["DateOnly"], daily_sales["Ingresos"], color="b")
            ax.set_title("Serie Temporal de Ventas", fontsize=14)
            ax.set_xlabel("Fecha", fontsize=12)
            ax.set_ylabel("Ingresos", fontsize=12)
            plt.xticks(rotation=45)
            self.save_figure(fig, "13_time_series_sales.png")
            
    def plot_top_products(self, df: pd.DataFrame, top_n: int = 20) -> None:
        """
        Genera gráficos de productos más vendidos.
        
        Args:
            df: DataFrame con los datos.
            top_n: Número de productos top a mostrar.
        """
        if "Description" in df.columns and "Quantity" in df.columns:
            # Top productos por cantidad vendida
            top_products = df.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(top_n)
            fig, ax = plt.subplots(figsize=(12, 8))
            top_products.plot(kind="barh", ax=ax, color="skyblue")
            ax.set_title(f"Top {top_n} Productos Más Vendidos", fontsize=14)
            ax.set_xlabel("Cantidad Vendida", fontsize=12)
            ax.set_ylabel("Producto", fontsize=12)
            plt.gca().invert_yaxis()  # Invertir eje y para que el primero quede arriba
            self.save_figure(fig, "9_top_products.png")
            
            # Top 20 productos (especial)
            self.save_figure(fig, "10_top_20_products.png")
            
    def plot_country_distribution(self, df: pd.DataFrame) -> None:
        """
        Genera gráfico de distribución por país.
        
        Args:
            df: DataFrame con los datos.
        """
        if "Country" in df.columns:
            country_counts = df["Country"].value_counts().head(10).reset_index()
            country_counts.columns = ["Country", "Count"]
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(x="Count", y="Country", data=country_counts, ax=ax, hue="Country", palette="viridis", legend=False)
            ax.set_title("Distribución por País (Top 10)", fontsize=14)
            ax.set_xlabel("Número de Transacciones", fontsize=12)
            ax.set_ylabel("País", fontsize=12)
            self.save_figure(fig, "11_country_distribution.png")
            
    def plot_revenue_distribution(self, df: pd.DataFrame) -> None:
        """
        Genera gráfico de distribución de ingresos.
        
        Args:
            df: DataFrame con los datos.
        """
        if "Ingresos" in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(df["Ingresos"], bins=50, kde=True, ax=ax, color="orange")
            ax.set_title("Distribución de Ingresos", fontsize=14)
            ax.set_xlabel("Ingresos", fontsize=12)
            ax.set_ylabel("Frecuencia", fontsize=12)
            self.save_figure(fig, "12_revenue_distribution.png")
            
    def plot_outlier_comparison(self, df_before: pd.DataFrame, df_after: pd.DataFrame, column: str) -> None:
        """
        Genera gráfico de comparación de outliers antes y después de la eliminación.
        
        Args:
            df_before: DataFrame antes de eliminar outliers.
            df_after: DataFrame después de eliminar outliers.
            column: Columna a comparar.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        sns.boxplot(y=df_before[column], ax=ax1, color="lightcoral")
        ax1.set_title(f"Antes de Eliminar Outliers - {column}", fontsize=12)
        
        sns.boxplot(y=df_after[column], ax=ax2, color="lightgreen")
        ax2.set_title(f"Después de Eliminar Outliers - {column}", fontsize=12)
        
        self.save_figure(fig, f"outlier_comparison_{column}.png")
        
    def generate_all_visualizations(self, df: pd.DataFrame) -> None:
        """
        Genera todas las visualizaciones requeridas.
        
        Args:
            df: DataFrame con los datos.
        """
        logger.info("Generando visualizaciones...")
        self.plot_histograms(df)
        self.plot_boxplots(df)
        self.plot_correlation_heatmap(df)
        self.plot_sales_over_time(df)
        self.plot_top_products(df)
        self.plot_country_distribution(df)
        self.plot_revenue_distribution(df)
        logger.info("Todas las visualizaciones generadas.")
