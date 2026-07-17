import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from ia.config.config import AIConfig
from ia.utils.logger import setup_logger
from ia.preprocessing.outlier_detector import OutlierDetector
from jinja2 import Template
import time

logger = setup_logger("preprocessing", Path("ia/logs/preprocessing.log"))


class ImprovedDataPreprocessor:
    """
    Clase mejorada para preprocesamiento de datos para predicción de demanda.
    Cumpliendo con requisitos:
    - No Data Leakage (elimina variables que dependen de Quantity)
    - Codificación adecuada de variables categóricas
    - Guarda scaler y encoders
    - Genera reporte HTML
    - Verifica dataset final (no object, no NaN, no inf)
    """

    def __init__(self, config: AIConfig):
        """
        Inicializa el preprocesador con la configuración.
        
        Args:
            config: Objeto de configuración con rutas y parámetros.
        """
        self.config = config
        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        self.scaler: StandardScaler = StandardScaler()
        self.encoders: Dict[str, LabelEncoder] = {}
        self.outlier_detector: OutlierDetector = OutlierDetector()
        self.report_data: Dict[str, Any] = {
            "columns_removed": [],
            "columns_transformed": [],
            "encodings_applied": [],
            "scalers_used": [],
            "missing_values": {},
            "duplicates": 0,
            "outliers": {},
            "final_variables": []
        }
        self.start_time = time.time()

    def load_data(self, file_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Carga el dataset desde el archivo especificado (Excel).
        
        Args:
            file_path: Ruta del archivo Excel. Si es None, usa la ruta de config.
            
        Returns:
            DataFrame con los datos cargados.
            
        Raises:
            FileNotFoundError: Si el archivo no existe.
            ValueError: Si faltan columnas requeridas.
        """
        if file_path is None:
            file_path = self.config.DATASET_PATH
            
        if not file_path.exists():
            error_msg = f"Archivo no encontrado en {file_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        logger.info(f"Cargando dataset desde {file_path}...")
        self.raw_data = pd.read_excel(file_path)
        logger.info(f"Dataset cargado con {len(self.raw_data)} registros y {len(self.raw_data.columns)} columnas")
        
        # Validar columnas requeridas
        missing_cols = [col for col in self.config.REQUIRED_COLUMNS if col not in self.raw_data.columns]
        if missing_cols:
            error_msg = f"Faltan columnas requeridas: {', '.join(missing_cols)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        logger.info("Todas las columnas requeridas están presentes")
        return self.raw_data

    def remove_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Elimina registros duplicados del dataset y registra.
        
        Args:
            data: DataFrame a procesar.
            
        Returns:
            DataFrame sin duplicados.
        """
        initial_count = len(data)
        df_clean = data.drop_duplicates()
        removed_count = initial_count - len(df_clean)
        self.report_data["duplicates"] = removed_count
        logger.info(f"Duplicados eliminados: {removed_count} registros (inicial: {initial_count}, final: {len(df_clean)})")
        return df_clean

    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Elimina registros con valores nulos en las columnas críticas y registra.
        
        Args:
            data: DataFrame con valores nulos.
            
        Returns:
            DataFrame sin registros con valores nulos en columnas críticas.
        """
        initial_count = len(data)
        
        # Eliminar registros con valores nulos en columnas específicas
        critical_cols = ["Description", "CustomerID", "UnitPrice", "InvoiceDate", "Quantity"]
        for col in data.columns:
            self.report_data["missing_values"][col] = int(data[col].isna().sum())
            
        df_clean = data.dropna(subset=critical_cols)
        
        removed_count = initial_count - len(df_clean)
        logger.info(f"Registros con valores nulos eliminados: {removed_count} (inicial: {initial_count}, final: {len(df_clean)})")
        return df_clean
        
    def remove_invalid_rows(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Elimina registros inválidos.
        
        Args:
            data: DataFrame a limpiar.
            
        Returns:
            DataFrame sin registros inválidos.
        """
        initial_count = len(data)
        
        # Eliminar Quantity <= 0 y UnitPrice <=0
        df_clean = data[(data["Quantity"] > 0) & (data["UnitPrice"] > 0)]
        
        # Eliminar facturas canceladas (que empiezan con C)
        df_clean = df_clean[~df_clean["InvoiceNo"].astype(str).str.startswith("C")]
        
        removed_count = initial_count - len(df_clean)
        logger.info(f"Registros inválidos eliminados: {removed_count} (inicial: {initial_count}, final: {len(df_clean)})")
        return df_clean
        
    def convert_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Convierte los tipos de datos de las columnas.
        
        Args:
            data: DataFrame con tipos de datos incorrectos.
            
        Returns:
            DataFrame con tipos de datos correctos.
        """
        logger.info("Convirtiendo tipos de datos...")
        
        df_clean = data.copy()
        
        # Convertir InvoiceDate a datetime
        df_clean["InvoiceDate"] = pd.to_datetime(df_clean["InvoiceDate"])
        
        # Convertir CustomerID a entero
        df_clean["CustomerID"] = df_clean["CustomerID"].astype(int)
        
        # Asegurar que Quantity y UnitPrice sean numéricos
        df_clean["Quantity"] = df_clean["Quantity"].astype(int)
        df_clean["UnitPrice"] = df_clean["UnitPrice"].astype(float)
        
        logger.info("Tipos de datos convertidos correctamente")
        return df_clean

    def aggregate_to_product_day(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Agrega las líneas de pedido a nivel de demanda diaria por producto.
        Cambia la unidad de análisis de "línea de pedido individual" a
        "unidades vendidas del producto X en el día D", que sí tiene
        autocorrelación temporal aprovechable por los modelos (a diferencia
        de la línea de pedido individual, que es prácticamente ruido).

        Args:
            data: DataFrame limpio a nivel de línea de pedido.

        Returns:
            DataFrame agregado a nivel producto-día.
        """
        logger.info("Agregando datos a nivel producto-día...")
        df = data.copy()
        df["InvoiceDate"] = df["InvoiceDate"].dt.normalize()

        agg = df.groupby(["StockCode", "InvoiceDate"]).agg(
            Quantity=("Quantity", "sum"),
            UnitPrice=("UnitPrice", "mean"),
            Country=("Country", lambda x: x.mode().iloc[0]),
            NumTransacciones=("Quantity", "count"),
        ).reset_index()

        agg = agg.sort_values(["StockCode", "InvoiceDate"]).reset_index(drop=True)
        logger.info(f"Dataset agregado: {len(agg)} filas producto-día (antes: {len(df)} líneas de pedido)")
        return agg

    def add_demand_history_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Añade variables de demanda histórica (lags y estadísticos móviles)
        por producto. Son las variables más informativas para pronóstico
        de series de tiempo y no existían en el pipeline original.

        Args:
            data: DataFrame agregado a nivel producto-día.

        Returns:
            DataFrame con columnas lag_1, lag_7, lag_14, lag_30,
            rolling_mean_7, rolling_std_7, rolling_mean_30.
        """
        logger.info("Generando variables de demanda histórica (lags, rolling)...")
        df = data.sort_values(["StockCode", "InvoiceDate"]).reset_index(drop=True).copy()

        for lag in [1, 7, 14, 30]:
            df[f"lag_{lag}"] = df.groupby("StockCode")["Quantity"].shift(lag)

        df["rolling_mean_7"] = (
            df.groupby("StockCode")["Quantity"]
              .apply(lambda s: s.shift(1).rolling(7).mean())
              .reset_index(level=0, drop=True)
        )
        df["rolling_std_7"] = (
            df.groupby("StockCode")["Quantity"]
              .apply(lambda s: s.shift(1).rolling(7).std())
              .reset_index(level=0, drop=True)
        )
        df["rolling_mean_30"] = (
            df.groupby("StockCode")["Quantity"]
              .apply(lambda s: s.shift(1).rolling(30).mean())
              .reset_index(level=0, drop=True)
        )

        before = len(df)
        lag_cols = [c for c in df.columns if c.startswith(("lag_", "rolling_"))]
        df = df.dropna(subset=lag_cols).reset_index(drop=True)
        logger.info(f"Filas sin historial suficiente eliminadas: {before - len(df)} (quedan {len(df)})")
        return df

    def feature_engineering(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Crea nuevas variables para el modelo y elimina las no necesarias (incluyendo data leakage).
        
        Args:
            data: DataFrame con las características originales.
            
        Returns:
            DataFrame con nuevas características añadidas.
        """
        logger.info("Realizando feature engineering...")
        df = data.copy()
        
        # Extraer componentes de fecha
        # Nota: no se extrae "Año" porque, tras agregar a producto-día y exigir
        # 30 días de historial por producto, todo el rango útil cae dentro de
        # un único año → quedaría como columna constante (sin valor predictivo).
        df["Mes"] = df["InvoiceDate"].dt.month
        df["Día"] = df["InvoiceDate"].dt.day
        df["DíaSemana"] = df["InvoiceDate"].dt.dayofweek  # 0=Lunes, 6=Domingo
        df["SemanaAño"] = df["InvoiceDate"].dt.isocalendar().week
        df["Trimestre"] = df["InvoiceDate"].dt.quarter
        
        # Variable EsFinDeSemana
        df["EsFinDeSemana"] = (df["DíaSemana"] >= 5).astype(int)
        
        # FechaOrden: representación numérica de la fecha, usada únicamente
        # para ordenar el split cronológico en split_dataset(). No se usa
        # como feature del modelo (se elimina antes de guardar los splits).
        df["FechaOrden"] = df["InvoiceDate"].astype("int64") // 10**9
        
        # MesNombre (lo codificaremos luego)
        mes_nombres = {1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
                      5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
                      9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"}
        df["MesNombre"] = df["Mes"].map(mes_nombres)
        
        # Eliminar columnas que causan DATA LEAKAGE o no son útiles
        columns_to_remove = []
        
        # 1. Ingresos: Calculado como Quantity * UnitPrice → DEPENDE DE TARGET → ELIMINAR
        if "Ingresos" in df.columns:
            columns_to_remove.append("Ingresos")
            self.report_data["columns_removed"].append(("Ingresos", "Data Leakage: Depende de Quantity (variable objetivo)"))
            
        # 2. InvoiceDate: Ya extrajimos todas las características temporales → ELIMINAR
        columns_to_remove.append("InvoiceDate")
        self.report_data["columns_removed"].append(("InvoiceDate", "Ya extrajimos características temporales (Año, Mes, Día, DíaSemana, SemanaAño, Trimestre, EsFinDeSemana, FechaOrden)"))
        
        # 3. InvoiceNo: Identificador único de transacción → No útil para predicción → ELIMINAR
        columns_to_remove.append("InvoiceNo")
        self.report_data["columns_removed"].append(("InvoiceNo", "Identificador único de transacción, no predictivo"))
        
        # 4. Description: Texto libre con alta cardinalidad, difícil de codificar efectivamente → ELIMINAR
        columns_to_remove.append("Description")
        self.report_data["columns_removed"].append(("Description", "Texto libre con alta cardinalidad, no útil para este pipeline"))
        
        df = df.drop(columns=columns_to_remove, errors="ignore")
        
        logger.info(f"Columnas eliminadas: {[col[0] for col in self.report_data['columns_removed']]}")
        logger.info(f"Nuevas variables creadas: {', '.join([col for col in df.columns if col not in data.columns])}")
        return df
        
    def encode_categorical(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
        """
        Codifica variables categóricas usando Label Encoding para alta cardinalidad.
        Justificación: StockCode, CustomerID tienen muchos valores únicos → OHE crearía demasiadas columnas.
        
        Args:
            data: DataFrame con variables categóricas.
            
        Returns:
            Tupla (DataFrame con variables codificadas, diccionario de encoders)
        """
        df = data.copy()
        encoders = {}
        
        # Definir columnas a codificar con Label Encoding
        categorical_cols = ["StockCode", "Country", "CustomerID", "MesNombre"]
        
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
                self.report_data["encodings_applied"].append((col, "Label Encoding"))
                logger.info(f"Aplicando Label Encoding a columna: {col} (n_values: {len(le.classes_)})")
        
        self.encoders = encoders
        return df, encoders
        
    def detect_outliers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detecta outliers y genera reporte, NO LOS ELIMINA.
        
        Args:
            data: DataFrame a analizar.
            
        Returns:
            Diccionario con estadísticas de outliers.
        """
        # Analizar solo variables numéricas continuas
        numeric_cols = ["UnitPrice"]
        outlier_stats = self.outlier_detector.detect_iqr(data, numeric_cols)
        self.report_data["outliers"] = outlier_stats
        return outlier_stats
        
    def scale_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
        """
        Escala únicamente variables numéricas continuas usando StandardScaler.
        No escala: target, variables temporales, variables codificadas.
        
        Args:
            data: DataFrame con características a escalar.
            
        Returns:
            Tupla con (DataFrame escalado, objeto scaler ajustado).
        """
        df = data.copy()
        
        # Variables a escalar: continuas (precio, historial de demanda)
        cols_to_scale = [
            "UnitPrice", "NumTransacciones",
            "lag_1", "lag_7", "lag_14", "lag_30",
            "rolling_mean_7", "rolling_std_7", "rolling_mean_30",
        ]
        cols_to_scale = [c for c in cols_to_scale if c in df.columns]
        
        self.scaler = StandardScaler()
        df[cols_to_scale] = self.scaler.fit_transform(df[cols_to_scale])
        self.report_data["scalers_used"].append(("StandardScaler", cols_to_scale))
        logger.info(f"Estandarización completada para: {cols_to_scale}")
        
        return df, self.scaler
        
    def verify_dataset(self, data: pd.DataFrame) -> bool:
        """
        Verifica que el dataset cumpla con:
        - No hay columnas tipo object
        - No hay NaN
        - No hay infinitos
        - No hay columnas constantes
        - No hay columnas duplicadas
        
        Args:
            data: DataFrame a verificar.
            
        Returns:
            True si todas las verificaciones pasan.
        """
        logger.info("Verificando dataset final...")
        
        checks = {
            "No object columns": len(data.select_dtypes(include=["object"]).columns) == 0,
            "No NaN values": data.isna().sum().sum() == 0,
            "No infinite values": np.isinf(data).sum().sum() == 0,
            "No constant columns": len(data.columns[data.nunique() <= 1]) == 0,
            "No duplicate columns": len(data.columns) == len(set(data.columns))
        }
        
        for check_name, result in checks.items():
            logger.info(f"  {check_name}: {'✅ PASSED' if result else '❌ FAILED'}")
            
        if not all(checks.values()):
            raise ValueError("Algunas verificaciones fallaron!")
            
        self.report_data["final_variables"] = list(data.columns)
        logger.info("Todas las verificaciones pasaron!")
        return True
        
    def split_dataset(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Divide el dataset en conjuntos de entrenamiento, validación y prueba (70-15-15).
        
        Args:
            data: DataFrame completo preprocesado.
            
        Returns:
            Tupla con (train_data, val_data, test_data).
        """
        logger.info("Dividiendo dataset en train, validation y test (corte cronológico)...")
        
        # Split cronológico por FechaOrden, no aleatorio: en un problema de
        # series de tiempo, un split aleatorio filtra información del futuro
        # hacia el pasado (data leakage temporal) y sobreestima el desempeño.
        data_sorted = data.sort_values("FechaOrden").reset_index(drop=True)
        n = len(data_sorted)
        train_end = int(n * (1 - self.config.TEST_SIZE - self.config.VALIDATION_SIZE))
        val_end = int(n * (1 - self.config.TEST_SIZE))
        
        train = data_sorted.iloc[:train_end].drop(columns=["FechaOrden"])
        val = data_sorted.iloc[train_end:val_end].drop(columns=["FechaOrden"])
        test = data_sorted.iloc[val_end:].drop(columns=["FechaOrden"])
        
        logger.info(f"Tamaños: Train={len(train)}, Validation={len(val)}, Test={len(test)}")
        return train, val, test
        
    def save_artifacts(self, scaler: StandardScaler, encoders: Dict[str, LabelEncoder]) -> None:
        """
        Guarda scaler y encoders en ia/models/.
        
        Args:
            scaler: Objeto StandardScaler ajustado.
            encoders: Diccionario con LabelEncoders.
        """
        # Guardar scaler
        scaler_path = self.config.MODELS_DIR / "scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        logger.info(f"Scaler guardado en {scaler_path}")
        
        # Guardar encoders
        for name, encoder in encoders.items():
            encoder_path = self.config.MODELS_DIR / f"{name.lower()}_encoder.pkl"
            with open(encoder_path, "wb") as f:
                pickle.dump(encoder, f)
            logger.info(f"Encoder {name} guardado en {encoder_path}")
            
    def generate_html_report(self) -> None:
        """
        Genera un reporte HTML en ia/reports/preprocessing_report.html.
        """
        # Template HTML simple
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Reporte de Preprocesamiento de Datos</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
                h1, h2, h3 { color: #2c3e50; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #4CAF50; color: white; }
                tr:nth-child(even) { background-color: #f2f2f2; }
                .success { color: green; }
                .warning { color: orange; }
                .error { color: red; }
            </style>
        </head>
        <body>
            <h1>Reporte de Preprocesamiento de Datos</h1>
            <p><strong>Fecha de generación:</strong> {{ timestamp }}</p>
            
            <h2>1. Columnas Eliminadas</h2>
            {% if columns_removed %}
            <table>
                <tr><th>Columna</th><th>Razón</th></tr>
                {% for col, reason in columns_removed %}
                <tr><td>{{ col }}</td><td>{{ reason }}</td></tr>
                {% endfor %}
            </table>
            {% else %}
            <p class="success">No se eliminaron columnas</p>
            {% endif %}
            
            <h2>2. Codificaciones Aplicadas</h2>
            {% if encodings_applied %}
            <table>
                <tr><th>Columna</th><th>Técnica</th></tr>
                {% for col, technique in encodings_applied %}
                <tr><td>{{ col }}</td><td>{{ technique }}</td></tr>
                {% endfor %}
            </table>
            {% else %}
            <p class="success">No se aplicaron codificaciones</p>
            {% endif %}
            
            <h2>3. Escaladores Utilizados</h2>
            {% if scalers_used %}
            <table>
                <tr><th>Scaler</th><th>Columnas</th></tr>
                {% for scaler, cols in scalers_used %}
                <tr><td>{{ scaler }}</td><td>{{ cols|join(', ') }}</td></tr>
                {% endfor %}
            </table>
            {% else %}
            <p class="success">No se usaron escaladores</p>
            {% endif %}
            
            <h2>4. Valores Faltantes</h2>
            <table>
                <tr><th>Columna</th><th>Cantidad</th></tr>
                {% for col, count in missing_values.items() %}
                <tr><td>{{ col }}</td><td>{{ count }}</td></tr>
                {% endfor %}
            </table>
            
            <h2>5. Duplicados</h2>
            <p>Duplicados eliminados: <strong>{{ duplicates }}</strong></p>
            
            <h2>6. Outliers</h2>
            {% if outliers %}
            <table>
                <tr><th>Columna</th><th>Cantidad</th><th>Porcentaje</th></tr>
                {% for col, stats in outliers.items() %}
                <tr><td>{{ col }}</td><td>{{ stats.count_outliers }}</td><td>{{ stats.percentage_outliers|round(2) }}%</td></tr>
                {% endfor %}
            </table>
            {% else %}
            <p class="success">No se detectaron outliers</p>
            {% endif %}
            
            <h2>7. Variables Finales</h2>
            <ul>
                {% for var in final_variables %}
                <li>{{ var }}</li>
                {% endfor %}
            </ul>
        </body>
        </html>
        """
        
        template = Template(html_template)
        report_html = template.render(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            **self.report_data
        )
        
        report_path = self.config.REPORTS_DIR / "preprocessing_report.html"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_html)
            
        logger.info(f"Reporte HTML generado en {report_path}")
        
    def save_splits(self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
        """
        Guarda los splits del dataset en archivos CSV.
        
        Args:
            train: Conjunto de entrenamiento.
            val: Conjunto de validación.
            test: Conjunto de prueba.
        """
        train.to_csv(self.config.TRAIN_DATA_PATH, index=False, encoding="utf-8")
        val.to_csv(self.config.VALIDATION_DATA_PATH, index=False, encoding="utf-8")
        test.to_csv(self.config.TEST_DATA_PATH, index=False, encoding="utf-8")
        
        logger.info("Splits guardados correctamente")
        
    def full_preprocessing_pipeline(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Ejecuta todo el pipeline de preprocesamiento de principio a fin.
        
        Returns:
            Tupla (train_data, val_data, test_data)
        """
        logger.info("=" * 80)
        logger.info("INICIANDO PIPELINE DE PREPROCESAMIENTO")
        logger.info("=" * 80)
        
        # Paso 1: Cargar datos
        df = self.load_data()
        
        # Paso 2: Limpieza básica
        df = self.remove_duplicates(df)
        df = self.handle_missing_values(df)
        df = self.remove_invalid_rows(df)
        df = self.convert_data_types(df)
        
        # Paso 2.5: Agregar a nivel producto-día y generar variables de demanda histórica
        df = self.aggregate_to_product_day(df)
        df = self.add_demand_history_features(df)
        
        # Paso 3: Feature Engineering
        df = self.feature_engineering(df)
        
        # Paso 4: Codificar variables categóricas
        df, encoders = self.encode_categorical(df)
        
        # Paso 5: Detectar outliers (no eliminar)
        self.detect_outliers(df)
        
        # Paso 6: Escalar variables
        df, scaler = self.scale_features(df)
        
        # Paso 7: Verificar dataset final
        self.verify_dataset(df)
        
        # Paso 8: Dividir dataset
        train, val, test = self.split_dataset(df)
        
        # Paso 9: Guardar artifacts
        self.save_artifacts(scaler, encoders)
        
        # Paso 10: Generar reporte HTML
        self.generate_html_report()
        
        # Paso 11: Guardar splits
        self.save_splits(train, val, test)
        
        logger.info("=" * 80)
        logger.info("PIPELINE DE PREPROCESAMIENTO COMPLETADO EXITOSAMENTE!")
        logger.info("=" * 80)
        
        return train, val, test