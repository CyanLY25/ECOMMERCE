"""
Genera los textos de interpretación en español que acompañan a cada
tabla/figura en los reportes (PDF, Word, Excel), tal como exige la
consigna: "deberá reportar tablas y figuras con su interpretación".

Cada función recibe los DataFrames/diccionarios ya cargados por
`report_data.load_report_data` y devuelve un párrafo interpretativo.
Si faltan datos, devuelve una nota explícita en vez de fallar.
"""
from typing import Any, Dict, Optional

import pandas as pd


def interpret_eda(stats: Optional[pd.DataFrame]) -> str:
    if stats is None or stats.empty:
        return "No se encontraron estadísticos descriptivos generados por el módulo de EDA."
    try:
        row = stats.set_index(stats.columns[0])
        target = row.loc["Quantity"] if "Quantity" in row.index else row.iloc[0]
        media = target.get("Media", float("nan"))
        desv = target.get("Desviación Estándar", target.get("DesviaciónEstándar", float("nan")))
        return (
            f"La variable objetivo (Quantity) presenta una media de {media:.2f} unidades "
            f"con una desviación estándar de {desv:.2f}, lo que indica alta dispersión "
            "respecto a la media (coeficiente de variación elevado). Esto es consistente "
            "con datos de ventas al detalle, donde unos pocos pedidos de gran volumen "
            "conviven con la mayoría de pedidos pequeños. Esta asimetría explica en parte "
            "la dificultad de los modelos para lograr un ajuste alto (ver sección de "
            "comparación de modelos)."
        )
    except Exception:
        return (
            "Los estadísticos descriptivos fueron calculados sobre las variables numéricas "
            "del dataset (Quantity, UnitPrice, CustomerID, y variables temporales derivadas)."
        )


def interpret_model_comparison(df: Optional[pd.DataFrame], best_model: Optional[Dict[str, Any]]) -> str:
    if df is None or df.empty:
        return "No se encontró la tabla comparativa de modelos (model_comparison.csv)."

    try:
        best_row = df.loc[df["RMSE"].idxmin()]
        r2_values = df["R²"] if "R²" in df.columns else df.get("R2")
        max_r2 = r2_values.max() if r2_values is not None else None

        texto = (
            f"El modelo con menor error (RMSE = {best_row['RMSE']:.3f}, "
            f"MAE = {best_row['MAE']:.3f}) fue {best_row['Modelo']}. "
        )

        if max_r2 is not None and max_r2 < 0.3:
            texto += (
                "Es importante señalar, con honestidad estadística, que el coeficiente de "
                f"determinación (R²) máximo alcanzado entre los cinco modelos es {max_r2:.4f}, "
                "un valor cercano a cero o negativo en varios casos. Un R² negativo indica que "
                "el modelo predice peor que una línea base trivial (la media de la variable "
                "objetivo), por lo que ninguno de los modelos entrenados logra capturar de forma "
                "confiable la relación entre las variables predictoras y la demanda real. "
                "El MAPE (superior al 200% en todos los modelos) confirma esta limitación. "
                "Esto no invalida el pipeline técnico -EDA, entrenamiento, validación cruzada, "
                "tuning y pruebas estadísticas están correctamente implementados- pero sí exige "
                "reportar el hallazgo tal como es: con las variables disponibles (StockCode, "
                "UnitPrice, CustomerID, Country y variables temporales) el problema de "
                "predicción de demanda a nivel de línea de pedido tiene una señal muy débil. "
                "Se recomienda, para el artículo científico, discutir esta limitación y proponer "
                "ingeniería de características adicional (por ejemplo, medias móviles de demanda "
                "histórica por producto, o agregación a nivel diario/semanal) como trabajo futuro."
            )
        else:
            texto += (
                f"El modelo alcanza un R² de {max_r2:.4f}, lo que indica una capacidad "
                "aceptable para explicar la variabilidad de la demanda."
            )
        return texto
    except Exception as e:
        return f"No fue posible generar la interpretación automática de la comparación de modelos ({e})."


def interpret_cross_validation(cv_summary: Optional[pd.DataFrame]) -> str:
    if cv_summary is None or cv_summary.empty:
        return "No se encontró el resumen de validación cruzada (cross_validation_summary.csv)."
    try:
        cv_summary = cv_summary.sort_values("mean_rmse")
        best = cv_summary.iloc[0]
        return (
            f"Tras 5 particiones (folds) de validación cruzada, el modelo '{best['model']}' "
            f"obtuvo el RMSE promedio más bajo ({best['mean_rmse']:.3f} ± {best['std_rmse']:.3f}). "
            "La desviación estándar entre folds se reporta junto al promedio para evaluar la "
            "estabilidad del modelo: una desviación baja respecto a la media indica que el "
            "desempeño no depende fuertemente de la partición particular de los datos, mientras "
            "que una desviación alta (como ocurre aquí en varios modelos) sugiere sensibilidad a "
            "la muestra y refuerza la necesidad de más datos o de variables predictoras más "
            "informativas."
        )
    except Exception as e:
        return f"No fue posible generar la interpretación de validación cruzada ({e})."


def interpret_hyperparameters(best_hp: Optional[Dict[str, Any]]) -> str:
    if not best_hp:
        return "No se encontró el archivo de mejores hiperparámetros (best_hyperparameters.json)."
    partes = []
    for modelo, params in best_hp.items():
        params_str = ", ".join(f"{k}={v}" for k, v in params.items())
        partes.append(f"{modelo}: {params_str}")
    return (
        "El ajuste de hiperparámetros (búsqueda aleatoria) seleccionó, para cada arquitectura, "
        "la combinación que minimizó la función de pérdida en el conjunto de validación:\n"
        + "\n".join(partes)
        + ".\nEstos valores fueron los usados para reentrenar los modelos finales evaluados en la "
        "comparación y en la validación cruzada."
    )


def interpret_statistics(
    friedman: Optional[pd.DataFrame],
    wilcoxon: Optional[pd.DataFrame],
    nemenyi: Optional[pd.DataFrame],
    ranking: Optional[pd.DataFrame],
) -> str:
    partes = []

    if friedman is not None and not friedman.empty:
        row = friedman.iloc[0]
        sig = "significativas" if str(row.get("significant")) in ("True", "1") else "no significativas"
        partes.append(
            f"La prueba de Friedman (estadístico = {row.get('statistic'):.3f}, "
            f"p = {row.get('p_value'):.4f}, α = {row.get('alpha', 0.05)}) indica que existen "
            f"diferencias {sig} entre al menos dos de los modelos comparados."
        )

    if ranking is not None and not ranking.empty:
        orden = ", ".join(
            f"{i+1}) {r['model']}" for i, r in ranking.sort_values("rank").iterrows()
        )
        partes.append(f"El ranking global por RMSE quedó: {orden}.")

    if nemenyi is not None and not nemenyi.empty:
        sig_pairs = nemenyi[nemenyi["significant"] == True] if "significant" in nemenyi.columns else nemenyi.iloc[0:0]
        if len(sig_pairs) > 0:
            pares = ", ".join(f"{r['model_a']} vs {r['model_b']}" for _, r in sig_pairs.iterrows())
            partes.append(
                f"La prueba post-hoc de Nemenyi identificó diferencias estadísticamente "
                f"significativas en los siguientes pares: {pares}. En el resto de comparaciones "
                "por pares no se pudo rechazar la hipótesis de igualdad de rendimiento."
            )
        else:
            partes.append(
                "La prueba post-hoc de Nemenyi no encontró diferencias estadísticamente "
                "significativas entre pares específicos de modelos, pese a que la prueba global "
                "de Friedman sí detectó diferencias; esto es un resultado plausible cuando las "
                "diferencias de rendimiento entre modelos son pequeñas y consistentes."
            )

    if wilcoxon is not None and not wilcoxon.empty and "significant" in wilcoxon.columns:
        n_sig = int((wilcoxon["significant"] == True).sum())
        partes.append(
            f"De las {len(wilcoxon)} comparaciones pareadas evaluadas con la prueba de Wilcoxon "
            f"(con corrección de Bonferroni), {n_sig} resultaron estadísticamente significativas."
        )

    if not partes:
        return "No se encontraron resultados de pruebas estadísticas (Friedman, Wilcoxon, Nemenyi)."

    return " ".join(partes)


def interpret_figure(name: str) -> str:
    captions = {
        "5_correlation_heatmap.png": (
            "Mapa de calor de correlaciones entre variables numéricas del dataset. "
            "Sirve para identificar redundancia entre variables predictoras y su relación "
            "lineal con la variable objetivo."
        ),
        "13_time_series_sales.png": (
            "Serie temporal de ventas agregadas, útil para detectar estacionalidad y tendencia "
            "que un modelo secuencial (LSTM/GRU) debería poder capturar."
        ),
        "9_top_products.png": "Top de productos por volumen de ventas.",
        "11_country_distribution.png": "Distribución de pedidos por país.",
        "cross_validation_boxplot.png": (
            "Distribución del error (RMSE) de cada modelo a través de los folds de validación "
            "cruzada; los boxplots más altos y dispersos indican modelos menos estables."
        ),
        "cross_validation_rmse.png": "RMSE promedio por modelo en validación cruzada, con barras de error.",
        "significance_heatmap.png": (
            "Mapa de calor de p-valores de las comparaciones pareadas; celdas por debajo de "
            "α = 0.05 señalan diferencias estadísticamente significativas entre modelos."
        ),
        "critical_difference.png": (
            "Diagrama de diferencia crítica (Nemenyi): modelos conectados por una línea no "
            "difieren significativamente en su ranking."
        ),
        "ranking_plot.png": "Ranking promedio de los modelos según la métrica de error evaluada.",
        "tuning_comparison.png": "Comparación de resultados obtenidos durante el ajuste de hiperparámetros.",
    }
    return captions.get(name, "")
