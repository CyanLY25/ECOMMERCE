"""
Generación de reportes en Markdown y HTML.
"""
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from datetime import datetime


class ReportGenerator:
    """
    Clase para generar reportes.
    """
    
    def __init__(self, reports_dir: Path):
        self.reports_dir = reports_dir
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_markdown(
        self,
        friedman_results: Dict[str, Any],
        nemenyi_results: Dict[str, Any],
        wilcoxon_results: Dict[str, Any],
        ci_df: pd.DataFrame,
        ranking_df: pd.DataFrame,
        best_model: str,
        metric: str,
        alpha: float,
        save_path: Path
    ):
        """
        Genera reporte en Markdown.
        """
        md_content = f"""# Reporte de Validación Estadística

Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Resumen Ejecutivo

- **Métrica evaluada**: {metric}
- **Nivel de significancia (α)**: {alpha}
- **Modelo con mejor rendimiento**: {best_model}

---

## 1. Prueba de Friedman

### Resultados
- **Estadístico χ²**: {friedman_results['statistic']:.4f}
- **Valor p**: {friedman_results['p_value']:.6f}
- **α**: {friedman_results['alpha']}

### Conclusión
{friedman_results['conclusion']}

### Ranking Promedio
"""
        # Añadir ranking
        rank_items = sorted(friedman_results['average_ranks'].items(), key=lambda x: x[1])
        for i, (model, rank) in enumerate(rank_items, 1):
            md_content += f"{i}. {model}: {rank:.3f}\n"
            
        md_content += f"""

---

## 2. Test Post-Hoc de Nemenyi

- **Diferencia Crítica (CD)**: {nemenyi_results['critical_difference']:.3f}

### Comparaciones Significativas
"""
        if nemenyi_results['significant_pairs']:
            for pair in nemenyi_results['significant_pairs']:
                md_content += f"- {pair[0]} vs {pair[1]}\n"
        else:
            md_content += "Ninguna comparación resultó significativa.\n"
            
        md_content += f"""

---

## 3. Test de Wilcoxon con Corrección de Bonferroni

- **Número de comparaciones**: {wilcoxon_results['num_comparisons']}

### Resultados
{wilcoxon_results['results_df'].to_markdown(index=False)}

---

## 4. Intervalos de Confianza (95%)

{ci_df.to_markdown(index=False)}

---

## 5. Ranking de Modelos

{ranking_df.to_markdown(index=False)}

---

## 6. Conclusiones Generales

1. **Mejor modelo**: {best_model}
2. **Significancia global**: {'Sí' if friedman_results['significant'] else 'No'}

---

*Report generado automáticamente por el módulo de validación estadística.*
"""
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(md_content)
            
    def generate_html(
        self,
        md_path: Path,
        save_path: Path
    ):
        """
        Genera reporte HTML a partir de Markdown.
        """
        try:
            import markdown
            with open(md_path, "r", encoding="utf-8") as f:
                md_text = f.read()
            html_content = markdown.markdown(md_text, extensions=['tables'])
            
            html_full = f"""<!DOCTYPE html>
<html>
<head>
    <title>Reporte de Validación Estadística</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #34495e; }}
    </style>
</head>
<body>
{html_content}
</body>
</html>
"""
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(html_full)
        except ImportError:
            print("Markdown library not available, skipping HTML report.")
            
    def generate_conclusions(
        self,
        friedman_results: Dict[str, Any],
        ranking_df: pd.DataFrame,
        best_model: str,
        metric: str,
        save_path: Path
    ):
        """
        Genera archivo de conclusiones.
        """
        conclusions = f"""# Conclusiones Estadísticas

Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. Resumen de Resultados
- **Métrica**: {metric}
- **Modelo ganador**: {best_model}

## 2. Justificación Estadística
- Prueba de Friedman: {'Significativa' if friedman_results['significant'] else 'No significativa'}
- p-valor: {friedman_results['p_value']:.6f}

## 3. Interpretación Científica
{f'El modelo {best_model} muestra un rendimiento superior estadísticamente significativo.' if friedman_results['significant'] else 'No se encontraron diferencias significativas entre los modelos evaluados.'}

## 4. Limitaciones
- El análisis se basa en los resultados de Cross Validation.
- La generalización depende de la representatividad del dataset.

## 5. Recomendaciones
- Utilizar el modelo {best_model} para predicciones.
- Considerar validar en datos externos.
"""
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(conclusions)
