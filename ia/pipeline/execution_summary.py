#!/usr/bin/env python3
"""
Módulo para manejar el resumen de ejecución del pipeline.
"""
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import markdown
import webbrowser


class ExecutionSummary:
    """
    Clase para registrar y generar resúmenes de ejecución del pipeline.
    """
    
    def __init__(self, config):
        self.config = config
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.phases: Dict[str, Dict[str, Any]] = {}
        self.best_model: Optional[str] = None
        self.best_model_metrics: Optional[Dict[str, float]] = None
        
    def add_phase(self, name: str, start: float, end: float, success: bool, error: Optional[str] = None):
        """
        Registra una fase del pipeline.
        """
        duration = end - start
        self.phases[name] = {
            "name": name,
            "start_time": datetime.fromtimestamp(start).isoformat(),
            "end_time": datetime.fromtimestamp(end).isoformat(),
            "duration_seconds": duration,
            "success": success,
            "error": error
        }
        
    def finish(self, best_model: Optional[str] = None, best_model_metrics: Optional[Dict[str, float]] = None):
        """
        Marca la ejecución como finalizada.
        """
        self.end_time = time.time()
        self.best_model = best_model
        self.best_model_metrics = best_model_metrics
        
    def get_total_duration(self) -> float:
        """
        Obtiene la duración total del proyecto en segundos.
        """
        if self.end_time is None:
            return 0.0
        return self.end_time - self.start_time
        
    def format_duration(self, seconds: float) -> str:
        """
        Formatea la duración en HH:MM:SS
        """
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
    def generate_json(self) -> Dict[str, Any]:
        """
        Genera el resumen en formato JSON.
        """
        return {
            "preprocessing": "preprocessing" in [p.lower() for p in self.phases],
            "training": "training" in [p.lower() for p in self.phases],
            "model_comparison": "model_comparison" in [p.lower() for p in self.phases],
            "cross_validation": "cross_validation" in [p.lower() for p in self.phases],
            "hyperparameter_tuning": "hyperparameter_tuning" in [p.lower() for p in self.phases],
            "statistical_tests": "statistical_tests" in [p.lower() for p in self.phases],
            "deployment": "deployment" in [p.lower() for p in self.phases],
            "best_model": self.best_model,
            "best_model_metrics": self.best_model_metrics,
            "execution_time": self.format_duration(self.get_total_duration()),
            "phases": list(self.phases.values())
        }
        
    def generate_markdown(self) -> str:
        """
        Genera el resumen en formato Markdown.
        """
        md_lines = []
        md_lines.append("# Resumen del Proyecto de Predicción de Demanda\n")
        md_lines.append(f"**Fecha:** {datetime.now().strftime('%Y-%m-%d')}\n")
        md_lines.append(f"**Hora:** {datetime.now().strftime('%H:%M:%S')}\n")
        md_lines.append("**Versión:** 1.0\n\n")
        
        # Dataset info
        md_lines.append("## 1. Dataset Utilizado\n")
        md_lines.append("- Archivo: OnlineRetail.xlsx\n")
        if self.config.TRAIN_DATA_PATH.exists():
            import pandas as pd
            try:
                df = pd.read_csv(self.config.TRAIN_DATA_PATH)
                md_lines.append(f"- Número de registros: {len(df)}\n")
            except Exception:
                pass
        md_lines.append("\n")
        
        # Fases ejecutadas
        md_lines.append("## 2. Fases Ejecutadas\n")
        for phase_name, phase_info in self.phases.items():
            status = "✅ Completado" if phase_info["success"] else "❌ Fallido"
            md_lines.append(f"- **{phase_name}**: {status}\n")
        md_lines.append("\n")
        
        # Modelos entrenados
        md_lines.append("## 3. Modelos Entrenados\n")
        model_names = ["MLP", "LSTM", "GRU", "CNN-LSTM", "CNN-GRU"]
        for model_name in model_names:
            model_path = self.config.MODELS_DIR / f"{model_name.lower().replace('-', '_')}.keras"
            if model_path.exists():
                metrics_path = self.config.MODELS_DIR / f"{model_name.lower().replace('-', '_')}_metrics.json"
                if metrics_path.exists():
                    import json
                    with open(metrics_path, "r") as f:
                        metrics = json.load(f)
                    md_lines.append(f"- **{model_name}**: RMSE={metrics.get('rmse'):.4f}, MAE={metrics.get('mae'):.4f}, R²={metrics.get('r2'):.4f}\n")
        md_lines.append("\n")
        
        # Mejor modelo
        if self.best_model:
            md_lines.append("## 4. Mejor Modelo\n")
            md_lines.append(f"- **Modelo**: {self.best_model}\n")
            if self.best_model_metrics:
                md_lines.append(f"- **RMSE**: {self.best_model_metrics.get('rmse'):.4f}\n")
                md_lines.append(f"- **MAE**: {self.best_model_metrics.get('mae'):.4f}\n")
                md_lines.append(f"- **R²**: {self.best_model_metrics.get('r2'):.4f}\n")
            md_lines.append("\n")
        
        # Deployment
        if "deployment" in [p.lower() for p in self.phases]:
            md_lines.append("## 5. Deployment\n")
            if (self.config.BACKEND_BEST_MODEL_PATH.exists() and 
                self.config.BACKEND_MODEL_INFO_PATH.exists()):
                md_lines.append("- **Estado**: ✅ Correcto\n")
            else:
                md_lines.append("- **Estado**: ⚠️ Incompleto\n")
            md_lines.append("\n")
        
        # Tiempo total
        md_lines.append("## 6. Tiempo Total\n")
        md_lines.append(f"- **{self.format_duration(self.get_total_duration())}**\n")
        
        return "\n".join(md_lines)
        
    def save_json(self):
        """
        Guarda el resumen en formato JSON.
        """
        summary = self.generate_json()
        with open(self.config.PROJECT_SUMMARY_JSON, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
            
    def save_markdown(self):
        """
        Guarda el resumen en formato Markdown y lo convierte a HTML.
        """
        md_content = self.generate_markdown()
        with open(self.config.PROJECT_SUMMARY_MD, "w", encoding="utf-8") as f:
            f.write(md_content)
            
        # Convertir a HTML
        html_content = markdown.markdown(md_content, extensions=['tables'])
        full_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Resumen del Proyecto</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #3498db; }}
        .success {{ color: green; }}
        .warning {{ color: orange; }}
    </style>
</head>
<body>
    {html_content}
</body>
</html>
        """
        with open(self.config.PROJECT_SUMMARY_HTML, "w", encoding="utf-8") as f:
            f.write(full_html)
            
    def validate_files(self) -> List[str]:
        """
        Verifica que existan los archivos necesarios.
        """
        warnings = []
        
        required_files = [
            self.config.BACKEND_BEST_MODEL_PATH,
            self.config.BACKEND_MODEL_INFO_PATH,
            self.config.BEST_MODEL_PATH,
            self.config.CV_RESULTS_CSV_PATH,
            self.config.TUNING_RESULTS_JSON_PATH,
            self.config.STATISTICS_REPORT_HTML
        ]
        
        for file_path in required_files:
            if not file_path.exists():
                warnings.append(str(file_path.name))
                
        return warnings
