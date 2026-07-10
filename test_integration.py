#!/usr/bin/env python3
"""
Script de prueba para verificar la integración de FastAPI con el ModelLoader.
"""
import asyncio
import sys
from pathlib import Path
from typing import Dict, Any

# Añadir el directorio raíz del proyecto al path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "backend"))


async def test_model_loader():
    """Testear la carga del modelo"""
    print("=" * 60)
    print("Probando ModelLoader...")
    from ia.inference.model_loader import model_loader_instance
    print(f"Modelo cargado: {model_loader_instance.is_loaded}")
    if model_loader_instance.is_loaded:
        print(f"Nombre del modelo: {model_loader_instance.model_name}")
        print(f"Métricas: {model_loader_instance.metrics}")
        print(f"Ruta: {model_loader_instance.model_path}")
    print("=" * 60)
    return model_loader_instance


async def test_endpoints():
    """Testear los endpoints de FastAPI"""
    print("\n" + "=" * 60)
    print("Probando endpoints...")
    
    # Importar la app de FastAPI
    from fastapi.testclient import TestClient
    from backend.app import app

    client = TestClient(app)

    # Test /api/model-info
    print("\n1. Test /api/model-info")
    try:
        response = client.get("/api/model-info")
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.json()}")
        assert response.status_code == 200
    except Exception as e:
        print(f"Error en /api/model-info: {e}")
        raise

    # Test /api/health
    print("\n2. Test /api/health")
    try:
        response = client.get("/api/health")
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.json()}")
        assert response.status_code == 200
    except Exception as e:
        print(f"Error en /api/health: {e}")
        raise

    # Test /api/predict con un sample válido
    print("\n3. Test /api/predict")
    try:
        sample_features = {
            "StockCode": 2790, 
            "UnitPrice": -0.08434179301427071, 
            "CustomerID": 1358, 
            "Country": 35, 
            "Año": 2011, 
            "Mes": 8, 
            "Día": 25, 
            "Hora": 14, 
            "DíaSemana": 3, 
            "SemanaAño": 34, 
            "Trimestre": 3, 
            "EsFinDeSemana": 0, 
            "MesNombre": 1
        }
        response = client.post("/api/predict", json={"features": sample_features})
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.json()}")
        assert response.status_code == 200
    except Exception as e:
        print(f"Error en /api/predict: {e}")
        raise

    print("\n" + "=" * 60)
    print("✅ Todas las pruebas pasaron!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_endpoints())
