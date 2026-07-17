"""
Script opcional para poblar/repoblar la tienda con productos.

Ya NO usa Supabase. El backend (backend/store.py) siembra automáticamente
los 30 productos más vendidos del dataset histórico apenas arranca, si la
tabla de productos está vacía, así que normalmente no necesitas correr
este script.

Úsalo solo si quieres, por ejemplo, agregar productos de ejemplo
adicionales a mano contra el backend ya desplegado.
"""

import sys
import requests

DEFAULT_API_URL = "https://ecommerce-9u1d.onrender.com"

SAMPLE_PRODUCTS = [
    {
        "name": "Polo Básico Blanco",
        "description": "Polo de algodón 100% suave y cómodo. Ideal para uso diario. Tallas: S, M, L, XL.",
        "price": 35.00,
        "stock": 50
    },
    {
        "name": "Jean Clásico Azul",
        "description": "Jean de mezclilla resistente con corte clásico. Perfecto para cualquier ocasión.",
        "price": 89.90,
        "stock": 30
    },
    {
        "name": "Mochila Urbana Impermeable",
        "description": "Mochila espaciosa con compartimento para laptop. Material impermeable de alta calidad.",
        "price": 79.90,
        "stock": 40
    },
]


def insert_products(api_url: str):
    """Inserta productos de ejemplo contra el backend propio (no Supabase)."""
    print(f"🔄 Conectando al backend en {api_url} ...")

    try:
        existing = requests.get(f"{api_url}/api/store/products", timeout=20)
        existing.raise_for_status()
        current = existing.json().get("products", [])
    except Exception as e:
        print(f"❌ No se pudo conectar al backend: {e}")
        return

    print(f"   Actualmente hay {len(current)} productos en la tienda.")
    print()

    success_count = 0
    for i, product in enumerate(SAMPLE_PRODUCTS, 1):
        try:
            resp = requests.post(f"{api_url}/api/store/products", json=product, timeout=20)
            if resp.status_code == 201:
                print(f"✅ [{i}/{len(SAMPLE_PRODUCTS)}] {product['name']} - S/ {product['price']:.2f}")
                success_count += 1
            else:
                print(f"❌ [{i}/{len(SAMPLE_PRODUCTS)}] Error: {resp.text}")
        except Exception as e:
            print(f"❌ [{i}/{len(SAMPLE_PRODUCTS)}] Error: {e}")

    print()
    print(f"✅ Proceso completado: {success_count}/{len(SAMPLE_PRODUCTS)} productos insertados")


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_API_URL
    print()
    print("=" * 60)
    print("🛍️  INSERCIÓN DE PRODUCTOS DE EJEMPLO - E-COMMERCE")
    print("=" * 60)
    print()
    insert_products(url)
    print()
    print("=" * 60)
