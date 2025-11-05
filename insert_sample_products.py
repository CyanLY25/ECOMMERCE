"""
Script para insertar productos de ejemplo en Supabase
Ejecuta este script para poblar tu base de datos con productos de prueba
"""

import toml
import os
from supabase import create_client, Client

# Productos de ejemplo
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
        "name": "Zapatillas Deportivas Running",
        "description": "Zapatillas ligeras con suela de alta tracción. Ideales para correr y entrenar.",
        "price": 159.90,
        "stock": 25
    },
    {
        "name": "Casaca de Cuero Sintético",
        "description": "Casaca elegante de cuero sintético con forro interno. Varios colores disponibles.",
        "price": 199.00,
        "stock": 15
    },
    {
        "name": "Mochila Urbana Impermeable",
        "description": "Mochila espaciosa con compartimento para laptop. Material impermeable de alta calidad.",
        "price": 79.90,
        "stock": 40
    },
    {
        "name": "Camisa Casual a Cuadros",
        "description": "Camisa de manga larga con diseño a cuadros. Tela fresca y cómoda.",
        "price": 55.00,
        "stock": 8
    },
    {
        "name": "Gorra Deportiva Ajustable",
        "description": "Gorra con visera curva y ajuste trasero. Protección UV. Varios colores.",
        "price": 25.00,
        "stock": 60
    },
    {
        "name": "Reloj Digital Inteligente",
        "description": "Smartwatch con monitor de frecuencia cardíaca, podómetro y notificaciones. Resistente al agua.",
        "price": 249.00,
        "stock": 12
    },
    {
        "name": "Billetera de Cuero Genuino",
        "description": "Billetera compacta con múltiples compartimentos para tarjetas y billetes.",
        "price": 45.00,
        "stock": 35
    },
    {
        "name": "Lentes de Sol Polarizados",
        "description": "Lentes con protección UV400 y cristales polarizados. Diseño moderno y elegante.",
        "price": 69.90,
        "stock": 0  # Este producto está agotado para pruebas
    }
]

def insert_products():
    """Inserta los productos de ejemplo en Supabase"""
    try:
        # Leer secrets del archivo TOML
        secrets_path = os.path.join(".streamlit", "secrets.toml")
        if not os.path.exists(secrets_path):
            print(f"❌ No se encontró el archivo {secrets_path}")
            print("   Asegúrate de tener configurado .streamlit/secrets.toml")
            return

        secrets = toml.load(secrets_path)
        url = secrets["supabase"]["url"]
        key = secrets["supabase"]["key"]

        # Conectar a Supabase
        supabase = create_client(url, key)

        print("🔄 Conectando a Supabase...")
        print(f"   URL: {url[:30]}...")
        print()

        # Verificar si ya hay productos
        existing = supabase.table("products").select("*").execute()
        if existing.data and len(existing.data) > 0:
            print(f"⚠️  Ya existen {len(existing.data)} productos en la base de datos.")
            respuesta = input("   ¿Deseas agregar los productos de ejemplo de todas formas? (s/n): ")
            if respuesta.lower() != 's':
                print("❌ Operación cancelada.")
                return

        print("📦 Insertando productos de ejemplo...")
        print()

        # Insertar productos
        success_count = 0
        for i, product in enumerate(SAMPLE_PRODUCTS, 1):
            try:
                response = supabase.table("products").insert(product).execute()
                if response.data:
                    print(f"✅ [{i}/{len(SAMPLE_PRODUCTS)}] {product['name']} - S/ {product['price']:.2f} (Stock: {product['stock']})")
                    success_count += 1
                else:
                    print(f"❌ [{i}/{len(SAMPLE_PRODUCTS)}] Error al insertar {product['name']}")
            except Exception as e:
                print(f"❌ [{i}/{len(SAMPLE_PRODUCTS)}] Error: {str(e)}")

        print()
        print(f"✅ Proceso completado: {success_count}/{len(SAMPLE_PRODUCTS)} productos insertados exitosamente")
        print()
        print("🎉 ¡Listo! Ya puedes usar las aplicaciones:")
        print("   - Panel Vendedor: streamlit run admin_vendedor.py")
        print("   - Tienda Cliente:  streamlit run tienda_cliente.py")

    except Exception as e:
        print(f"❌ Error al conectar con Supabase: {str(e)}")
        print()
        print("💡 Asegúrate de:")
        print("   1. Tener configurado .streamlit/secrets.toml con tus credenciales")
        print("   2. Haber creado la tabla 'products' en Supabase")
        print("   3. Verificar que las credenciales sean correctas")

if __name__ == "__main__":
    # Configurar encoding UTF-8 para Windows
    import sys
    if sys.platform == "win32":
        import codecs
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

    print()
    print("=" * 60)
    print("🛍️  INSERCIÓN DE PRODUCTOS DE EJEMPLO - E-COMMERCE")
    print("=" * 60)
    print()

    insert_products()

    print()
    print("=" * 60)
