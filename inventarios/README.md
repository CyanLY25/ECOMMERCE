# Inventarios - Gestion de Stock con Streamlit y Supabase

Aplicacion Streamlit de un solo archivo para administrar lineas, productos y ordenes almacenadas en Supabase.

## Caracteristicas
- Panel de resumen con metricas de stock, valor de inventario y alertas de productos bajo minimo.
- Mantenedor de lineas, productos y ordenes con formularios para crear y actualizar registros.
- Graficas interactivas (Plotly) para stock por linea y movimiento de ordenes.
- Generacion de reportes PDF descargables con el estado actual del inventario.

## Requisitos
Instala las dependencias listadas en `requirements.txt` dentro de este directorio:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Configuracion
1. Copia `./.streamlit/secrets.toml.example` a `./.streamlit/secrets.toml` y completa tu configuracion Supabase:
   ```toml
   [supabase]
   url = "https://<tu-proyecto>.supabase.co"
   key = "<anon-o-service-role-key>"
   lineas_table = "lineas"       # opcional
   productos_table = "productos" # opcional
   ordenes_table = "ordenes"     # opcional
   ```
2. Ejecuta el script `supabase_schema.sql` en la consola SQL de Supabase para crear las tablas y politicas requeridas.

## Ejecucion
Inicia la aplicacion desde este directorio:

```bash
streamlit run inventarios.py
```

## Estructura del proyecto
- `inventarios.py`: interfaz Streamlit con logica de consultas, graficas y reportes.
- `supabase_schema.sql`: definicion SQL de las tablas `lineas`, `productos` y `ordenes`.
- `requirements.txt`: dependencias necesarias para correr la app.
- `.streamlit/secrets.toml.example`: plantilla de credenciales Supabase.

## Notas
- Las notificaciones se integraran posteriormente (por ejemplo, via n8n). Actualmente la app expone en la UI los productos bajo stock minimo para facilitar esa automatizacion.
- Recarga la pagina despues de crear o actualizar registros para reflejar los cambios cacheados en Streamlit (el propio flujo del app ya fuerza recargas en la mayoria de los casos).
