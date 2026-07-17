# 🛍️ Sistema E-Commerce con Streamlit + FastAPI

Sistema completo de comercio electrónico con dos aplicaciones web:
- **Panel de Vendedor**: Administración de productos y órdenes
- **Tienda Cliente**: E-commerce para que los clientes realicen pedidos

Ambas se despliegan por separado en Streamlit Community Cloud y comparten
datos a través de **nuestro propio backend FastAPI** (el mismo que ya expone
`/api/predict` para el módulo de predicción de demanda), que guarda todo en
una base de datos SQLite propia. **Ya no se usa Supabase.**

## 🧩 Arquitectura

```
┌─────────────────────┐        ┌──────────────────────┐
│  admin_vendedor.py   │──HTTP─▶│                       │
│  (Streamlit Cloud)   │        │   Backend FastAPI     │
└─────────────────────┘        │   (Render)            │
                                │                        │
┌─────────────────────┐        │  /api/store/products   │
│  tienda_cliente.py   │──HTTP─▶│  /api/store/orders     │
│  (Streamlit Cloud)   │        │  (SQLite: store.db)   │
└─────────────────────┘        │                        │
                                │  /api/predict (IA)    │
                                └──────────────────────┘
```

- `store_api.py` (en la raíz del repo) es el cliente HTTP compartido que
  usan tanto `admin_vendedor.py` como `tienda_cliente.py` para hablar con el
  backend. Reemplaza al cliente de Supabase.
- `backend/store.py` implementa la persistencia (productos y órdenes) con
  SQLite, sin depender de ningún servicio externo.
- `backend/routes.py` expone esos datos como endpoints REST bajo
  `/api/store/...`.
- Al arrancar, si la tabla de productos está vacía, el backend se
  autosiembra con los **30 productos más vendidos** del dataset histórico
  (`backend/model/top_products_seed.json`) — los mismos StockCode que tienen
  historial de demanda calculado para el módulo de predicción, así el
  catálogo de la tienda queda coherente con lo que se puede predecir.

## ✨ Características Principales

### 🏪 Panel de Vendedor (`admin_vendedor.py`)

- **Dashboard Interactivo**
  - Métricas en tiempo real (productos, órdenes, ingresos)
  - Órdenes recientes con estados visuales

- **Gestión Completa de Productos (CRUD)**
  - Crear, editar y eliminar productos
  - Imágenes: se suben como archivo y se guardan como data URI directamente
    en la base de datos del backend (sin necesitar un servicio de storage
    externo)
  - Control de stock con indicadores visuales

- **Gestión de Órdenes**
  - Visualización detallada de cada pedido
  - Filtros por estado (Pendiente, Completado, Cancelado)
  - Al marcar como "Completado", el **backend** descuenta el stock
    automáticamente (una sola vez, de forma centralizada)

### 🛍️ Tienda Cliente (`tienda_cliente.py`)

- **Catálogo de Productos**: sembrado con los 30 productos más vendidos,
  con su código de producto visible para relacionarlo con el módulo de
  predicción de demanda
- **Carrito de Compras**: agregar/quitar, modificar cantidades, totales
- **Checkout**: formulario con validaciones (email, teléfono, dirección) y
  verificación de stock contra el backend antes de confirmar

## 🚀 Instalación y Ejecución Local

### Requisitos Previos

- Python 3.10+

### 1. Clonar el Repositorio

```bash
git clone <tu-repositorio>
cd ECOMMERCE
```

### 2. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 3. Configurar la URL del backend (opcional)

Por defecto, `store_api.py` apunta al backend ya desplegado en Render:
`https://ecommerce-9u1d.onrender.com`. Si quieres apuntar a tu propia
instancia (por ejemplo, corriendo el backend en local), crea
`.streamlit/secrets.toml`:

```toml
[backend]
api_url = "http://localhost:8000"

[app_urls]
prediccion = "https://tu-app-prediccion.streamlit.app"
vendedor = "https://tu-app-vendedor.streamlit.app"
tienda = "https://tu-app-tienda.streamlit.app"
```

Si no configuras nada, las apps funcionan igual usando el backend de Render
por defecto.

### 4. Ejecutar las Aplicaciones

**Backend (opcional en local; ya está desplegado en Render):**
```bash
uvicorn backend.app:app --reload
```

**Panel de Vendedor:**
```bash
streamlit run admin_vendedor.py
```

**Tienda Cliente:**
```bash
streamlit run tienda_cliente.py
```

Los 30 productos más vendidos aparecen automáticamente la primera vez que
el backend arranca (no hace falta ejecutar ningún script de siembra).

## 📖 Uso del Sistema

### Flujo de Trabajo Completo

1. **La tienda ya arranca poblada** con los 30 productos más vendidos del
   histórico (sembrados por el backend).
2. **Vendedor gestiona productos** (`admin_vendedor.py`): puede editar
   precio/stock/imagen, agregar productos nuevos o eliminar.
3. **Cliente realiza pedido** (`tienda_cliente.py`): navega el catálogo,
   arma su carrito y hace checkout.
4. **El backend valida y crea la orden** (verifica stock antes de aceptarla).
5. **Vendedor gestiona la orden** (`admin_vendedor.py`): al marcarla como
   "Completado", el backend descuenta el stock automáticamente.

### Gestión de Stock

- **Al crear orden**: el stock NO se descuenta todavía (permite cancelar
  sin afectar inventario), pero sí se valida que haya suficiente.
- **Al marcar como "Completado"**: el backend descuenta el stock una sola
  vez, de forma centralizada (evita condiciones de carrera entre el panel
  de vendedor y la tienda).

### Estados de Órdenes

- **Pendiente** (🟡): orden recién creada
- **Completado** (🟢): procesada y despachada, stock descontado
- **Cancelado** (🔴): orden cancelada, stock no afectado

## 📁 Estructura del Proyecto (relevante para la tienda)

```
ECOMMERCE/
│
├── admin_vendedor.py           # Panel de administración (Streamlit)
├── tienda_cliente.py           # Tienda para clientes (Streamlit)
├── store_api.py                # Cliente HTTP compartido hacia el backend
├── insert_sample_products.py   # Script opcional para agregar productos a mano
│
├── backend/
│   ├── app.py                  # App FastAPI (incluye router de predicción y de tienda)
│   ├── routes.py                # Endpoints /api/predict/... y /api/store/...
│   ├── store.py                 # Persistencia SQLite de productos/órdenes
│   └── model/
│       └── top_products_seed.json  # Los 30 productos más vendidos (seed inicial)
│
├── requirements.txt
└── README.md
```

## 🐛 Troubleshooting

### "Error al obtener productos/órdenes"

- El backend en Render puede tardar ~30-60s en "despertar" si estuvo
  inactivo (plan gratuito). Reintenta en unos segundos.
- Verifica la URL configurada en `st.secrets["backend"]["api_url"]` (si la
  configuraste) o revisa que `https://ecommerce-9u1d.onrender.com/api/store/products`
  responda en el navegador.

### Los productos no aparecen en la tienda

- La siembra automática solo ocurre si la tabla está vacía. Si el backend
  se redesplegó y perdió su disco (plan gratuito de Render es efímero),
  volverá a autosembrarse con los 30 productos en el próximo arranque.
- También puedes correr `python insert_sample_products.py` para agregar
  productos manualmente contra el backend.

### Stock negativo después de completar órdenes

- El sistema previene stocks negativos poniéndolos en 0.
- Ajusta manualmente el stock desde "Gestión de Productos".

## 🚀 Despliegue en Producción

### Streamlit Community Cloud

Cada archivo (`admin_vendedor.py`, `tienda_cliente.py` y
`frontend/streamlit_app.py`) se despliega como una app independiente,
apuntando al mismo repositorio de GitHub:

1. Sube tu código a GitHub.
2. En [Streamlit Cloud](https://streamlit.io/cloud), crea una app por cada
   archivo principal (`admin_vendedor.py`, `tienda_cliente.py`).
3. No necesitas configurar ningún secret para que funcionen: por defecto
   usan el backend ya desplegado en Render. Si quieres usar otro backend,
   agrega `[backend] api_url = "..."` en los secrets de cada app.
4. (Opcional) Configura `[app_urls]` en los secrets de las tres apps para
   que se enlacen entre sí desde la barra lateral.

### Backend (Render)

El backend FastAPI (`backend/app.py`) ya está desplegado en Render y sirve
tanto las predicciones de demanda como los datos de la tienda. Si necesitas
redesplegarlo, el `Procfile` en la raíz define el comando de arranque
(`uvicorn backend.app:app`).

## 📝 Licencia

Este proyecto es de código abierto. Siéntete libre de usarlo y modificarlo
según tus necesidades.

---

⭐ Si te gusta este proyecto, dale una estrella en GitHub!
