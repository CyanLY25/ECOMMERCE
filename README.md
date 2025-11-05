# 🛍️ Sistema E-Commerce con Streamlit + Supabase + n8n

Sistema completo de comercio electrónico con dos aplicaciones web:
- **Panel de Vendedor**: Administración de productos y órdenes
- **Tienda Cliente**: E-commerce para que los clientes realicen pedidos

## ✨ Características Principales

### 🏪 Panel de Vendedor (`admin_vendedor.py`)

- **Dashboard Interactivo**
  - Métricas en tiempo real (productos, órdenes, ingresos)
  - Órdenes recientes con estados visuales
  - Diseño minimalista e intuitivo

- **Gestión Completa de Productos (CRUD)**
  - Crear, editar y eliminar productos
  - Control de stock con indicadores visuales
  - Precios y descripciones personalizables

- **Gestión de Órdenes**
  - Visualización detallada de cada pedido
  - Filtros por estado (Pendiente, Completado, Cancelado)
  - Cambio de estado con actualización automática de stock
  - Al marcar como "Completado", el stock se descuenta automáticamente

### 🛍️ Tienda Cliente (`tienda_cliente.py`)

- **Catálogo de Productos**
  - Grid responsive de 3 columnas
  - Badges de disponibilidad por stock
  - Productos agotados visibles pero deshabilitados

- **Carrito de Compras**
  - Agregar/quitar productos
  - Modificar cantidades
  - Cálculo automático de totales
  - Indicador flotante del carrito

- **Checkout Inteligente**
  - Formulario de datos del cliente
  - Validaciones robustas (email, teléfono, campos obligatorios)
  - Verificación de stock antes de confirmar
  - Mensaje de éxito con confetti

- **Integración Automática**
  - Al crear orden, trigger de Supabase activa n8n
  - n8n envía notificaciones automáticas:
    - 📱 WhatsApp al cliente
    - 📧 Email al vendedor
    - 📊 Registro en Google Sheets

## 🚀 Instalación

### Requisitos Previos

- Python 3.8 o superior
- Cuenta en [Supabase](https://supabase.com)
- Cuenta en [n8n](https://n8n.io) (opcional, para notificaciones)

### 1. Clonar el Repositorio

```bash
git clone <tu-repositorio>
cd ECOMMERCE
```

### 2. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 3. Configurar Supabase

#### a) Crear las Tablas

Ejecuta el siguiente SQL en el editor SQL de Supabase:

```sql
-- Tabla de productos
CREATE TABLE products (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  name TEXT NOT NULL,
  description TEXT,
  price DECIMAL(10, 2) NOT NULL DEFAULT 0.00,
  stock INT NOT NULL DEFAULT 0,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Tabla de órdenes
CREATE TABLE orders (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  customer_name TEXT NOT NULL,
  customer_email TEXT,
  customer_phone TEXT,
  shipping_address TEXT,
  total_amount DECIMAL(10, 2) NOT NULL,
  status TEXT NOT NULL DEFAULT 'Pendiente',
  items JSONB,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Índices para optimización
CREATE INDEX idx_products_name ON products(name);
CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_orders_customer_phone ON orders(customer_phone);
```

#### b) Configurar el Trigger para n8n (Opcional)

Si deseas que las órdenes activen automáticamente n8n:

```sql
-- Habilitar extensión HTTP
CREATE EXTENSION IF NOT EXISTS http WITH SCHEMA extensions;

-- Función que envía datos a n8n
CREATE OR REPLACE FUNCTION notify_new_order_to_n8n()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
  webhook_url TEXT := 'TU_WEBHOOK_N8N_AQUI';
  payload JSONB;
  http_response extensions.http_response;
BEGIN
  payload := jsonb_build_object(
    'type', 'INSERT',
    'table', 'orders',
    'schema', 'public',
    'record', row_to_json(NEW)::jsonb,
    'timestamp', NOW()
  );

  SELECT * INTO http_response
  FROM extensions.http_post(
    webhook_url,
    payload::text,
    'application/json'
  );

  RAISE NOTICE 'Webhook enviado. Status: %', http_response.status;

  IF http_response.status NOT BETWEEN 200 AND 299 THEN
    RAISE WARNING 'Error al enviar webhook: Status %, Content: %',
      http_response.status,
      http_response.content;
  END IF;

  RETURN NEW;
EXCEPTION
  WHEN OTHERS THEN
    RAISE WARNING 'Error en función notify_new_order_to_n8n: %', SQLERRM;
    RETURN NEW;
END;
$$;

-- Crear trigger
DROP TRIGGER IF EXISTS trigger_notify_new_order ON orders;

CREATE TRIGGER trigger_notify_new_order
  AFTER INSERT ON orders
  FOR EACH ROW
  EXECUTE FUNCTION notify_new_order_to_n8n();
```

**Nota:** Reemplaza `TU_WEBHOOK_N8N_AQUI` con la URL de tu webhook de n8n.

### 4. Configurar Credenciales

Edita el archivo `.streamlit/secrets.toml` y agrega tus credenciales de Supabase:

```toml
[supabase]
url = "https://tu-proyecto.supabase.co"
key = "tu-anon-key-aqui"
```

Para obtener estas credenciales:
1. Ve a tu proyecto en Supabase
2. Settings → API
3. Copia la "Project URL" y la "anon/public key"

### 5. Insertar Productos de Ejemplo

```bash
streamlit run insert_sample_products.py
```

Este script insertará 10 productos de ejemplo en tu base de datos.

### 6. Ejecutar las Aplicaciones

**Panel de Vendedor:**
```bash
streamlit run admin_vendedor.py
```

**Tienda Cliente:**
```bash
streamlit run tienda_cliente.py
```

## 📖 Uso del Sistema

### Flujo de Trabajo Completo

1. **Vendedor configura productos** (admin_vendedor.py)
   - Crea productos con nombre, descripción, precio y stock
   - Los productos aparecen automáticamente en la tienda

2. **Cliente realiza pedido** (tienda_cliente.py)
   - Navega el catálogo
   - Agrega productos al carrito
   - Completa formulario de datos
   - Confirma el pedido

3. **Sistema procesa orden**
   - Se crea la orden en Supabase con estado "Pendiente"
   - El trigger activa n8n automáticamente
   - n8n envía notificaciones (WhatsApp, Email, Sheets)

4. **Vendedor gestiona orden** (admin_vendedor.py)
   - Revisa la orden en "Gestión de Órdenes"
   - Ve todos los detalles del cliente y productos
   - Al marcar como "Completado", se descuenta el stock automáticamente

### Gestión de Stock

El sistema maneja el stock de forma inteligente:

- **Al crear orden**: El stock NO se descuenta (permite cancelaciones sin afectar inventario)
- **Al marcar como "Completado"**: El stock se descuenta automáticamente
- **Validación**: Antes del checkout, se valida que haya stock suficiente

### Estados de Órdenes

- **Pendiente** (🟡): Orden recién creada, esperando procesamiento
- **Completado** (🟢): Orden procesada y despachada, stock descontado
- **Cancelado** (🔴): Orden cancelada, stock no afectado

## 🎨 Personalización

### Cambiar Colores

Los colores principales están definidos en los estilos CSS de cada archivo. Para cambiar el color primario (verde por defecto):

1. Abre `admin_vendedor.py` o `tienda_cliente.py`
2. Busca `#059669` (verde primario)
3. Reemplaza por tu color preferido en todas las instancias

### Modificar Productos de Ejemplo

Edita el archivo `insert_sample_products.py` y modifica el array `SAMPLE_PRODUCTS`.

### Ajustar Validaciones

Las validaciones están en `tienda_cliente.py`:
- `validate_email()`: Formato de email
- `validate_phone()`: Formato de teléfono (9+ dígitos)
- `validate_stock_availability()`: Disponibilidad de stock

## 🔧 Configuración de n8n (Opcional)

### Estructura del Flujo

El flujo de n8n debe:
1. Recibir webhook con datos de la orden
2. Validar datos (teléfono, email)
3. Normalizar formato de datos
4. Enviar notificaciones en paralelo:
   - WhatsApp (Twilio u otro proveedor)
   - Email (SMTP, SendGrid, etc.)
   - Google Sheets
5. Responder con éxito/error

### Ejemplo de Datos Recibidos

```json
{
  "type": "INSERT",
  "table": "orders",
  "schema": "public",
  "record": {
    "id": "uuid-aqui",
    "customer_name": "Juan Pérez García",
    "customer_email": "juan.perez@gmail.com",
    "customer_phone": "921971743",
    "shipping_address": "Av. Larco 123, Miraflores, Lima",
    "total_amount": 35.00,
    "status": "Pendiente",
    "items": [
      {
        "name": "Polo Básico Blanco",
        "quantity": 1,
        "price": 35.00
      }
    ],
    "created_at": "2024-01-15T10:30:00Z"
  }
}
```

## 🐛 Troubleshooting

### Error: "Error al conectar con Supabase"

- Verifica que `.streamlit/secrets.toml` tenga las credenciales correctas
- Asegúrate de que la URL termine en `.supabase.co`
- Confirma que estés usando la `anon/public` key, no la `service_role` key

### Error: "Error al obtener productos/órdenes"

- Verifica que las tablas `products` y `orders` existan en Supabase
- Revisa que los nombres de las columnas coincidan con el esquema
- Comprueba los permisos RLS (Row Level Security) en Supabase

### Los productos no aparecen en la tienda

- Ejecuta `insert_sample_products.py` para agregar productos de prueba
- Verifica en Supabase → Table Editor que los productos existan
- Asegúrate de que `stock >= 0`

### El trigger no activa n8n

- Verifica que la extensión `http` esté habilitada en Supabase
- Comprueba que la URL del webhook sea correcta
- Revisa los logs en Supabase → Database → Logs
- Verifica que n8n esté ejecutándose y el webhook activo

### Stock negativo después de completar órdenes

- Esto puede ocurrir si una orden se marca como "Completado" múltiples veces
- El sistema previene stocks negativos poniéndolos en 0
- Ajusta manualmente el stock desde "Gestión de Productos"

## 📁 Estructura del Proyecto

```
ECOMMERCE/
│
├── .streamlit/
│   └── secrets.toml          # Credenciales (no incluir en git)
│
├── admin_vendedor.py          # Panel de administración
├── tienda_cliente.py          # Tienda para clientes
├── insert_sample_products.py  # Script de productos de ejemplo
│
├── requirements.txt           # Dependencias Python
├── .gitignore                # Archivos a ignorar en git
├── README.md                 # Este archivo
└── INICIO_RAPIDO.md          # Guía de inicio rápido
```

## 🔒 Seguridad

### Buenas Prácticas

- **NUNCA** commitear `.streamlit/secrets.toml` a git
- Usa la `anon/public` key de Supabase, no la `service_role`
- Habilita RLS (Row Level Security) en Supabase para producción
- Valida todos los inputs del usuario
- Usa HTTPS en producción

### Row Level Security (RLS)

Para producción, habilita RLS en Supabase:

```sql
-- Habilitar RLS
ALTER TABLE products ENABLE ROW LEVEL SECURITY;
ALTER TABLE orders ENABLE ROW LEVEL SECURITY;

-- Permitir lectura pública de productos
CREATE POLICY "Productos visibles públicamente" ON products
  FOR SELECT USING (true);

-- Permitir inserción pública de órdenes
CREATE POLICY "Permitir crear órdenes" ON orders
  FOR INSERT WITH CHECK (true);
```

## 🚀 Despliegue en Producción

### Streamlit Cloud

1. Sube tu código a GitHub (sin secrets.toml)
2. Conéctate a [Streamlit Cloud](https://streamlit.io/cloud)
3. Despliega ambas aplicaciones
4. Configura los secrets en la interfaz web de Streamlit Cloud

### Otras Opciones

- **Heroku**: Usa el buildpack de Python
- **Railway**: Deployment automático desde GitHub
- **DigitalOcean**: Usa App Platform
- **AWS**: EC2 + Nginx como reverse proxy

## 📝 Licencia

Este proyecto es de código abierto. Siéntete libre de usarlo y modificarlo según tus necesidades.

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Para cambios importantes:
1. Abre un issue primero para discutir los cambios
2. Fork el proyecto
3. Crea un branch para tu feature
4. Commit tus cambios
5. Push al branch
6. Abre un Pull Request

## 📧 Soporte

Si tienes problemas o preguntas:
- Abre un issue en GitHub
- Revisa la sección de Troubleshooting
- Consulta la documentación de [Supabase](https://supabase.com/docs)
- Revisa los logs de [n8n](https://docs.n8n.io)

## 🎉 Agradecimientos

- **Streamlit**: Framework para aplicaciones web en Python
- **Supabase**: Backend as a Service con PostgreSQL
- **n8n**: Herramienta de automatización workflow

---

⭐ Si te gusta este proyecto, dale una estrella en GitHub!
