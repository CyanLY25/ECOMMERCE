"""
🛍️ TIENDA CLIENTE - E-COMMERCE
Tienda online para que los clientes realicen sus pedidos
"""

import streamlit as st
from supabase import create_client, Client
import json
import re
from datetime import datetime

# ==================== CONFIGURACIÓN ====================

st.set_page_config(
    page_title="Tienda Online - E-Commerce",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

from app_links import render_app_navigation
render_app_navigation("tienda")

# CSS para estandarizar el tamaño de las imágenes de productos
st.markdown("""
    <style>
    /* Contenedor de imagen de producto con tamaño fijo */
    .product-image-container {
        width: 100%;
        height: 250px;
        overflow: hidden;
        display: flex;
        align-items: center;
        justify-content: center;
        background-color: #f8f9fa;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .product-image-container img {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== CONEXIÓN A SUPABASE ====================

@st.cache_resource
def init_supabase() -> Client:
    """Inicializa la conexión a Supabase"""
    try:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
        return create_client(url, key)
    except Exception as e:
        st.error(f"❌ Error al conectar con Supabase: {str(e)}")
        st.info("💡 Asegúrate de configurar tus credenciales en `.streamlit/secrets.toml`")
        st.stop()

supabase = init_supabase()

# ==================== INICIALIZAR SESIÓN ====================

if 'cart' not in st.session_state:
    st.session_state.cart = []

if 'order_completed' not in st.session_state:
    st.session_state.order_completed = False

if 'order_id' not in st.session_state:
    st.session_state.order_id = None

# ==================== FUNCIONES DE PRODUCTOS ====================

def get_all_products():
    """Obtiene todos los productos disponibles"""
    try:
        response = supabase.table("products").select("*").order("created_at", desc=True).execute()
        return response.data
    except Exception as e:
        st.error(f"Error al obtener productos: {str(e)}")
        return []

# ==================== FUNCIONES DE CARRITO ====================

def add_to_cart(product, quantity):
    """Agrega un producto al carrito"""
    # Verificar si el producto ya está en el carrito
    for item in st.session_state.cart:
        if item['id'] == product['id']:
            item['quantity'] += quantity
            return True

    # Si no está, agregarlo
    st.session_state.cart.append({
        'id': product['id'],
        'name': product['name'],
        'price': float(product['price']),
        'quantity': quantity,
        'stock_available': product['stock']
    })
    return True

def remove_from_cart(product_id):
    """Elimina un producto del carrito"""
    st.session_state.cart = [item for item in st.session_state.cart if item['id'] != product_id]

def update_cart_quantity(product_id, new_quantity):
    """Actualiza la cantidad de un producto en el carrito"""
    for item in st.session_state.cart:
        if item['id'] == product_id:
            if new_quantity <= 0:
                remove_from_cart(product_id)
            else:
                item['quantity'] = new_quantity
            break

def clear_cart():
    """Vacía el carrito completamente"""
    st.session_state.cart = []

def get_cart_total():
    """Calcula el total del carrito"""
    return sum(item['price'] * item['quantity'] for item in st.session_state.cart)

def get_cart_item_count():
    """Obtiene la cantidad total de items en el carrito"""
    return sum(item['quantity'] for item in st.session_state.cart)

# ==================== FUNCIONES DE VALIDACIÓN ====================

def validate_email(email):
    """Valida el formato de un email"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_phone(phone):
    """Valida el formato de teléfono peruano"""
    # Eliminar espacios y caracteres especiales
    phone_clean = re.sub(r'[^\d]', '', phone)
    # Debe tener al menos 9 dígitos
    return len(phone_clean) >= 9

def validate_stock_availability(cart_items):
    """Valida que todos los productos en el carrito tengan stock disponible"""
    products = get_all_products()
    products_dict = {p['id']: p for p in products}

    for item in cart_items:
        product = products_dict.get(item['id'])
        if not product:
            return False, f"El producto {item['name']} ya no está disponible"
        if product['stock'] < item['quantity']:
            return False, f"Stock insuficiente para {item['name']}. Disponible: {product['stock']}"

    return True, "Stock disponible"

# ==================== FUNCIONES DE ORDEN ====================

def create_order(customer_name, customer_email, customer_phone, shipping_address, cart_items):
    """Crea una nueva orden en la base de datos"""
    try:
        # Preparar items en el formato correcto
        items_json = [
            {
                "name": item['name'],
                "quantity": item['quantity'],
                "price": item['price']
            }
            for item in cart_items
        ]

        # Calcular total
        total_amount = sum(item['price'] * item['quantity'] for item in cart_items)

        # Normalizar teléfono (eliminar caracteres especiales)
        phone_clean = re.sub(r'[^\d]', '', customer_phone)

        # Crear orden
        order_data = {
            "customer_name": customer_name,
            "customer_email": customer_email,
            "customer_phone": phone_clean,
            "shipping_address": shipping_address,
            "total_amount": float(total_amount),
            "status": "Pendiente",
            "items": items_json
        }

        response = supabase.table("orders").insert(order_data).execute()

        if response.data:
            return True, response.data[0]['id'], "✅ Pedido creado exitosamente"
        else:
            return False, None, "Error al crear el pedido"

    except Exception as e:
        return False, None, f"❌ Error al crear el pedido: {str(e)}"

# ==================== FUNCIONES DE UI ====================

def render_product_card(product):
    """Renderiza una card de producto usando componentes nativos de Streamlit"""

    # Determinar badge de stock
    stock = product['stock']
    if stock > 10:
        stock_text = "✓ Disponible"
        is_available = True
    elif stock > 0:
        stock_text = f"⚠️ Solo {stock} disponibles"
        is_available = True
    else:
        stock_text = "✗ Agotado"
        is_available = False

    # Descripción
    description = product.get('description', 'Sin descripción')
    if len(description) > 100:
        description = description[:100] + "..."

    # Usar container de Streamlit en lugar de HTML
    with st.container():
        # Mostrar imagen del producto con tamaño estandarizado
        image_url = product.get('image_url')
        if image_url:
            # Contenedor con altura fija para todas las imágenes usando CSS personalizado
            st.markdown(
                f"""
                <div class='product-image-container'>
                    <img src='{image_url}' alt='{product['name']}'>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            # Placeholder cuando no hay imagen con altura idéntica
            st.markdown(
                """
                <div class='product-image-container'>
                    <span style='font-size: 64px; opacity: 0.3;'>🖼️</span>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.write(f"### {product['name']}")
        st.write(f"*{description}*")
        st.write(f"**S/ {product['price']:.2f}**")
        st.caption(stock_text)

        # Input de cantidad y botón
        if is_available:
            col1, col2 = st.columns([2, 3])
            with col1:
                quantity = st.number_input(
                    "Cantidad",
                    min_value=1,
                    max_value=stock,
                    value=1,
                    key=f"qty_{product['id']}",
                    label_visibility="collapsed"
                )
            with col2:
                if st.button("🛒 Agregar", key=f"add_{product['id']}", use_container_width=True):
                    add_to_cart(product, quantity)
                    st.success(f"✅ {quantity}x {product['name']} agregado al carrito!")
                    st.rerun()
        else:
            st.button("❌ No Disponible", disabled=True, use_container_width=True)

def render_cart():
    """Renderiza el carrito de compras"""
    if not st.session_state.cart:
        st.info("🛒 Tu carrito está vacío. ¡Agrega productos desde el catálogo!")
        return

    st.subheader("🛍️ Productos en tu carrito")

    # Mostrar items del carrito
    for item in st.session_state.cart:
        with st.container():
            col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 2, 1])

            with col1:
                st.write(f"**{item['name']}**")

            with col2:
                st.write(f"S/ {item['price']:.2f}")

            with col3:
                new_qty = st.number_input(
                    "Cantidad",
                    min_value=1,
                    max_value=item['stock_available'],
                    value=item['quantity'],
                    key=f"cart_qty_{item['id']}",
                    label_visibility="collapsed"
                )
                if new_qty != item['quantity']:
                    update_cart_quantity(item['id'], new_qty)
                    st.rerun()

            with col4:
                subtotal = item['price'] * item['quantity']
                st.write(f"**S/ {subtotal:.2f}**")

            with col5:
                if st.button("🗑️", key=f"remove_{item['id']}"):
                    remove_from_cart(item['id'])
                    st.rerun()

            st.divider()

    # Total
    total = get_cart_total()
    st.write(f"### **Total: S/ {total:.2f}**")

    # Botones de acción
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Vaciar Carrito", use_container_width=True):
            clear_cart()
            st.rerun()

    st.divider()

    # Formulario de checkout
    render_checkout_form()

def render_checkout_form():
    """Renderiza el formulario de checkout"""
    st.subheader("📋 Datos de Entrega")

    with st.form("checkout_form"):
        customer_name = st.text_input(
            "Nombre Completo *",
            placeholder="Ej: Juan Pérez García"
        )

        col1, col2 = st.columns(2)
        with col1:
            customer_email = st.text_input(
                "Email *",
                placeholder="ejemplo@gmail.com"
            )
        with col2:
            customer_phone = st.text_input(
                "WhatsApp / Teléfono *",
                placeholder="921971743"
            )

        shipping_address = st.text_area(
            "Dirección de Entrega Completa *",
            placeholder="Ej: Av. Larco 123, Miraflores, Lima"
        )

        submitted = st.form_submit_button("✅ Confirmar Pedido", use_container_width=True, type="primary")

        if submitted:
            # Validaciones
            errors = []

            if not customer_name or len(customer_name.strip()) < 3:
                errors.append("❌ El nombre completo es obligatorio (mínimo 3 caracteres)")

            if not customer_email or not validate_email(customer_email):
                errors.append("❌ El email no es válido")

            if not customer_phone or not validate_phone(customer_phone):
                errors.append("❌ El teléfono debe tener al menos 9 dígitos")

            if not shipping_address or len(shipping_address.strip()) < 10:
                errors.append("❌ La dirección de entrega es obligatoria (mínimo 10 caracteres)")

            # Validar stock
            stock_valid, stock_message = validate_stock_availability(st.session_state.cart)
            if not stock_valid:
                errors.append(f"❌ {stock_message}")

            # Mostrar errores o procesar orden
            if errors:
                for error in errors:
                    st.error(error)
            else:
                # Crear orden
                success, order_id, message = create_order(
                    customer_name,
                    customer_email,
                    customer_phone,
                    shipping_address,
                    st.session_state.cart
                )

                if success:
                    st.session_state.order_completed = True
                    st.session_state.order_id = order_id
                    st.session_state.order_total = get_cart_total()
                    clear_cart()
                    st.rerun()
                else:
                    st.error(message)

def render_success_message():
    """Renderiza el mensaje de éxito después de crear una orden usando componentes nativos"""
    st.balloons()

    # Mensaje de éxito principal
    st.success("### 🎉 ¡Pedido Confirmado!")
    st.write("Tu pedido ha sido registrado exitosamente")

    st.divider()

    # Información del pedido en columnas
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="📦 Número de Pedido",
            value=f"#{st.session_state.order_id[:8]}"
        )

    with col2:
        st.metric(
            label="💰 Total Pagado",
            value=f"S/ {st.session_state.order_total:.2f}"
        )

    with col3:
        st.metric(
            label="📱 Estado",
            value="Pendiente"
        )

    st.divider()

    # Información adicional
    st.info("""
    ### 📲 ¡Revisa tu WhatsApp y Email!

    Te hemos enviado:
    - 📱 **WhatsApp**: Confirmación del pedido con todos los detalles
    - 📧 **Email**: Notificación al vendedor para procesar tu orden

    El vendedor revisará tu pedido y se pondrá en contacto contigo pronto.
    """)

    if st.button("🛍️ Realizar Otro Pedido", use_container_width=True, type="primary"):
        st.session_state.order_completed = False
        st.session_state.order_id = None
        st.session_state.order_total = 0
        st.rerun()

# ==================== PÁGINA PRINCIPAL ====================

def main():
    """Función principal de la tienda"""

    # Header con componentes nativos
    st.title("🛍️ Tienda Online")
    st.write("Encuentra los mejores productos al mejor precio")

    # Indicador de carrito
    cart_count = get_cart_item_count()
    if cart_count > 0:
        st.info(f"🛒 {cart_count} {'producto' if cart_count == 1 else 'productos'} en tu carrito")

    st.divider()

    # Si se completó una orden, mostrar mensaje de éxito
    if st.session_state.order_completed:
        render_success_message()
        return

    # Tabs principales
    tab1, tab2 = st.tabs(["🛍️ Catálogo de Productos", "🛒 Mi Carrito"])

    with tab1:
        st.subheader("Nuestros Productos")
        st.divider()

        # Obtener productos
        products = get_all_products()

        if products:
            # Mostrar productos en grid de 3 columnas
            cols_per_row = 3
            for i in range(0, len(products), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    if i + j < len(products):
                        with col:
                            render_product_card(products[i + j])
        else:
            st.info("No hay productos disponibles en este momento")

    with tab2:
        render_cart()

if __name__ == "__main__":
    main()
