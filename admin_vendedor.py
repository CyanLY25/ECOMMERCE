"""
🏪 PANEL DE ADMINISTRACIÓN - E-COMMERCE
Panel completo para vendedores con gestión de productos y órdenes
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import json
import plotly.express as px
import plotly.graph_objects as go
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from io import BytesIO
from PIL import Image
import uuid

import store_api

# ==================== CONFIGURACIÓN ====================

st.set_page_config(
    page_title="Panel Vendedor - E-Commerce",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== AUTENTICACIÓN ====================
# El dashboard completo (métricas, productos, órdenes) solo se
# construye/renderiza después de validar las credenciales del vendedor.
from auth_utils import require_login, logout_button

require_login(app_title="🏪 Panel de Administración - E-Commerce")
logout_button()

from app_links import render_app_navigation
render_app_navigation("vendedor")

# Sin estilos CSS personalizados - usamos colores nativos de Streamlit

# ==================== FUNCIONES DE PRODUCTOS / ÓRDENES ====================
# Antes hablaban directo con Supabase; ahora hablan por HTTP con nuestro
# propio backend (el mismo desplegado en Render para las predicciones),
# a través de store_api. Se mantienen los mismos nombres y firmas para no
# tocar el resto del archivo.

def get_all_products():
    """Obtiene todos los productos de la tienda"""
    return store_api.get_all_products()


def create_product(name, description, price, stock, image_url=None):
    """Crea un nuevo producto. `image_url` aquí es en realidad una data URI
    (ya convertida desde el archivo subido) o None."""
    return store_api.create_product(name, description, price, stock, image_data_uri=image_url)


def update_product(product_id, name, description, price, stock, image_url=None, remove_image=False):
    """Actualiza un producto existente."""
    return store_api.update_product(
        product_id, name, description, price, stock,
        image_data_uri=image_url, remove_image=remove_image
    )


def delete_product(product_id):
    """Elimina un producto"""
    return store_api.delete_product(product_id)


def get_all_orders():
    """Obtiene todas las órdenes"""
    return store_api.get_all_orders()


def update_order_status(order_id, new_status, items):
    """Actualiza el estado de una orden. El backend descuenta el stock
    automáticamente cuando el nuevo estado es 'Completado'."""
    return store_api.update_order_status(order_id, new_status)

# ==================== FUNCIONES DE MÉTRICAS ====================


def get_dashboard_metrics():
    """Calcula las métricas del dashboard"""
    products = get_all_products()
    orders = get_all_orders()

    total_products = len(products)
    total_orders = len(orders)
    pending_orders = len([o for o in orders if o["status"] == "Pendiente"])
    total_revenue = sum([o["total_amount"]
                        for o in orders if o["status"] == "Completado"])

    return {
        "total_products": total_products,
        "total_orders": total_orders,
        "pending_orders": pending_orders,
        "total_revenue": total_revenue
    }

# ==================== FUNCIONES DE GRÁFICOS ====================

def create_sales_by_day_chart(orders):
    """Crea gráfico de ventas por día"""
    if not orders or len(orders) == 0:
        return None

    try:
        # Filtrar solo órdenes completadas para el cálculo de ventas
        completed_orders = [o for o in orders if o.get('status') == 'Completado']

        if not completed_orders:
            return None

        # Crear DataFrame
        df = pd.DataFrame(completed_orders)

        # Convertir fechas
        df['date'] = pd.to_datetime(df['created_at']).dt.date

        # Agrupar por fecha y sumar ventas
        df_grouped = df.groupby('date')['total_amount'].sum().reset_index()
        df_grouped.columns = ['Fecha', 'Ventas']

        # Ordenar por fecha
        df_grouped = df_grouped.sort_values('Fecha')

        # Crear gráfico
        fig = px.line(
            df_grouped,
            x='Fecha',
            y='Ventas',
            title=f'Ventas Diarias (Órdenes Completadas)',
            labels={'Ventas': 'Ventas (S/)', 'Fecha': 'Fecha'},
            markers=True
        )

        fig.update_traces(
            line_color='#059669',
            line_width=3,
            marker=dict(size=8)
        )

        fig.update_layout(
            hovermode='x unified',
            height=400
        )

        return fig
    except Exception as e:
        st.error(f"Error al generar gráfico de ventas: {str(e)}")
        return None

def create_orders_by_status_chart(orders):
    """Crea gráfico de órdenes por estado"""
    if not orders or len(orders) == 0:
        return None

    try:
        # Contar manualmente los estados
        status_count = {'Pendiente': 0, 'Completado': 0, 'Cancelado': 0}

        for order in orders:
            status = order.get('status', 'Desconocido')
            if status in status_count:
                status_count[status] += 1

        # Filtrar estados con 0 órdenes y convertir a lista
        states = []
        counts = []
        for estado, cantidad in status_count.items():
            if cantidad > 0:
                states.append(estado)
                counts.append(cantidad)

        if not states:
            return None

        # Colores para cada estado
        colors_list = []
        colors_map = {
            'Pendiente': '#fbbf24',
            'Completado': '#10b981',
            'Cancelado': '#ef4444'
        }

        for estado in states:
            colors_list.append(colors_map.get(estado, '#999999'))

        # Crear gráfico circular usando go.Pie directamente
        fig = go.Figure(data=[go.Pie(
            labels=states,
            values=counts,
            hole=0.3,
            marker=dict(colors=colors_list),
            textposition='auto',
            texttemplate='%{label}<br>%{value}<br>(%{percent})',
            hovertemplate='<b>%{label}</b><br>Cantidad: %{value}<br>Porcentaje: %{percent}<extra></extra>'
        )])

        fig.update_layout(
            title=f'Distribución de Órdenes por Estado (Total: {len(orders)})',
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.1
            ),
            height=400
        )

        return fig
    except Exception as e:
        st.error(f"Error al generar gráfico de estados: {str(e)}")
        return None

def create_top_products_chart(orders):
    """Crea gráfico de productos más vendidos"""
    if not orders or len(orders) == 0:
        return None

    try:
        # Solo contar productos de órdenes completadas
        completed_orders = [o for o in orders if o.get('status') == 'Completado']

        products_sold = {}
        for order in completed_orders:
            if order.get('items'):
                for item in order['items']:
                    product_name = item.get('name', 'Desconocido')
                    quantity = item.get('quantity', 0)
                    if product_name in products_sold:
                        products_sold[product_name] += quantity
                    else:
                        products_sold[product_name] = quantity

        if not products_sold:
            return None

        # Convertir a lista y ordenar
        products_list = [{'Producto': name, 'Cantidad': qty} for name, qty in products_sold.items()]
        products_list.sort(key=lambda x: x['Cantidad'], reverse=True)

        # Tomar top 10
        top_products = products_list[:10]

        if not top_products:
            return None

        df = pd.DataFrame(top_products)

        # Crear gráfico horizontal
        fig = px.bar(
            df,
            x='Cantidad',
            y='Producto',
            title='Top 10 Productos Más Vendidos',
            orientation='h',
            labels={'Cantidad': 'Cantidad Vendida', 'Producto': 'Producto'},
            text='Cantidad'
        )

        fig.update_traces(
            marker_color='#059669',
            texttemplate='%{text}',
            textposition='outside'
        )

        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            height=400
        )

        return fig
    except Exception as e:
        st.error(f"Error al generar gráfico de productos: {str(e)}")
        return None

def create_revenue_by_month_chart(orders):
    """Crea gráfico de ingresos por mes"""
    if not orders or len(orders) == 0:
        return None

    try:
        # Filtrar solo órdenes completadas
        completed_orders = [o for o in orders if o.get('status') == 'Completado']

        if not completed_orders:
            return None

        # Crear DataFrame
        df = pd.DataFrame(completed_orders)

        # Convertir fechas y extraer mes
        df['date'] = pd.to_datetime(df['created_at'])
        df['month'] = df['date'].dt.to_period('M')

        # Agrupar por mes
        df_grouped = df.groupby('month')['total_amount'].sum().reset_index()
        df_grouped['month_str'] = df_grouped['month'].astype(str)
        df_grouped = df_grouped.sort_values('month')

        # Crear gráfico de barras
        fig = px.bar(
            df_grouped,
            x='month_str',
            y='total_amount',
            title='Ingresos Mensuales (Órdenes Completadas)',
            labels={'total_amount': 'Ingresos (S/)', 'month_str': 'Mes'},
            text='total_amount'
        )

        fig.update_traces(
            marker_color='#047857',
            texttemplate='S/ %{text:.2f}',
            textposition='outside'
        )

        fig.update_layout(
            height=400,
            xaxis_tickangle=-45
        )

        return fig
    except Exception as e:
        st.error(f"Error al generar gráfico de ingresos: {str(e)}")
        return None

def create_orders_trend_chart(orders):
    """Crea gráfico de tendencia de órdenes a lo largo del tiempo"""
    if not orders or len(orders) == 0:
        return None

    try:
        # Preparar datos por fecha y estado
        orders_by_date = {}
        for order in orders:
            try:
                date = pd.to_datetime(order['created_at']).date()
                status = order.get('status', 'Desconocido')

                if date not in orders_by_date:
                    orders_by_date[date] = {'Pendiente': 0, 'Completado': 0, 'Cancelado': 0}

                if status in orders_by_date[date]:
                    orders_by_date[date][status] += 1
            except:
                continue

        if not orders_by_date:
            return None

        # Convertir a lista para DataFrame
        trend_data = []
        for date, statuses in sorted(orders_by_date.items()):
            for status, count in statuses.items():
                if count > 0:  # Solo incluir si hay órdenes
                    trend_data.append({
                        'Fecha': date,
                        'Estado': status,
                        'Cantidad': count
                    })

        if not trend_data:
            return None

        df = pd.DataFrame(trend_data)

        # Colores para cada estado
        colors_map = {
            'Pendiente': '#fbbf24',
            'Completado': '#10b981',
            'Cancelado': '#ef4444'
        }

        # Crear gráfico de líneas por estado
        fig = px.line(
            df,
            x='Fecha',
            y='Cantidad',
            color='Estado',
            title='Tendencia de Órdenes por Estado',
            labels={'Cantidad': 'Número de Órdenes', 'Fecha': 'Fecha'},
            color_discrete_map=colors_map,
            markers=True
        )

        fig.update_traces(line_width=3, marker=dict(size=8))

        fig.update_layout(
            hovermode='x unified',
            height=400,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.3,
                xanchor="center",
                x=0.5
            )
        )

        return fig
    except Exception as e:
        st.error(f"Error al generar gráfico de tendencias: {str(e)}")
        return None

# ==================== FUNCIONES DE EXPORTACIÓN ====================

def generate_order_pdf(order, items):
    """Genera un PDF de una orden individual"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()

    # Título
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#059669'),
        spaceAfter=30,
        alignment=1  # Centrado
    )
    elements.append(Paragraph("ORDEN DE COMPRA", title_style))
    elements.append(Spacer(1, 0.3*inch))

    # Información de la orden
    order_info_style = styles['Normal']
    date = datetime.fromisoformat(order['created_at'].replace('Z', '+00:00'))

    elements.append(Paragraph(f"<b>Número de Orden:</b> {order['id'][:13]}", order_info_style))
    elements.append(Spacer(1, 0.1*inch))
    elements.append(Paragraph(f"<b>Fecha:</b> {date.strftime('%d/%m/%Y %H:%M')}", order_info_style))
    elements.append(Spacer(1, 0.1*inch))
    elements.append(Paragraph(f"<b>Estado:</b> {order['status']}", order_info_style))
    elements.append(Spacer(1, 0.3*inch))

    # Información del cliente
    elements.append(Paragraph("<b>DATOS DEL CLIENTE</b>", styles['Heading2']))
    elements.append(Spacer(1, 0.1*inch))
    elements.append(Paragraph(f"<b>Nombre:</b> {order['customer_name']}", order_info_style))
    elements.append(Spacer(1, 0.1*inch))
    elements.append(Paragraph(f"<b>Email:</b> {order.get('customer_email', 'No proporcionado')}", order_info_style))
    elements.append(Spacer(1, 0.1*inch))
    elements.append(Paragraph(f"<b>Teléfono:</b> {order.get('customer_phone', 'No proporcionado')}", order_info_style))
    elements.append(Spacer(1, 0.1*inch))
    elements.append(Paragraph(f"<b>Dirección:</b> {order.get('shipping_address', 'No proporcionada')}", order_info_style))
    elements.append(Spacer(1, 0.3*inch))

    # Tabla de productos
    elements.append(Paragraph("<b>PRODUCTOS</b>", styles['Heading2']))
    elements.append(Spacer(1, 0.1*inch))

    table_data = [['Producto', 'Cantidad', 'Precio Unit.', 'Subtotal']]
    for item in items:
        subtotal = item['price'] * item['quantity']
        table_data.append([
            item['name'],
            str(item['quantity']),
            f"S/ {item['price']:.2f}",
            f"S/ {subtotal:.2f}"
        ])

    table = Table(table_data, colWidths=[3*inch, 1*inch, 1.2*inch, 1.2*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#059669')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(table)
    elements.append(Spacer(1, 0.3*inch))

    # Total
    total_style = ParagraphStyle(
        'Total',
        parent=styles['Normal'],
        fontSize=16,
        textColor=colors.HexColor('#059669'),
        alignment=2  # Derecha
    )
    elements.append(Paragraph(f"<b>TOTAL: S/ {order['total_amount']:.2f}</b>", total_style))

    doc.build(elements)
    buffer.seek(0)
    return buffer

def generate_executive_summary_excel(orders, start_date, end_date):
    """Genera un resumen ejecutivo en Excel"""
    if not orders:
        return None

    df = pd.DataFrame(orders)
    df['created_at'] = pd.to_datetime(df['created_at'])

    # Aplicar filtro de fechas
    if start_date:
        df = df[df['created_at'].dt.date >= start_date]
    if end_date:
        df = df[df['created_at'].dt.date <= end_date]

    if df.empty:
        return None

    # Crear resumen
    summary_data = []
    for _, order in df.iterrows():
        items_count = len(order['items']) if order.get('items') else 0
        summary_data.append({
            'Orden ID': order['id'][:13],
            'Fecha': order['created_at'].strftime('%d/%m/%Y %H:%M'),
            'Cliente': order['customer_name'],
            'Teléfono': order.get('customer_phone', 'N/A'),
            'Estado': order['status'],
            'Items': items_count,
            'Total (S/)': f"{order['total_amount']:.2f}"
        })

    summary_df = pd.DataFrame(summary_data)

    # Guardar en buffer
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        summary_df.to_excel(writer, index=False, sheet_name='Resumen de Órdenes')

        # Agregar hoja de estadísticas
        stats_data = {
            'Métrica': [
                'Total de Órdenes',
                'Órdenes Pendientes',
                'Órdenes Completadas',
                'Órdenes Canceladas',
                'Ingresos Totales (Completadas)',
                'Ticket Promedio'
            ],
            'Valor': [
                len(df),
                len(df[df['status'] == 'Pendiente']),
                len(df[df['status'] == 'Completado']),
                len(df[df['status'] == 'Cancelado']),
                f"S/ {df[df['status'] == 'Completado']['total_amount'].sum():.2f}",
                f"S/ {df['total_amount'].mean():.2f}"
            ]
        }
        stats_df = pd.DataFrame(stats_data)
        stats_df.to_excel(writer, index=False, sheet_name='Estadísticas')

    buffer.seek(0)
    return buffer

# ==================== FUNCIONES DE UI ====================


def render_stock_badge(stock):
    """Renderiza un badge de stock con color según disponibilidad"""
    if stock > 10:
        return "🟢 Stock: " + str(stock)
    elif stock > 0:
        return "🟡 Stock: " + str(stock)
    else:
        return "🔴 Agotado"


def render_status_text(status):
    """Renderiza el texto de estado con emoji"""
    status_lower = status.lower()
    if status_lower == "pendiente":
        return f"⏳ {status}"
    elif status_lower == "completado":
        return f"✅ {status}"
    elif status_lower == "cancelado":
        return f"❌ {status}"
    return status

# ==================== PÁGINA: DASHBOARD ====================


def page_dashboard():
    """Página principal con métricas y gráficos"""
    st.title("📊 Dashboard")
    st.divider()

    # Obtener datos
    metrics = get_dashboard_metrics()
    all_orders = get_all_orders()

    # Mostrar métricas en columnas con st.metric (nativo de Streamlit)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="📦 Total Productos",
            value=metrics['total_products']
        )

    with col2:
        st.metric(
            label="🛒 Total Órdenes",
            value=metrics['total_orders']
        )

    with col3:
        st.metric(
            label="⏳ Órdenes Pendientes",
            value=metrics['pending_orders']
        )

    with col4:
        st.metric(
            label="💰 Ingresos Totales",
            value=f"S/ {metrics['total_revenue']:.2f}"
        )

    st.divider()

    # SECCIÓN DE GRÁFICOS
    if all_orders:
        st.subheader("📈 Análisis de Ventas y Órdenes")

        # Primera fila: Ventas y Estados
        col1, col2 = st.columns(2)

        with col1:
            # Gráfico de ventas por día
            chart_sales = create_sales_by_day_chart(all_orders)
            if chart_sales:
                st.plotly_chart(chart_sales, use_container_width=True)
            else:
                st.info("No hay ventas completadas para mostrar")

        with col2:
            # Gráfico de órdenes por estado
            chart_status = create_orders_by_status_chart(all_orders)
            if chart_status:
                st.plotly_chart(chart_status, use_container_width=True)
            else:
                st.info("No hay órdenes para mostrar distribución")

        # Segunda fila: Productos y Tendencias
        col1, col2 = st.columns(2)

        with col1:
            # Gráfico de productos más vendidos
            chart_products = create_top_products_chart(all_orders)
            if chart_products:
                st.plotly_chart(chart_products, use_container_width=True)
            else:
                st.info("No hay productos vendidos para mostrar")

        with col2:
            # Gráfico de tendencias de órdenes
            chart_trend = create_orders_trend_chart(all_orders)
            if chart_trend:
                st.plotly_chart(chart_trend, use_container_width=True)
            else:
                st.info("No hay datos suficientes para mostrar tendencias")

        # Tercera fila: Ingresos mensuales (ancho completo)
        st.write("")  # Espaciado
        chart_revenue = create_revenue_by_month_chart(all_orders)
        if chart_revenue:
            st.plotly_chart(chart_revenue, use_container_width=True)
        else:
            st.info("No hay ingresos mensuales para mostrar")

        st.divider()
    else:
        st.info("📊 No hay órdenes para mostrar gráficos. Las estadísticas aparecerán cuando recibas tus primeras órdenes.")

    # Órdenes recientes
    st.subheader("📋 Órdenes Recientes")
    orders = all_orders[:5] if all_orders else []  # Últimas 5 órdenes

    if orders:
        for order in orders:
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
                with col1:
                    st.write(f"**{order['customer_name']}**")
                with col2:
                    st.write(render_status_text(order['status']))
                with col3:
                    st.write(f"**S/ {order['total_amount']:.2f}**")
                with col4:
                    date = datetime.fromisoformat(
                        order['created_at'].replace('Z', '+00:00'))
                    st.write(f"*{date.strftime('%d/%m/%Y %H:%M')}*")
                st.divider()
    else:
        st.info("No hay órdenes aún")

# ==================== PÁGINA: PRODUCTOS ====================


def page_products():
    """Página de gestión de productos"""
    st.title("📦 Gestión de Productos")
    st.divider()

    # Tabs para separar lista y creación
    tab1, tab2 = st.tabs(["📋 Lista de Productos", "➕ Nuevo Producto"])

    with tab1:
        products = get_all_products()

        if products:
            st.subheader(f"Total: {len(products)} productos")

            for product in products:
                with st.container():
                    col1, col2, col3, col4, col5 = st.columns([3, 3, 2, 2, 2])

                    with col1:
                        # Mostrar imagen si existe
                        image_url = product.get('image_url')
                        if image_url:
                            st.image(image_url, width=80)
                        st.write(f"**{product['name']}**")
                        if product.get('stock_code'):
                            st.caption(f"🔖 {product['stock_code']}")

                    with col2:
                        desc = product.get('description', '')
                        if desc and len(desc) > 40:
                            desc = desc[:40] + "..."
                        st.write(f"*{desc}*")

                    with col3:
                        st.write(f"**S/ {product['price']:.2f}**")

                    with col4:
                        st.write(render_stock_badge(product['stock']))

                    with col5:
                        col_edit, col_delete = st.columns(2)
                        with col_edit:
                            if st.button("✏️", key=f"edit_{product['id']}"):
                                st.session_state.editing_product = product
                                st.rerun()
                        with col_delete:
                            if st.button("🗑️", key=f"delete_{product['id']}"):
                                st.session_state.deleting_product = product
                                st.rerun()

                    st.divider()
        else:
            st.info("No hay productos registrados. ¡Crea el primero!")

    with tab2:
        st.subheader("Crear Nuevo Producto")

        # File uploader fuera del form (Streamlit no permite file_uploader dentro de forms)
        uploaded_image = st.file_uploader(
            "📷 Imagen del Producto (opcional)",
            type=["jpg", "jpeg", "png", "webp"],
            help="Sube una imagen del producto. Formatos soportados: JPG, PNG, WEBP"
        )

        # Vista previa de la imagen
        if uploaded_image:
            col_preview1, col_preview2 = st.columns([1, 3])
            with col_preview1:
                st.image(uploaded_image, caption="Vista previa", width=150)
            with col_preview2:
                st.info("📸 La imagen se subirá cuando crees el producto")

        with st.form("form_create_product"):
            name = st.text_input("Nombre del Producto *",
                                 placeholder="Ej: Polo Básico Blanco")
            description = st.text_area(
                "Descripción", placeholder="Describe el producto...")
            col1, col2 = st.columns(2)
            with col1:
                price = st.number_input(
                    "Precio (S/) *", min_value=0.0, step=0.5, format="%.2f")
            with col2:
                stock = st.number_input("Stock *", min_value=0, step=1)

            submitted = st.form_submit_button(
                "✅ Crear Producto", use_container_width=True)

            if submitted:
                if not name:
                    st.error("❌ El nombre del producto es obligatorio")
                elif price <= 0:
                    st.error("❌ El precio debe ser mayor a 0")
                elif uploaded_image and uploaded_image.size > 3 * 1024 * 1024:
                    st.error("❌ La imagen no debe superar 3 MB")
                else:
                    # Convertir la imagen a data URI (se guarda directo en
                    # nuestro backend, sin depender de un servicio externo)
                    image_url = store_api.file_to_data_uri(uploaded_image) if uploaded_image else None

                    success, message = create_product(
                        name, description, price, stock, image_url)
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)

    # Modal de edición
    if 'editing_product' in st.session_state:
        product = st.session_state.editing_product

        st.divider()
        st.subheader(f"✏️ Editar: {product['name']}")

        # Mostrar imagen actual si existe
        current_image_url = product.get('image_url')
        if current_image_url:
            col_img1, col_img2 = st.columns([1, 3])
            with col_img1:
                st.write("**Imagen Actual:**")
                st.image(current_image_url, width=150)
            with col_img2:
                st.info("📸 Esta es la imagen actual del producto")

        # File uploader para nueva imagen
        new_uploaded_image = st.file_uploader(
            "📷 Cambiar Imagen del Producto (opcional)",
            type=["jpg", "jpeg", "png", "webp"],
            help="Sube una nueva imagen para reemplazar la actual",
            key="edit_image_uploader"
        )

        # Vista previa de nueva imagen
        if new_uploaded_image:
            col_img1, col_img2 = st.columns([1, 3])
            with col_img1:
                st.write("**Nueva Imagen:**")
                st.image(new_uploaded_image, caption="Vista previa", width=150)
            with col_img2:
                st.warning("⚠️ La imagen anterior será eliminada al guardar")

        remove_current_image = False
        if current_image_url and not new_uploaded_image:
            remove_current_image = st.checkbox("🗑️ Quitar la imagen actual (dejar el producto sin imagen)")

        with st.form("form_edit_product"):
            name = st.text_input("Nombre del Producto *",
                                 value=product['name'])
            description = st.text_area(
                "Descripción", value=product.get('description', ''))
            col1, col2 = st.columns(2)
            with col1:
                price = st.number_input(
                    "Precio (S/) *", min_value=0.0, step=0.5, value=float(product['price']), format="%.2f")
            with col2:
                stock = st.number_input(
                    "Stock *", min_value=0, step=1, value=product['stock'])

            col_save, col_cancel = st.columns(2)
            with col_save:
                submitted = st.form_submit_button(
                    "💾 Guardar Cambios", use_container_width=True)
            with col_cancel:
                cancel = st.form_submit_button(
                    "❌ Cancelar", use_container_width=True)

            if submitted:
                if not name:
                    st.error("❌ El nombre del producto es obligatorio")
                elif price <= 0:
                    st.error("❌ El precio debe ser mayor a 0")
                elif new_uploaded_image and new_uploaded_image.size > 3 * 1024 * 1024:
                    st.error("❌ La imagen no debe superar 3 MB")
                else:
                    # Convertir la nueva imagen (si hay) a data URI; si no hay
                    # imagen nueva y no se pidió quitarla, se mantiene la actual.
                    image_url_to_update = store_api.file_to_data_uri(new_uploaded_image) if new_uploaded_image else None

                    success, message = update_product(
                        product['id'], name, description, price, stock,
                        image_url=image_url_to_update, remove_image=remove_current_image)
                    if success:
                        st.success(message)
                        del st.session_state.editing_product
                        st.rerun()
                    else:
                        st.error(message)

            if cancel:
                del st.session_state.editing_product
                st.rerun()

    # Modal de eliminación
    if 'deleting_product' in st.session_state:
        product = st.session_state.deleting_product

        st.divider()
        st.warning(
            f"⚠️ ¿Estás seguro de eliminar el producto **{product['name']}**?")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Sí, Eliminar", use_container_width=True, type="primary"):
                success, message = delete_product(product['id'])
                if success:
                    st.success(message)
                    del st.session_state.deleting_product
                    st.rerun()
                else:
                    st.error(message)
        with col2:
            if st.button("❌ Cancelar", use_container_width=True):
                del st.session_state.deleting_product
                st.rerun()

# ==================== PÁGINA: ÓRDENES ====================


def page_orders():
    """Página de gestión de órdenes"""
    st.title("🛒 Gestión de Órdenes")
    st.divider()

    # Filtros
    col1, col2 = st.columns([3, 1])
    with col1:
        filter_status = st.selectbox(
            "Filtrar por Estado",
            ["Todos", "Pendiente", "Completado", "Cancelado"]
        )

    # Obtener órdenes
    orders = get_all_orders()

    # Aplicar filtro
    if filter_status != "Todos":
        orders = [o for o in orders if o["status"] == filter_status]

    st.subheader(f"Total: {len(orders)} órdenes")
    st.divider()

    if orders:
        for order in orders:
            status_text = render_status_text(order['status'])
            with st.expander(f"📦 Orden #{order['id'][:8]}... - {order['customer_name']} - {status_text}", expanded=False):

                # Información del cliente
                st.subheader("👤 Datos del Cliente")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Nombre:** {order['customer_name']}")
                    st.write(
                        f"**Email:** {order.get('customer_email', 'No proporcionado')}")
                with col2:
                    st.write(
                        f"**Teléfono:** {order.get('customer_phone', 'No proporcionado')}")
                    date = datetime.fromisoformat(
                        order['created_at'].replace('Z', '+00:00'))
                    st.write(f"**Fecha:** {date.strftime('%d/%m/%Y %H:%M')}")

                st.write(
                    f"**Dirección de Envío:** {order.get('shipping_address', 'No proporcionada')}")

                st.divider()

                # Items del pedido
                st.subheader("🛍️ Productos del Pedido")
                items = order.get('items', [])

                if items:
                    # Crear tabla de productos
                    df_items = pd.DataFrame(items)
                    df_items['subtotal'] = df_items['price'] * \
                        df_items['quantity']
                    df_items['price'] = df_items['price'].apply(
                        lambda x: f"S/ {x:.2f}")
                    df_items['subtotal'] = df_items['subtotal'].apply(
                        lambda x: f"S/ {x:.2f}")
                    df_items.columns = ['Producto',
                                        'Cantidad', 'Precio Unit.', 'Subtotal']

                    st.dataframe(
                        df_items, use_container_width=True, hide_index=True)

                    st.write(f"### **Total: S/ {order['total_amount']:.2f}**")
                else:
                    st.info("No hay items en este pedido")

                st.divider()

                # Acciones de estado
                st.subheader("🔄 Cambiar Estado")
                col1, col2, col3 = st.columns(3)

                with col1:
                    if order['status'] != "Completado":
                        if st.button("✅ Marcar como Completado", key=f"complete_{order['id']}", use_container_width=True):
                            success, message = update_order_status(
                                order['id'], "Completado", items)
                            if success:
                                st.success(message)
                                st.info(
                                    "📦 El stock de los productos ha sido actualizado")
                                st.rerun()
                            else:
                                st.error(message)

                with col2:
                    if order['status'] != "Cancelado":
                        if st.button("❌ Cancelar Orden", key=f"cancel_{order['id']}", use_container_width=True):
                            success, message = update_order_status(
                                order['id'], "Cancelado", items)
                            if success:
                                st.success(message)
                                st.rerun()
                            else:
                                st.error(message)

                with col3:
                    if order['status'] != "Pendiente":
                        if st.button("⏳ Marcar como Pendiente", key=f"pending_{order['id']}", use_container_width=True):
                            success, message = update_order_status(
                                order['id'], "Pendiente", items)
                            if success:
                                st.success(message)
                                st.rerun()
                            else:
                                st.error(message)

                st.write(
                    f"**Estado actual:** {render_status_text(order['status'])}")

                # Botones de exportación (solo si está completada)
                if order['status'] == "Completado":
                    st.divider()
                    st.subheader("📥 Exportar Orden")

                    col_pdf, col_excel = st.columns(2)

                    with col_pdf:
                        if st.button("📄 Generar PDF", key=f"gen_pdf_{order['id']}", use_container_width=True, type="primary"):
                            pdf_buffer = generate_order_pdf(order, items)
                            st.download_button(
                                label="⬇️ Descargar PDF",
                                data=pdf_buffer,
                                file_name=f"orden_{order['id'][:8]}.pdf",
                                mime="application/pdf",
                                key=f"download_pdf_{order['id']}",
                                use_container_width=True
                            )

                    with col_excel:
                        if st.button("📊 Generar Excel", key=f"gen_excel_{order['id']}", use_container_width=True, type="secondary"):
                            # Crear un resumen ejecutivo de solo esta orden
                            order_date = datetime.fromisoformat(order['created_at'].replace('Z', '+00:00')).date()
                            excel_buffer = generate_executive_summary_excel([order], order_date, order_date)
                            if excel_buffer:
                                st.download_button(
                                    label="⬇️ Descargar Excel",
                                    data=excel_buffer,
                                    file_name=f"orden_{order['id'][:8]}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    key=f"download_excel_{order['id']}",
                                    use_container_width=True
                                )
    else:
        st.info("No hay órdenes con el filtro seleccionado")

    # RESUMEN EJECUTIVO
    st.divider()
    st.subheader("📊 Resumen Ejecutivo")

    with st.expander("🔍 Generar Resumen de Órdenes", expanded=False):
        st.write("Filtra órdenes por rango de fechas y exporta un resumen ejecutivo en Excel")

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Fecha Inicio",
                value=datetime.now() - timedelta(days=30),
                max_value=datetime.now()
            )
        with col2:
            end_date = st.date_input(
                "Fecha Fin",
                value=datetime.now(),
                max_value=datetime.now()
            )

        if st.button("📊 Generar Resumen Ejecutivo", use_container_width=True, type="primary"):
            excel_buffer = generate_executive_summary_excel(orders, start_date, end_date)

            if excel_buffer:
                st.success("✅ Resumen generado exitosamente")
                st.download_button(
                    label="⬇️ Descargar Resumen en Excel",
                    data=excel_buffer,
                    file_name=f"resumen_ejecutivo_{start_date}_a_{end_date}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.warning("No hay órdenes en el rango de fechas seleccionado")

# ==================== NAVEGACIÓN ====================


def main():
    """Función principal con navegación"""

    # Sidebar
    with st.sidebar:
        st.title("🏪 Panel Vendedor")
        st.divider()

        page = st.radio(
            "Navegación",
            ["📊 Dashboard", "📦 Productos", "🛒 Órdenes"],
            label_visibility="collapsed"
        )

        st.divider()
        st.subheader("ℹ️ Información")
        st.write("**Sistema E-Commerce**")
        st.write("Versión 1.0")
        st.write("Powered by Streamlit + FastAPI")

    # Renderizar página seleccionada
    if page == "📊 Dashboard":
        page_dashboard()
    elif page == "📦 Productos":
        page_products()
    elif page == "🛒 Órdenes":
        page_orders()


if __name__ == "__main__":
    main()
