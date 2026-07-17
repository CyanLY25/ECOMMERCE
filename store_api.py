"""
Cliente de la tienda contra el backend FastAPI propio (reemplaza a Supabase).

Tanto `admin_vendedor.py` como `tienda_cliente.py` importan este módulo.
Cada uno se despliega por separado en Streamlit Community Cloud, pero como
ambos hablan HTTP con el mismo backend (el mismo que ya expone
`/api/predict`, desplegado en Render), ambas apps ven siempre los mismos
productos y órdenes.

La URL del backend se lee de `st.secrets["backend"]["api_url"]` si existe;
si no, se usa `DEFAULT_API_URL` como respaldo, así la app funciona "out of
the box" apenas se despliega, sin configuración adicional obligatoria.
"""
from __future__ import annotations

import base64
from typing import Any, Optional

import requests
import streamlit as st

DEFAULT_API_URL = "https://ecommerce-9u1d.onrender.com"
_TIMEOUT = 20  # segundos; el backend en Render puede tardar en "despertar"


def _api_base() -> str:
    try:
        return str(st.secrets["backend"]["api_url"]).rstrip("/")
    except Exception:
        return DEFAULT_API_URL


def _handle_error(resp: requests.Response) -> str:
    try:
        detail = resp.json().get("detail")
    except Exception:
        detail = None
    return detail or f"Error HTTP {resp.status_code}"


def file_to_data_uri(uploaded_file) -> str:
    """Convierte un archivo subido por st.file_uploader en una data URI base64."""
    uploaded_file.seek(0)
    raw = uploaded_file.read()
    mime = uploaded_file.type or "image/png"
    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime};base64,{b64}"


# ==================== PRODUCTOS ====================

def get_all_products() -> list[dict[str, Any]]:
    """Obtiene todos los productos de la tienda."""
    try:
        resp = requests.get(f"{_api_base()}/api/store/products", timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp.json().get("products", [])
    except Exception as e:
        st.error(f"Error al obtener productos: {e}")
        return []


def create_product(
    name: str,
    description: str,
    price: float,
    stock: int,
    image_data_uri: Optional[str] = None,
) -> tuple[bool, str]:
    """Crea un nuevo producto."""
    try:
        payload = {
            "name": name,
            "description": description,
            "price": float(price),
            "stock": int(stock),
        }
        if image_data_uri:
            payload["image_base64"] = image_data_uri
        resp = requests.post(f"{_api_base()}/api/store/products", json=payload, timeout=_TIMEOUT)
        if resp.status_code == 201:
            return True, "✅ Producto creado exitosamente"
        return False, f"❌ Error al crear producto: {_handle_error(resp)}"
    except Exception as e:
        return False, f"❌ Error al crear producto: {e}"


def update_product(
    product_id: str,
    name: str,
    description: str,
    price: float,
    stock: int,
    image_data_uri: Optional[str] = None,
    remove_image: bool = False,
) -> tuple[bool, str]:
    """Actualiza un producto existente."""
    try:
        payload = {
            "name": name,
            "description": description,
            "price": float(price),
            "stock": int(stock),
            "remove_image": remove_image,
        }
        if image_data_uri:
            payload["image_base64"] = image_data_uri
        resp = requests.put(f"{_api_base()}/api/store/products/{product_id}", json=payload, timeout=_TIMEOUT)
        if resp.status_code == 200:
            return True, "✅ Producto actualizado exitosamente"
        return False, f"❌ Error al actualizar producto: {_handle_error(resp)}"
    except Exception as e:
        return False, f"❌ Error al actualizar producto: {e}"


def delete_product(product_id: str) -> tuple[bool, str]:
    """Elimina un producto."""
    try:
        resp = requests.delete(f"{_api_base()}/api/store/products/{product_id}", timeout=_TIMEOUT)
        if resp.status_code == 200:
            return True, "✅ Producto eliminado exitosamente"
        return False, f"❌ Error al eliminar producto: {_handle_error(resp)}"
    except Exception as e:
        return False, f"❌ Error al eliminar producto: {e}"


# ==================== ÓRDENES ====================

def get_all_orders() -> list[dict[str, Any]]:
    """Obtiene todas las órdenes."""
    try:
        resp = requests.get(f"{_api_base()}/api/store/orders", timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp.json().get("orders", [])
    except Exception as e:
        st.error(f"Error al obtener órdenes: {e}")
        return []


def create_order(
    customer_name: str,
    customer_email: str,
    customer_phone: str,
    shipping_address: str,
    items: list[dict[str, Any]],
) -> tuple[bool, Optional[str], str]:
    """Crea una nueva orden. `items` es una lista de {name, quantity, price}."""
    try:
        payload = {
            "customer_name": customer_name,
            "customer_email": customer_email,
            "customer_phone": customer_phone,
            "shipping_address": shipping_address,
            "items": [
                {"name": i["name"], "quantity": int(i["quantity"]), "price": float(i["price"])}
                for i in items
            ],
        }
        resp = requests.post(f"{_api_base()}/api/store/orders", json=payload, timeout=_TIMEOUT)
        if resp.status_code == 201:
            data = resp.json()
            return True, data["id"], "✅ Pedido creado exitosamente"
        return False, None, f"❌ Error al crear el pedido: {_handle_error(resp)}"
    except Exception as e:
        return False, None, f"❌ Error al crear el pedido: {e}"


def update_order_status(order_id: str, new_status: str) -> tuple[bool, str]:
    """Actualiza el estado de una orden (el backend descuenta el stock si corresponde)."""
    try:
        resp = requests.put(
            f"{_api_base()}/api/store/orders/{order_id}/status",
            json={"status": new_status},
            timeout=_TIMEOUT,
        )
        if resp.status_code == 200:
            return True, f"✅ Orden marcada como {new_status}"
        return False, f"❌ Error al actualizar orden: {_handle_error(resp)}"
    except Exception as e:
        return False, f"❌ Error al actualizar orden: {e}"
