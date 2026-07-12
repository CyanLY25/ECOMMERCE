"""
Navegación entre las 3 aplicaciones del sistema (cada una desplegada por
separado en Streamlit Community Cloud):
  - "prediccion": Predicción de Demanda E-Commerce (frontend/streamlit_app.py)
  - "vendedor":   Panel de Administración / Vendedor (admin_vendedor.py)
  - "tienda":     Tienda Cliente (tienda_cliente.py)

Como cada una vive en una URL distinta, la "navegación" se resuelve con
enlaces (st.link_button) hacia las otras dos, leyendo las URLs desde
`.streamlit/secrets.toml` bajo la sección [app_urls]. Si una URL no está
configurada, se muestra un texto indicándolo en vez de romper la app.
"""
import streamlit as st

APP_NAMES = {
    "prediccion": "📊 Predicción de Demanda",
    "vendedor": "🏪 Panel de Vendedor",
    "tienda": "🛍️ Tienda Cliente",
}


def _get_configured_urls() -> dict:
    try:
        return dict(st.secrets["app_urls"])
    except (KeyError, FileNotFoundError):
        return {}


def render_app_navigation(current_app: str, location=None) -> None:
    """
    Dibuja, en la barra lateral (por defecto), los enlaces hacia las otras
    aplicaciones del sistema. `current_app` debe ser una de las claves de
    APP_NAMES ("prediccion", "vendedor", "tienda").
    """
    location = location or st.sidebar
    urls = _get_configured_urls()

    location.divider()
    location.subheader("🔗 Sistema E-Commerce")

    for key, label in APP_NAMES.items():
        if key == current_app:
            location.caption(f"📍 {label} — estás aquí")
            continue

        url = urls.get(key)
        if url:
            location.link_button(label, url, use_container_width=True)
        else:
            location.caption(f"{label}: URL no configurada aún")
