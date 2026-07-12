"""
Autenticación simple para el Panel de Vendedor (admin_vendedor.py).

Diseño:
- Un único usuario "vendedor" (o los que se agreguen) definidos en
  `.streamlit/secrets.toml`, con la contraseña guardada como hash
  PBKDF2-HMAC-SHA256 (nunca en texto plano).
- No requiere una tabla nueva en Supabase ni dependencias externas
  (usa `hashlib`, de la librería estándar de Python).
- Bloquea con `st.stop()` la ejecución del resto del dashboard hasta
  que las credenciales sean válidas, cumpliendo "dashboard al cargar
  después de validar las credenciales".

Cómo generar tus credenciales: correr `generar_credenciales_vendedor.py`
y copiar el resultado a `.streamlit/secrets.toml`.
"""
import hashlib
import hmac

import streamlit as st

SESSION_KEY = "vendedor_autenticado"
SESSION_USER_KEY = "vendedor_usuario"
PBKDF2_ITERATIONS = 100_000


def hash_password(password: str, salt: str) -> str:
    """Genera el hash PBKDF2-HMAC-SHA256 de una contraseña dada una sal."""
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), PBKDF2_ITERATIONS
    ).hex()


def _get_configured_users() -> dict:
    """
    Lee los usuarios configurados en `.streamlit/secrets.toml`, bajo la
    sección [vendedor]. Soporta un único usuario (username/password_hash/salt)
    o varios usuarios bajo [vendedor.usuarios.<nombre>].
    """
    try:
        vendedor_secrets = st.secrets["vendedor"]
    except (KeyError, FileNotFoundError):
        return {}

    users = {}

    # Formato simple: un solo usuario
    if "username" in vendedor_secrets and "password_hash" in vendedor_secrets:
        users[vendedor_secrets["username"]] = {
            "password_hash": vendedor_secrets["password_hash"],
            "salt": vendedor_secrets["salt"],
        }

    # Formato múltiple: [vendedor.usuarios.<nombre>]
    if "usuarios" in vendedor_secrets:
        for username, creds in vendedor_secrets["usuarios"].items():
            users[username] = {
                "password_hash": creds["password_hash"],
                "salt": creds["salt"],
            }

    return users


def _validate_credentials(username: str, password: str) -> bool:
    users = _get_configured_users()
    if not users or username not in users:
        # Comparación dummy para no filtrar por timing si el usuario no existe
        hash_password(password, "salt_dummy_no_existe")
        return False

    creds = users[username]
    entered_hash = hash_password(password, creds["salt"])
    return hmac.compare_digest(entered_hash, creds["password_hash"])


def require_login(app_title: str = "🏪 Panel de Vendedor") -> bool:
    """
    Muestra el formulario de login si el usuario aún no se ha autenticado
    en esta sesión de Streamlit. Debe llamarse al inicio del script,
    inmediatamente después de `st.set_page_config(...)`.

    Devuelve True si ya está autenticado (el resto del script puede
    continuar). Si no lo está, dibuja el formulario y hace `st.stop()`
    para impedir que se ejecute el resto del dashboard.
    """
    if st.session_state.get(SESSION_KEY):
        return True

    st.title(app_title)
    st.subheader("🔒 Iniciar sesión")
    st.caption("Ingresa tus credenciales de vendedor para acceder al dashboard.")

    if not _get_configured_users():
        st.error(
            "⚠️ No hay credenciales de vendedor configuradas en `.streamlit/secrets.toml`. "
            "Ejecuta `generar_credenciales_vendedor.py` para crearlas."
        )
        st.stop()

    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Usuario")
        password = st.text_input("Contraseña", type="password")
        submitted = st.form_submit_button("Ingresar", use_container_width=True)

    if submitted:
        if _validate_credentials(username, password):
            st.session_state[SESSION_KEY] = True
            st.session_state[SESSION_USER_KEY] = username
            st.rerun()
        else:
            st.error("❌ Usuario o contraseña incorrectos.")

    st.stop()


def logout_button(location=st.sidebar) -> None:
    """Dibuja un botón de cierre de sesión (por defecto, en la barra lateral)."""
    usuario = st.session_state.get(SESSION_USER_KEY, "")
    if usuario:
        location.caption(f"Sesión iniciada como **{usuario}**")
    if location.button("🚪 Cerrar sesión", use_container_width=True):
        st.session_state[SESSION_KEY] = False
        st.session_state.pop(SESSION_USER_KEY, None)
        st.rerun()
