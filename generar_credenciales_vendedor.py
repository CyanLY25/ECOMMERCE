#!/usr/bin/env python3
"""
Genera credenciales de vendedor listas para pegar en `.streamlit/secrets.toml`.

Uso:
    python generar_credenciales_vendedor.py

No guarda nada por sí solo: solo imprime el bloque TOML para que lo copies
manualmente a tu `.streamlit/secrets.toml` local y, para producción, a los
"Secrets" de Streamlit Community Cloud (Settings > Secrets del app).
"""
import getpass
import secrets as secrets_module

from auth_utils import hash_password


def main():
    print("=" * 60)
    print("Generador de credenciales de vendedor (Panel de Administración)")
    print("=" * 60)

    username = input("Usuario (ej. 'admin'): ").strip()
    if not username:
        print("El usuario no puede estar vacío.")
        return

    password = getpass.getpass("Contraseña: ")
    password_confirm = getpass.getpass("Confirma la contraseña: ")
    if password != password_confirm:
        print("❌ Las contraseñas no coinciden. Intenta de nuevo.")
        return
    if len(password) < 8:
        print("⚠️  Se recomienda una contraseña de al menos 8 caracteres.")

    salt = secrets_module.token_hex(16)
    password_hash = hash_password(password, salt)

    print("\n✅ Agrega este bloque a tu .streamlit/secrets.toml "
          "(y a los Secrets del despliegue en Streamlit Community Cloud):\n")
    print("[vendedor]")
    print(f'username = "{username}"')
    print(f'password_hash = "{password_hash}"')
    print(f'salt = "{salt}"')
    print()


if __name__ == "__main__":
    main()
