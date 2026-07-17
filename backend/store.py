"""
Persistencia de la tienda (productos y órdenes) usando SQLite local.

Reemplaza a Supabase: en vez de depender de un servicio externo, el propio
backend FastAPI (ya desplegado en Render) guarda los datos en un archivo
SQLite. Tanto `admin_vendedor.py` como `tienda_cliente.py` (cada uno
desplegado por separado en Streamlit Community Cloud) hablan con este
backend por HTTP, así que ambos ven siempre los mismos datos.

Al arrancar, si la tabla de productos está vacía, se siembra automáticamente
con los 30 productos más vendidos del dataset histórico (los mismos que
tienen historial de demanda calculado en `product_history.json`), para que
la tienda y el panel de vendedor tengan contenido coherente con el módulo
de predicción desde el primer despliegue.
"""
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from backend.config import settings

DB_PATH: Path = settings.BACKEND_MODEL_DIR.parent / "store" / "store.db"
SEED_PATH: Path = settings.BACKEND_MODEL_DIR / "top_products_seed.json"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS products (
    id TEXT PRIMARY KEY,
    stock_code TEXT,
    name TEXT NOT NULL,
    description TEXT,
    price REAL NOT NULL DEFAULT 0,
    stock INTEGER NOT NULL DEFAULT 0,
    image_data TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS orders (
    id TEXT PRIMARY KEY,
    customer_name TEXT NOT NULL,
    customer_email TEXT,
    customer_phone TEXT,
    shipping_address TEXT,
    total_amount REAL NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'Pendiente',
    items TEXT,
    created_at TEXT NOT NULL
);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Crea las tablas si no existen y siembra los 30 productos más vendidos."""
    with _connect() as conn:
        conn.executescript(_SCHEMA)
        count = conn.execute("SELECT COUNT(*) AS c FROM products").fetchone()["c"]
        if count == 0:
            _seed_products(conn)


def _seed_products(conn: sqlite3.Connection) -> None:
    if not SEED_PATH.exists():
        return
    try:
        with open(SEED_PATH, "r", encoding="utf-8") as f:
            seed = json.load(f)
    except Exception:
        return

    now = _now_iso()
    rows = [
        (
            str(uuid.uuid4()),
            item.get("stock_code"),
            item.get("name"),
            item.get("description", ""),
            float(item.get("price", 0)),
            int(item.get("stock", 0)),
            None,
            now,
        )
        for item in seed
    ]
    conn.executemany(
        "INSERT INTO products (id, stock_code, name, description, price, stock, image_data, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )


def _product_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    # `image_data` ya es una data URI (data:image/...;base64,...) lista para
    # usarse directamente como `src` de una etiqueta <img>, así el frontend
    # no necesita cambios para renderizarla.
    d["image_url"] = d.pop("image_data")
    return d


def _order_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    try:
        d["items"] = json.loads(d["items"]) if d["items"] else []
    except Exception:
        d["items"] = []
    return d


# ==================== PRODUCTOS ====================

def list_products() -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM products ORDER BY created_at DESC"
        ).fetchall()
        return [_product_to_dict(r) for r in rows]


def get_product(product_id: str) -> Optional[dict[str, Any]]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM products WHERE id = ?", (product_id,)
        ).fetchone()
        return _product_to_dict(row) if row else None


def create_product(
    name: str,
    description: str,
    price: float,
    stock: int,
    image_data: Optional[str] = None,
    stock_code: Optional[str] = None,
) -> dict[str, Any]:
    product_id = str(uuid.uuid4())
    now = _now_iso()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO products (id, stock_code, name, description, price, stock, image_data, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (product_id, stock_code, name, description, float(price), int(stock), image_data, now),
        )
    return get_product(product_id)


def update_product(
    product_id: str,
    name: str,
    description: str,
    price: float,
    stock: int,
    image_data: Optional[str] = None,
    update_image: bool = False,
) -> Optional[dict[str, Any]]:
    with _connect() as conn:
        if update_image:
            conn.execute(
                "UPDATE products SET name=?, description=?, price=?, stock=?, image_data=? WHERE id=?",
                (name, description, float(price), int(stock), image_data, product_id),
            )
        else:
            conn.execute(
                "UPDATE products SET name=?, description=?, price=?, stock=? WHERE id=?",
                (name, description, float(price), int(stock), product_id),
            )
    return get_product(product_id)


def delete_product(product_id: str) -> bool:
    with _connect() as conn:
        cur = conn.execute("DELETE FROM products WHERE id = ?", (product_id,))
        return cur.rowcount > 0


def adjust_stock(product_id: str, delta: int) -> None:
    """Suma (o resta, si delta es negativo) unidades al stock, sin bajar de 0."""
    with _connect() as conn:
        row = conn.execute("SELECT stock FROM products WHERE id=?", (product_id,)).fetchone()
        if row is None:
            return
        new_stock = max(0, row["stock"] + delta)
        conn.execute("UPDATE products SET stock=? WHERE id=?", (new_stock, product_id))


# ==================== ÓRDENES ====================

def list_orders() -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM orders ORDER BY created_at DESC"
        ).fetchall()
        return [_order_to_dict(r) for r in rows]


def get_order(order_id: str) -> Optional[dict[str, Any]]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM orders WHERE id=?", (order_id,)).fetchone()
        return _order_to_dict(row) if row else None


def create_order(
    customer_name: str,
    customer_email: str,
    customer_phone: str,
    shipping_address: str,
    items: list[dict[str, Any]],
    total_amount: float,
) -> dict[str, Any]:
    order_id = str(uuid.uuid4())
    now = _now_iso()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO orders (id, customer_name, customer_email, customer_phone, shipping_address, "
            "total_amount, status, items, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                order_id,
                customer_name,
                customer_email,
                customer_phone,
                shipping_address,
                float(total_amount),
                "Pendiente",
                json.dumps(items, ensure_ascii=False),
                now,
            ),
        )
    return get_order(order_id)


def update_order_status(order_id: str, new_status: str) -> Optional[dict[str, Any]]:
    order = get_order(order_id)
    if order is None:
        return None

    with _connect() as conn:
        conn.execute("UPDATE orders SET status=? WHERE id=?", (new_status, order_id))

    # Si se marca como Completado, descuenta stock automáticamente (una sola
    # vez, aquí en el servidor, para evitar condiciones de carrera entre
    # el panel de vendedor y la tienda).
    if new_status == "Completado" and order["status"] != "Completado":
        products = {p["name"]: p for p in list_products()}
        for item in order.get("items", []):
            product = products.get(item.get("name"))
            if product:
                adjust_stock(product["id"], -int(item.get("quantity", 0)))

    return get_order(order_id)
