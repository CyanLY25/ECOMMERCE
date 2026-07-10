"""Streamlit dashboard for managing lineas, productos y ordenes en Supabase."""
from __future__ import annotations

import io
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import time
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from supabase import Client, create_client

st.set_page_config(page_title="Inventarios", layout="wide")


def ensure_configuration() -> bool:
    """Verify Supabase secrets before running the app."""
    supabase_secrets = st.secrets.get("supabase", {})
    missing: List[str] = []
    if not supabase_secrets.get("url"):
        missing.append("supabase.url")
    if not (supabase_secrets.get("key") or supabase_secrets.get("service_role_key")):
        missing.append("supabase.key (o service_role_key)")
    if missing:
        st.error(
            "Faltan configuraciones en `.streamlit/secrets.toml`: " + ", ".join(missing))
        return False
    return True


def supabase_tables() -> Dict[str, str]:
    supabase_secrets = st.secrets.get("supabase", {})
    return {
        "lineas": supabase_secrets.get("lineas_table", "lineas"),
        "productos": supabase_secrets.get("productos_table", "productos"),
        "ordenes": supabase_secrets.get("ordenes_table", "ordenes"),
    }


@st.cache_resource(show_spinner=False)
def get_supabase_client() -> Client:
    """Create the Supabase client using secrets."""
    supabase_secrets = st.secrets["supabase"]
    key = supabase_secrets.get("service_role_key") or supabase_secrets["key"]
    return create_client(supabase_secrets["url"], key)


def _process_response(response: Any) -> List[Dict[str, Any]]:
    if getattr(response, "error", None):
        raise RuntimeError(str(response.error))
    return getattr(response, "data", None) or []


def fetch_table_rows(table: str, order_by: Optional[str] = None) -> List[Dict[str, Any]]:
    query = get_supabase_client().table(table).select("*")
    if order_by:
        query = query.order(order_by)
    response = query.execute()
    return _process_response(response)


def _coerce_datetime(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")
    return df


def _coerce_numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    return df


def fetch_lineas_df() -> pd.DataFrame:
    tables = supabase_tables()
    data = fetch_table_rows(tables["lineas"], order_by="nombre")
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df = _coerce_datetime(df, ["created_at", "updated_at"])
    return df.sort_values("nombre")


def fetch_productos_df() -> pd.DataFrame:
    tables = supabase_tables()
    data = fetch_table_rows(tables["productos"], order_by="nombre")
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df = _coerce_datetime(df, ["created_at", "updated_at"])
    df = _coerce_numeric(df, ["stock", "stock_minimo", "price"])
    lineas_df = fetch_lineas_df()
    if not lineas_df.empty and "linea_id" in df.columns:
        lineas_merge = lineas_df.rename(
            columns={"id": "linea_id_ref", "nombre": "linea_nombre"})
        df = df.merge(
            lineas_merge[["linea_id_ref", "linea_nombre"]],
            left_on="linea_id",
            right_on="linea_id_ref",
            how="left",
        ).drop(columns=["linea_id_ref"])
    return df


def fetch_ordenes_df() -> pd.DataFrame:
    tables = supabase_tables()
    data = fetch_table_rows(tables["ordenes"], order_by="created_at")
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df = _coerce_datetime(df, ["created_at", "updated_at"])
    df = _coerce_numeric(df, ["cantidad", "total"])
    return df


def clear_data_caches() -> None:
    """Invalidate cached Supabase queries.

    This is safe to call whether or not the fetch functions are cached. If the
    fetch functions are decorated with Streamlit caching decorators they will
    expose a `.clear()` method; otherwise this function is a no-op.
    """
    for fn_name in ("fetch_lineas_df", "fetch_productos_df", "fetch_ordenes_df"):
        fn = globals().get(fn_name)
        if fn and hasattr(fn, "clear"):
            try:
                fn.clear()
            except Exception:
                # best-effort clear; ignore failures
                pass


def render_sidebar_navigation() -> str:
    """Render sidebar navigation and utility actions."""
    with st.sidebar:
        st.header("Navegacion")
        selection = st.radio(
            "Selecciona la vista",
            ["Resumen", "Lineas", "Productos", "Ordenes"],
            index=0,
        )
        if st.button("Actualizar datos", use_container_width=True):
            clear_data_caches()
            st.experimental_rerun()
        st.caption(
            "Usa el boton de arriba si actualizaste datos fuera de la app (por ejemplo con un seeder).")
    return selection


def insert_row(table: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    response = get_supabase_client().table(table).insert(payload).execute()
    data = _process_response(response)
    return data[0] if data else payload


def update_row(table: str, row_id: Any, updates: Dict[str, Any], id_column: str = "id") -> Dict[str, Any]:
    """Update a row in a table with retries for statement timeouts.

    If the update repeatedly fails due to a statement timeout, a RuntimeError
    with a helpful message will be raised.
    """
    max_attempts = 3
    delay = 1.0
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = get_supabase_client().table(table).update(
                updates).eq(id_column, row_id).execute()
            data = _process_response(response)
            return data[0] if data else {**updates, id_column: row_id}
        except Exception as exc:
            last_exc = exc
            msg = str(exc).lower()
            # retry on statement timeout
            if "statement timeout" in msg and attempt < max_attempts:
                time.sleep(delay)
                delay *= 2
                continue
            # otherwise raise a clearer error
            raise RuntimeError(
                f"Error al actualizar fila en '{table}' (id={row_id}): {exc}.\n"
                "Posibles causas: problemas de conectividad, locks en la BD, triggers lentos o configuracion de statement_timeout en Postgres."
            ) from exc
    # if we exhausted attempts
    raise RuntimeError(
        f"Fallo al actualizar fila {row_id} en {table}") from last_exc


def productos_summary(df: pd.DataFrame) -> Dict[str, float]:
    if df is None or df.empty:
        return {"total_productos": 0, "stock_total": 0.0, "valor_inventario": 0.0, "productos_bajo_stock": 0}
    stock_series = pd.to_numeric(
        df.get("stock", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    price_series = pd.to_numeric(
        df.get("price", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    minimo_series = pd.to_numeric(
        df.get("stock_minimo", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    low_stock = stock_series < minimo_series
    return {
        "total_productos": int(df.shape[0]),
        "stock_total": float(stock_series.sum()),
        "valor_inventario": float((stock_series * price_series).sum()),
        "productos_bajo_stock": int(low_stock.sum()),
    }


def low_stock_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    return df[df["stock"] < df["stock_minimo"]].copy()


def build_stock_by_line_chart(productos_df: pd.DataFrame) -> go.Figure:
    if productos_df.empty:
        fig = go.Figure()
        fig.update_layout(title="Stock por linea",
                          xaxis_title="Linea", yaxis_title="Stock")
        return fig
    if "linea_nombre" not in productos_df.columns:
        productos_df = productos_df.assign(linea_nombre="Sin linea")
    grouped = (
        productos_df.groupby("linea_nombre", as_index=False)["stock"]
        .sum()
        .sort_values("stock", ascending=False)
    )
    fig = px.bar(grouped, x="linea_nombre", y="stock",
                 title="Stock por linea", text_auto=True)
    fig.update_layout(xaxis_title="Linea", yaxis_title="Stock")
    return fig


def build_orders_timeseries(ordenes_df: pd.DataFrame) -> go.Figure:
    if ordenes_df.empty or "created_at" not in ordenes_df.columns:
        fig = go.Figure()
        fig.update_layout(title="Movimiento de ordenes",
                          xaxis_title="Fecha", yaxis_title="Cantidad")
        return fig
    frame = ordenes_df.dropna(subset=["created_at"]).copy()
    frame = frame.set_index("created_at").resample("D")[
        "cantidad"].sum().reset_index()
    fig = px.line(frame, x="created_at", y="cantidad",
                  title="Movimiento de ordenes", markers=True)
    fig.update_layout(xaxis_title="Fecha", yaxis_title="Cantidad")
    return fig


def build_inventory_report(productos_df: pd.DataFrame, summary: Dict[str, float]) -> io.BytesIO:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("Reporte de inventario", styles["Title"]),
        Paragraph(datetime.now().strftime("%d/%m/%Y %H:%M"), styles["Normal"]),
        Spacer(1, 12),
    ]

    if summary:
        story.append(Paragraph("Resumen", styles["Heading2"]))
        story.append(Paragraph(
            f"Total de productos: {summary['total_productos']}", styles["Normal"]))
        story.append(
            Paragraph(f"Stock total: {summary['stock_total']}", styles["Normal"]))
        story.append(Paragraph(
            f"Valor inventario: {summary['valor_inventario']:.2f}", styles["Normal"]))
        story.append(Paragraph(
            f"Productos bajo stock: {summary['productos_bajo_stock']}", styles["Normal"]))
        story.append(Spacer(1, 12))

    if productos_df.empty:
        story.append(
            Paragraph("No hay productos registrados.", styles["Italic"]))
    else:
        report_df = productos_df.copy()
        if "linea_nombre" not in report_df.columns:
            report_df = report_df.assign(linea_nombre="Sin linea")
        columns_order = ["nombre", "sku", "linea_nombre",
                         "stock", "stock_minimo", "price"]
        existing_columns = [
            col for col in columns_order if col in report_df.columns]
        report_df = report_df[existing_columns].rename(
            columns={
                "nombre": "Producto",
                "sku": "SKU",
                "linea_nombre": "Linea",
                "stock": "Stock",
                "stock_minimo": "Stock minimo",
                "price": "Precio",
            }
        )
        table = _build_pdf_table(report_df)
        story.append(table)

    doc.build(story)
    buffer.seek(0)
    return buffer


def _build_pdf_table(df: pd.DataFrame) -> Table:
    # Backwards compatible wrapper that uses a smaller default font so wide
    # tables fit better on the page. For finer control use
    # _build_pdf_table_with_options below.
    return _build_pdf_table_with_options(df, font_size=8)


def _build_pdf_table_with_options(df: pd.DataFrame, font_size: int = 8, col_widths: Optional[List[float]] = None) -> Table:
    """Build a ReportLab Table from a DataFrame with adjustable font size.

    Args:
        df: dataframe to render
        font_size: font size in points to use for the whole table
        col_widths: optional list of column widths (in points) or None to auto-size
    """
    headers = list(df.columns)
    body = df.fillna("").astype(str).values.tolist()
    table = Table([headers, *body], repeatRows=1, colWidths=col_widths)
    style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.whitesmoke, colors.HexColor("#ecf0f1")]),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("BOTTOMPADDING", (0, 0), (-1, -1), max(2, int(font_size / 2))),
        ("TOPPADDING", (0, 0), (-1, -1), max(2, int(font_size / 2))),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ])
    table.setStyle(style)
    return table


def orders_summary(ordenes_df: pd.DataFrame) -> Dict[str, float]:
    if ordenes_df is None or ordenes_df.empty:
        return {
            "total_ordenes": 0,
            "total_revenue": 0.0,
            "avg_order_value": 0.0,
            "avg_quantity": 0.0,
        }
    total_ordenes = int(ordenes_df.shape[0])
    total_revenue = float(pd.to_numeric(ordenes_df.get(
        "total", 0.0), errors="coerce").fillna(0.0).sum())
    avg_order_value = float(ordenes_df.get("total", 0.0).astype(
        float).mean()) if total_ordenes else 0.0
    avg_quantity = float(pd.to_numeric(ordenes_df.get(
        "cantidad", 0.0), errors="coerce").fillna(0.0).mean()) if total_ordenes else 0.0
    return {
        "total_ordenes": total_ordenes,
        "total_revenue": total_revenue,
        "avg_order_value": avg_order_value,
        "avg_quantity": avg_quantity,
    }


def build_price_histogram(productos_df: pd.DataFrame) -> go.Figure:
    if productos_df.empty or "price" not in productos_df.columns:
        fig = go.Figure()
        fig.update_layout(title="Distribucion de precios",
                          xaxis_title="Precio", yaxis_title="Conteo")
        return fig
    fig = px.histogram(productos_df, x="price", nbins=30,
                       title="Distribucion de precios")
    fig.update_layout(xaxis_title="Precio", yaxis_title="Conteo")
    return fig


def build_stock_histogram(productos_df: pd.DataFrame) -> go.Figure:
    if productos_df.empty or "stock" not in productos_df.columns:
        fig = go.Figure()
        fig.update_layout(title="Distribucion de stock",
                          xaxis_title="Stock", yaxis_title="Conteo")
        return fig
    fig = px.histogram(productos_df, x="stock", nbins=30,
                       title="Distribucion de stock")
    fig.update_layout(xaxis_title="Stock", yaxis_title="Conteo")
    return fig


def build_top_value_products_chart(productos_df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    if productos_df.empty or "stock" not in productos_df.columns or "price" not in productos_df.columns:
        fig = go.Figure()
        fig.update_layout(title="Top productos por valor de inventario",
                          xaxis_title="Producto", yaxis_title="Valor")
        return fig
    df = productos_df.copy()
    df["valor"] = pd.to_numeric(df.get("stock", 0.0), errors="coerce").fillna(
        0.0) * pd.to_numeric(df.get("price", 0.0), errors="coerce").fillna(0.0)
    df = df.sort_values("valor", ascending=False).head(top_n)
    name_col = "linea_nombre" if "linea_nombre" in df.columns else "nombre"
    label = df.get("nombre", df.index.astype(str))
    fig = px.bar(df, x=label, y="valor",
                 title=f"Top {top_n} productos por valor de inventario", text_auto=True)
    fig.update_layout(xaxis_title="Producto", yaxis_title="Valor")
    return fig


def build_order_status_pie(ordenes_df: pd.DataFrame) -> go.Figure:
    if ordenes_df.empty or "estado" not in ordenes_df.columns:
        fig = go.Figure()
        fig.update_layout(title="Distribucion de estados de ordenes")
        return fig
    counts = ordenes_df["estado"].value_counts().reset_index()
    counts.columns = ["estado", "cantidad"]
    fig = px.pie(counts, names="estado", values="cantidad",
                 title="Distribucion de estados de ordenes")
    return fig


def build_orders_report(ordenes_df: pd.DataFrame, productos_df: pd.DataFrame, summary: Dict[str, float]) -> io.BytesIO:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("Reporte de ordenes", styles["Title"]),
        Paragraph(datetime.now().strftime("%d/%m/%Y %H:%M"), styles["Normal"]),
        Spacer(1, 12),
    ]

    if summary:
        story.append(Paragraph("Resumen", styles["Heading2"]))
        story.append(
            Paragraph(f"Total de ordenes: {summary['total_ordenes']}", styles["Normal"]))
        story.append(
            Paragraph(f"Total revenue: {summary['total_revenue']:.2f}", styles["Normal"]))
        story.append(Paragraph(
            f"Valor promedio por orden: {summary['avg_order_value']:.2f}", styles["Normal"]))
        story.append(Paragraph(
            f"Cantidad promedio por orden: {summary['avg_quantity']:.2f}", styles["Normal"]))
        story.append(Spacer(1, 12))

    if ordenes_df.empty:
        story.append(
            Paragraph("No hay ordenes registradas.", styles["Italic"]))
    else:
        report_df = ordenes_df.copy()
        # join producto info when available
        if not productos_df.empty and "id" in productos_df.columns:
            prod_lookup = productos_df.set_index("id")[[]]
            merge_cols = [c for c in ["nombre", "sku",
                                      "linea_nombre"] if c in productos_df.columns]
            if merge_cols:
                prod_lookup = productos_df.set_index("id")[merge_cols]
                report_df = report_df.merge(
                    prod_lookup, left_on="producto_id", right_index=True, how="left")

        # Select only a subset of fields for the PDF to keep it compact
        desired_columns = ["created_at", "estado",
                           "cantidad", "total", "nombre", "sku"]
        # format created_at as readable string when present
        if "created_at" in report_df.columns:
            report_df["created_at"] = pd.to_datetime(
                report_df["created_at"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")

        existing_columns = [
            col for col in desired_columns if col in report_df.columns]
        report_df = report_df[existing_columns].fillna("")
        # Use smaller font so the table fits better on the page
        table = _build_pdf_table_with_options(report_df, font_size=8)
        story.append(table)

    doc.build(story)
    buffer.seek(0)
    return buffer


def render_textual_statistics(productos_df: pd.DataFrame, ordenes_df: Optional[pd.DataFrame] = None) -> None:
    """Render simple descriptive statistics in words into the Streamlit app.

    This prints human-readable sentences summarizing price and stock
    distributions and a short orders summary when `ordenes_df` is provided.
    """
    st.subheader("Estadísticas descriptivas")
    if productos_df is None or productos_df.empty:
        st.write("No hay productos para generar estadísticas descriptivas.")
        return

    # precios
    prices = pd.to_numeric(productos_df.get(
        "price", pd.Series(dtype=float)), errors="coerce").dropna()
    stocks = pd.to_numeric(productos_df.get(
        "stock", pd.Series(dtype=float)), errors="coerce").dropna()
    total_products = int(productos_df.shape[0])
    avg_price = float(prices.mean()) if not prices.empty else 0.0
    median_price = float(prices.median()) if not prices.empty else 0.0
    min_price = float(prices.min()) if not prices.empty else 0.0
    max_price = float(prices.max()) if not prices.empty else 0.0

    avg_stock = float(stocks.mean()) if not stocks.empty else 0.0
    median_stock = float(stocks.median()) if not stocks.empty else 0.0
    min_stock = float(stocks.min()) if not stocks.empty else 0.0
    max_stock = float(stocks.max()) if not stocks.empty else 0.0

    productos_bajo = int((pd.to_numeric(productos_df.get("stock", 0), errors="coerce").fillna(
        0) < pd.to_numeric(productos_df.get("stock_minimo", 0), errors="coerce").fillna(0)).sum())

    # Product with highest inventory value
    top_val_name = None
    try:
        temp = productos_df.copy()
        temp["valor"] = pd.to_numeric(temp.get("stock", 0), errors="coerce").fillna(
            0) * pd.to_numeric(temp.get("price", 0), errors="coerce").fillna(0)
        if not temp.empty:
            top = temp.sort_values("valor", ascending=False).iloc[0]
            top_val_name = top.get("nombre") or top.get("sku") or None
            top_val_amount = float(top.get("valor", 0.0))
        else:
            top_val_amount = 0.0
    except Exception:
        top_val_name = None
        top_val_amount = 0.0

    st.markdown(f"- Tenemos **{total_products}** productos registrados.")
    st.markdown(
        f"- Precio promedio: **{avg_price:.2f}**, mediana: **{median_price:.2f}**, rango: **{min_price:.2f}** – **{max_price:.2f}**.")
    st.markdown(
        f"- Stock promedio: **{avg_stock:.1f}**, mediana: **{median_stock:.1f}**, rango: **{min_stock:.0f}** – **{max_stock:.0f}**.")
    st.markdown(
        f"- Hay **{productos_bajo}** productos por debajo del stock mínimo configurado.")
    if top_val_name:
        st.markdown(
            f"- El producto con mayor valor de inventario es **{top_val_name}** (valor aproximado: **{top_val_amount:.2f}**).")

    # Orders summary in words when available
    if ordenes_df is not None and not ordenes_df.empty:
        ord_summary = orders_summary(ordenes_df)
        most_common_estado = None
        try:
            most_common_estado = ordenes_df["estado"].mode().iloc[0]
        except Exception:
            most_common_estado = None
        st.markdown("**Resumen de órdenes:**")
        st.markdown(f"- Total de órdenes: **{ord_summary['total_ordenes']}**.")
        st.markdown(
            f"- Ingreso total aproximado: **{ord_summary['total_revenue']:.2f}**, valor promedio por orden: **{ord_summary['avg_order_value']:.2f}**.")
        if most_common_estado:
            st.markdown(
                f"- Estado más frecuente de órdenes: **{most_common_estado}**.")


def render_overview(productos_df: pd.DataFrame) -> None:
    st.subheader("Resumen general")
    summary = productos_summary(productos_df)
    cols = st.columns(4)
    cols[0].metric("Productos registrados", summary["total_productos"])
    cols[1].metric("Stock total", f"{summary['stock_total']:.0f}")
    cols[2].metric("Valor inventario", f"{summary['valor_inventario']:.2f}")
    cols[3].metric("Productos bajo stock", summary["productos_bajo_stock"])

    low_stock = low_stock_df(productos_df)
    if low_stock.empty:
        st.success("No hay productos por debajo del stock minimo.")
    else:
        st.warning("Productos con stock por debajo del minimo configurado:")
        display_columns = ["nombre", "sku",
                           "linea_nombre", "stock", "stock_minimo"]
        existing_columns = [
            col for col in display_columns if col in low_stock.columns]
        st.dataframe(low_stock[existing_columns], use_container_width=True)

    # Texto: estadísticas descriptivas en palabras
    try:
        render_textual_statistics(productos_df)
    except Exception:
        # no queremos romper la vista si hay algún problema con cálculos
        pass


def render_charts(productos_df: pd.DataFrame, ordenes_df: pd.DataFrame) -> None:
    st.subheader("Graficos")
    # Row 1: stock by linea + orders timeseries
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(build_stock_by_line_chart(
            productos_df), use_container_width=True)
    with col2:
        st.plotly_chart(build_orders_timeseries(
            ordenes_df), use_container_width=True)

    # Row 2: price distribution + stock distribution
    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(build_price_histogram(
            productos_df), use_container_width=True)
    with col4:
        st.plotly_chart(build_stock_histogram(
            productos_df), use_container_width=True)

    # Row 3: top value products + orders status pie
    col5, col6 = st.columns(2)
    with col5:
        st.plotly_chart(build_top_value_products_chart(
            productos_df), use_container_width=True)
    with col6:
        st.plotly_chart(build_order_status_pie(
            ordenes_df), use_container_width=True)


def render_reports(productos_df: pd.DataFrame) -> None:
    st.subheader("Reportes")
    summary = productos_summary(productos_df)
    if productos_df.empty:
        st.info("Carga productos para generar un reporte.")
        return
    if st.button("Generar reporte PDF"):
        buffer = build_inventory_report(productos_df, summary)
        filename = f"reporte_inventario_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        st.download_button(
            label=f"Descargar {filename}",
            data=buffer,
            file_name=filename,
            mime="application/pdf",
        )


def render_lineas_manager(lineas_df: pd.DataFrame) -> None:
    tables = supabase_tables()
    st.subheader("Lineas")
    if lineas_df.empty:
        st.info(
            "No hay lineas registradas. Utiliza el formulario siguiente para crear la primera linea.")
    else:
        display_columns = ["nombre", "descripcion", "created_at", "updated_at"]
        existing_columns = [
            column for column in display_columns if column in lineas_df.columns]
        st.dataframe(lineas_df[existing_columns], use_container_width=True)
        st.caption(
            "Gestiona la creacion y actualizacion de lineas usando los formularios a continuacion.")

    with st.expander("Registrar nueva linea", expanded=lineas_df.empty):
        with st.form("form_linea_new"):
            nombre = st.text_input("Nombre de la linea")
            descripcion = st.text_area("Descripcion", height=80)
            submit = st.form_submit_button("Guardar linea")
        if submit:
            if not nombre.strip():
                st.warning("El nombre es obligatorio.")
            else:
                insert_row(
                    tables["lineas"],
                    {"nombre": nombre.strip(), "descripcion": descripcion.strip() or None},
                )
                clear_data_caches()
                st.success("Linea registrada correctamente.")
                st.experimental_rerun()

    if not lineas_df.empty:
        with st.expander("Actualizar linea existente"):
            options = {
                row["id"]: f"{row['nombre']} ({row['id'][:8]})"
                for _, row in lineas_df.iterrows()
            }
            selected_id = st.selectbox(
                "Selecciona la linea",
                options=list(options.keys()),
                format_func=lambda value: options.get(value, value),
                key="linea_update_select",
            )
            if selected_id:
                selected_row = lineas_df[lineas_df["id"]
                                         == selected_id].iloc[0]
                with st.form(f"form_linea_update_{selected_id}"):
                    new_nombre = st.text_input(
                        "Nombre",
                        value=selected_row.get("nombre", ""),
                    )
                    new_descripcion = st.text_area(
                        "Descripcion",
                        value=selected_row.get("descripcion", "") or "",
                        height=80,
                    )
                    submit_update = st.form_submit_button("Actualizar linea")
                if submit_update:
                    if not new_nombre.strip():
                        st.warning("El nombre es obligatorio.")
                    else:
                        try:
                            update_row(
                                tables["lineas"],
                                selected_id,
                                {"nombre": new_nombre.strip(
                                ), "descripcion": new_descripcion.strip() or None},
                            )
                        except Exception as e:
                            st.error(f"No se pudo actualizar la linea: {e}")
                        else:
                            clear_data_caches()
                            st.success("Linea actualizada.")
                            st.experimental_rerun()


def render_productos_manager(productos_df: pd.DataFrame, lineas_df: pd.DataFrame) -> None:
    tables = supabase_tables()
    st.subheader("Productos")

    if productos_df.empty:
        if lineas_df.empty:
            st.info(
                "No hay productos registrados. Crea una linea y luego registra productos usando el formulario de esta vista.")
        else:
            st.info(
                "No hay productos registrados. Usa el formulario de esta vista para agregar productos.")
    else:
        display_columns = [
            "nombre",
            "sku",
            "linea_nombre",
            "stock",
            "stock_minimo",
            "price",
            "updated_at",
        ]
        existing_columns = [
            col for col in display_columns if col in productos_df.columns]
        st.dataframe(productos_df[existing_columns], use_container_width=True)
        st.caption(
            "Los formularios de alta y edicion se encuentran a continuacion.")

    with st.expander("Registrar nuevo producto", expanded=productos_df.empty):
        with st.form("form_producto_new"):
            if lineas_df.empty:
                st.info("Registra al menos una linea antes de crear productos.")
            linea_options = {row["id"]: row["nombre"]
                             for _, row in lineas_df.iterrows()}
            linea_id = st.selectbox(
                "Linea",
                options=list(linea_options.keys()) if linea_options else [],
                format_func=lambda value: linea_options.get(
                    value, "Sin lineas"),
            )
            nombre = st.text_input("Nombre del producto")
            sku = st.text_input("SKU")
            stock = st.number_input("Stock inicial", min_value=0.0, step=1.0)
            stock_minimo = st.number_input(
                "Stock minimo aceptable", min_value=0.0, step=1.0, value=5.0)
            price = st.number_input("Precio unitario", min_value=0.0, step=1.0)
            submit = st.form_submit_button(
                "Guardar producto", disabled=lineas_df.empty)
        if submit:
            if not nombre.strip():
                st.warning("El nombre es obligatorio.")
            elif not linea_id:
                st.warning("Selecciona una linea.")
            else:
                insert_row(
                    tables["productos"],
                    {
                        "linea_id": linea_id,
                        "nombre": nombre.strip(),
                        "sku": sku.strip() or None,
                        "stock": float(stock),
                        "stock_minimo": float(stock_minimo),
                        "price": float(price),
                    },
                )
                clear_data_caches()
                st.success("Producto registrado correctamente.")
                st.experimental_rerun()

    if not productos_df.empty:
        with st.expander("Actualizar inventario de producto"):
            producto_options = {
                row["id"]: f"{row['nombre']} ({row.get('sku', 'sin SKU')})"
                for _, row in productos_df.iterrows()
            }
            selected_id = st.selectbox(
                "Selecciona el producto",
                options=list(producto_options.keys()),
                format_func=lambda value: producto_options.get(value, value),
                key="producto_update_select",
            )
            if selected_id:
                selected_row = productos_df[productos_df["id"]
                                            == selected_id].iloc[0]
                with st.form(f"form_producto_update_{selected_id}"):
                    new_stock = st.number_input(
                        "Stock",
                        min_value=0.0,
                        value=float(selected_row.get("stock", 0.0)),
                        step=1.0,
                    )
                    new_stock_minimo = st.number_input(
                        "Stock minimo aceptable",
                        min_value=0.0,
                        value=float(selected_row.get("stock_minimo", 0.0)),
                        step=1.0,
                    )
                    new_price = st.number_input(
                        "Precio unitario",
                        min_value=0.0,
                        value=float(selected_row.get("price", 0.0)),
                        step=1.0,
                    )
                    submit_update = st.form_submit_button("Guardar cambios")
                if submit_update:
                    try:
                        update_row(
                            tables["productos"],
                            selected_id,
                            {
                                "stock": float(new_stock),
                                "stock_minimo": float(new_stock_minimo),
                                "price": float(new_price),
                            },
                        )
                    except Exception as e:
                        st.error(f"No se pudo actualizar el producto: {e}")
                    else:
                        clear_data_caches()
                        st.success("Producto actualizado.")
                        st.experimental_rerun()


def render_ordenes_manager(ordenes_df: pd.DataFrame, productos_df: pd.DataFrame) -> None:
    tables = supabase_tables()
    st.subheader("Ordenes")

    if ordenes_df.empty:
        st.info("No hay ordenes registradas.")
    else:
        productos_lookup = productos_df.set_index(
            "id") if not productos_df.empty else pd.DataFrame()
        display_df = ordenes_df.copy()
        if not productos_lookup.empty:
            display_df = display_df.merge(
                productos_lookup[["nombre", "sku", "linea_nombre"]],
                left_on="producto_id",
                right_index=True,
                how="left",
            )
        display_columns = ["created_at", "estado", "cantidad",
                           "total", "nombre", "sku", "linea_nombre", "nota"]
        existing_columns = [
            col for col in display_columns if col in display_df.columns]
        st.dataframe(display_df[existing_columns], use_container_width=True)
        # PDF export for orders
        summary = orders_summary(ordenes_df)
        if st.button("Generar reporte PDF de ordenes"):
            buffer = build_orders_report(ordenes_df, productos_df, summary)
            filename = f"reporte_ordenes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            st.download_button(
                label=f"Descargar {filename}",
                data=buffer,
                file_name=filename,
                mime="application/pdf",
            )

    with st.expander("Registrar nueva orden", expanded=ordenes_df.empty):
        with st.form("form_orden_new"):
            if productos_df.empty:
                st.info("Registra al menos un producto antes de crear ordenes.")
            producto_options = {
                row["id"]: f"{row['nombre']} (stock {row['stock']:.0f})"
                for _, row in productos_df.iterrows()
            }
            producto_id = st.selectbox(
                "Producto",
                options=list(producto_options.keys()
                             ) if producto_options else [],
                format_func=lambda value: producto_options.get(
                    value, "Sin productos"),
            )
            cantidad = st.number_input(
                "Cantidad", min_value=0.0, step=1.0, value=1.0)
            estado = st.selectbox(
                "Estado", options=["pendiente", "procesando", "completada", "cancelada"])
            nota = st.text_area("Nota", height=80)
            submit = st.form_submit_button("Guardar orden")
        if submit:
            if not producto_id:
                st.warning("Selecciona un producto.")
            elif cantidad <= 0:
                st.warning("La cantidad debe ser mayor a cero.")
            else:
                producto_row = productos_df[productos_df["id"] == producto_id]
                if producto_row.empty:
                    st.error("Producto no encontrado.")
                else:
                    producto_data = producto_row.iloc[0]
                    total = float(cantidad) * \
                        float(producto_data.get("price", 0.0))
                    insert_row(
                        tables["ordenes"],
                        {
                            "producto_id": producto_id,
                            "cantidad": float(cantidad),
                            "estado": estado,
                            "nota": nota.strip() or None,
                            "total": total,
                        },
                    )
                    nuevo_stock = max(float(producto_data.get(
                        "stock", 0.0)) - float(cantidad), 0.0)
                    update_row(
                        tables["productos"],
                        producto_id,
                        {"stock": nuevo_stock},
                    )
                    # invalidate any query caches if present
                    clear_data_caches()
                    st.success("Orden registrada y stock actualizado.")
                    st.experimental_rerun()


def main() -> None:
    st.title("Panel de Inventarios")
    if not ensure_configuration():
        return

    lineas_df = fetch_lineas_df()
    productos_df = fetch_productos_df()
    ordenes_df = fetch_ordenes_df()

    selected_view = render_sidebar_navigation()

    render_overview(productos_df)
    st.divider()

    if selected_view == "Resumen":
        render_charts(productos_df, ordenes_df)
        render_reports(productos_df)
        st.subheader("Detalle de productos")
        render_productos_manager(productos_df, lineas_df)
    elif selected_view == "Lineas":
        render_lineas_manager(lineas_df)
    elif selected_view == "Productos":
        render_productos_manager(productos_df, lineas_df)
    elif selected_view == "Ordenes":
        render_ordenes_manager(ordenes_df, productos_df)
    else:
        st.info("Selecciona una vista desde el sidebar para continuar.")


if __name__ == "__main__":
    main()
