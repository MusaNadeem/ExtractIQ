import json
import os
import sys
import tempfile
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

# Ensure project root is on sys.path regardless of launch directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from app.extractor import extract_invoice_from_image, extract_invoice_from_pdf
from app.schemas.invoice import InvoiceData

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ExtractIQ",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    /* Tighten default Streamlit padding */
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }

    /* Uncertain-field label highlight */
    .uncertain-label {
        display: inline-block;
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 4px;
        padding: 1px 6px;
        font-size: 0.78rem;
        font-weight: 600;
        color: #856404;
        margin-bottom: 2px;
    }

    /* Route badge chip */
    .route-chip {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .route-text   { background:#d1fae5; color:#065f46; }
    .route-vision { background:#dbeafe; color:#1e40af; }
    .route-ocr    { background:#fef3c7; color:#92400e; }

    /* Section divider */
    hr.section { margin: 0.6rem 0; border-color: #e5e7eb; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📄 ExtractIQ")
st.caption("AI-powered invoice data extraction · upload a PDF or image to begin")
st.markdown("---")

# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
uploaded = st.file_uploader(
    "Drop a PDF or image here",
    type=["pdf", "png", "jpg", "jpeg"],
    label_visibility="visible",
)

if not uploaded:
    st.markdown(
        """
        <div style="text-align:center;padding:3rem 0;color:#9ca3af;">
            <div style="font-size:3rem">📂</div>
            <div>Upload an invoice PDF or image to extract structured data</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ---------------------------------------------------------------------------
# Session-state reset when a new file is uploaded
# ---------------------------------------------------------------------------
_WIDGET_KEYS = [
    "vendor_name", "invoice_number", "vendor_address",
    "invoice_date", "due_date", "currency",
    "subtotal", "tax", "total",
    "line_items_editor",
]

if st.session_state.get("_uploaded_name") != uploaded.name:
    for k in ["_result", "_doc_images"] + _WIDGET_KEYS:
        st.session_state.pop(k, None)
    st.session_state["_uploaded_name"] = uploaded.name

# ---------------------------------------------------------------------------
# Extraction (runs once per upload, cached in session state)
# ---------------------------------------------------------------------------
if "_result" not in st.session_state:
    with st.spinner("Extracting data…"):
        file_bytes = uploaded.read()
        is_pdf = uploaded.type == "application/pdf"

        if is_pdf:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = Path(tmp.name)
            try:
                from app.ocr import pdf_to_images
                doc_images = pdf_to_images(tmp_path)
                result = extract_invoice_from_pdf(tmp_path)
            finally:
                os.unlink(tmp_path)
        else:
            pil_img = Image.open(BytesIO(file_bytes)).convert("RGB")
            doc_images = [pil_img]
            result = extract_invoice_from_image(pil_img)

    st.session_state["_result"] = result
    st.session_state["_doc_images"] = doc_images

result = st.session_state["_result"]
doc_images: list[Image.Image] = st.session_state["_doc_images"]
data: InvoiceData = result.data
uncertain: set[str] = set(data.uncertain_fields)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_ROUTE_META = {
    "text":   ("📄 Text  —  pdfplumber → GPT-4o",      "route-text"),
    "vision": ("🔍 Vision  —  GPT-4o vision API",        "route-vision"),
    "ocr":    ("🔠 OCR  —  Tesseract → GPT-4o fallback", "route-ocr"),
}


def _lbl(field: str, display: str) -> str:
    """Return label text, prefixed with ⚠️ for uncertain fields."""
    return f"⚠️ {display}" if field in uncertain else display


def _fmt_num(v) -> str:
    return f"{v:.2f}" if v is not None else ""


def _parse_num(s: str):
    try:
        return float(s.strip()) if s.strip() else None
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
left_col, right_col = st.columns([9, 11], gap="large")

# ── LEFT: Document preview ──────────────────────────────────────────────────
with left_col:
    st.subheader("Document")

    if len(doc_images) > 1:
        page_idx = st.selectbox(
            "Page",
            range(len(doc_images)),
            format_func=lambda i: f"Page {i + 1} of {len(doc_images)}",
        )
    else:
        page_idx = 0

    st.image(doc_images[page_idx], use_column_width=True)

# ── RIGHT: Extracted data ────────────────────────────────────────────────────
with right_col:

    # Route badge
    route_label, route_cls = _ROUTE_META.get(
        result.route, (f"Route: {result.route}", "route-text")
    )
    st.markdown(
        f'<span class="route-chip {route_cls}">{route_label}</span>',
        unsafe_allow_html=True,
    )

    # Uncertain-field legend (only when there are uncertain fields)
    if uncertain:
        fields_str = ", ".join(sorted(uncertain))
        st.markdown(
            f'<span class="uncertain-label">⚠️ Uncertain fields: {fields_str}</span>',
            unsafe_allow_html=True,
        )
    st.markdown("<hr class='section'>", unsafe_allow_html=True)

    # ── Vendor ───────────────────────────────────────────────────────────────
    st.markdown("**Vendor**")
    v_col1, v_col2 = st.columns(2)
    vendor_name = v_col1.text_input(
        _lbl("vendor_name", "Vendor Name"),
        value=data.vendor_name or "",
        key="vendor_name",
    )
    invoice_number = v_col2.text_input(
        _lbl("invoice_number", "Invoice Number"),
        value=data.invoice_number or "",
        key="invoice_number",
    )
    vendor_address = st.text_area(
        _lbl("vendor_address", "Vendor Address"),
        value=data.vendor_address or "",
        height=72,
        key="vendor_address",
    )

    st.markdown("<hr class='section'>", unsafe_allow_html=True)

    # ── Dates ────────────────────────────────────────────────────────────────
    st.markdown("**Dates & Currency**")
    d_col1, d_col2, d_col3 = st.columns(3)
    invoice_date = d_col1.text_input(
        _lbl("invoice_date", "Invoice Date"),
        value=str(data.invoice_date) if data.invoice_date else "",
        key="invoice_date",
        placeholder="YYYY-MM-DD",
    )
    due_date = d_col2.text_input(
        _lbl("due_date", "Due Date"),
        value=str(data.due_date) if data.due_date else "",
        key="due_date",
        placeholder="YYYY-MM-DD",
    )
    currency = d_col3.text_input(
        _lbl("currency", "Currency"),
        value=data.currency or "",
        key="currency",
    )

    st.markdown("<hr class='section'>", unsafe_allow_html=True)

    # ── Line items ───────────────────────────────────────────────────────────
    li_header_col, _ = st.columns([3, 1])
    li_header_col.markdown("**Line Items**")
    if "line_items" in uncertain:
        st.markdown(
            '<span class="uncertain-label">⚠️ Line item data has low confidence '
            "or arithmetic issues</span>",
            unsafe_allow_html=True,
        )

    raw_rows = [item.model_dump() for item in data.line_items]
    if not raw_rows:
        raw_rows = [{"description": None, "quantity": None, "unit_price": None, "total": None}]

    edited_df = st.data_editor(
        pd.DataFrame(raw_rows),
        num_rows="dynamic",
        use_container_width=True,
        key="line_items_editor",
        column_config={
            "description": st.column_config.TextColumn("Description", width="large"),
            "quantity":    st.column_config.NumberColumn("Qty",        format="%.2f", width="small"),
            "unit_price":  st.column_config.NumberColumn("Unit Price", format="%.2f"),
            "total":       st.column_config.NumberColumn("Total",      format="%.2f"),
        },
    )

    st.markdown("<hr class='section'>", unsafe_allow_html=True)

    # ── Totals ───────────────────────────────────────────────────────────────
    st.markdown("**Totals**")
    t_col1, t_col2, t_col3 = st.columns(3)
    subtotal_str = t_col1.text_input(
        _lbl("subtotal", "Subtotal"),
        value=_fmt_num(data.subtotal),
        key="subtotal",
    )
    tax_str = t_col2.text_input(
        _lbl("tax", "Tax"),
        value=_fmt_num(data.tax),
        key="tax",
    )
    total_str = t_col3.text_input(
        _lbl("total", "Total"),
        value=_fmt_num(data.total),
        key="total",
    )

    st.markdown("<hr class='section'>", unsafe_allow_html=True)

    # ── Validation warnings ──────────────────────────────────────────────────
    if result.warnings:
        with st.expander(f"⚠️ {len(result.warnings)} validation warning(s)", expanded=False):
            for w in result.warnings:
                st.warning(w, icon="⚠️")
    else:
        st.success("All checks passed", icon="✅")

    st.markdown("<hr class='section'>", unsafe_allow_html=True)

    # ── JSON download ────────────────────────────────────────────────────────
    line_items_out = (
        edited_df.where(pd.notnull(edited_df), None).to_dict(orient="records")
    )

    output = {
        "vendor_name":    vendor_name or None,
        "vendor_address": vendor_address or None,
        "invoice_number": invoice_number or None,
        "invoice_date":   invoice_date or None,
        "due_date":       due_date or None,
        "currency":       currency or None,
        "line_items":     line_items_out,
        "subtotal":       _parse_num(subtotal_str),
        "tax":            _parse_num(tax_str),
        "total":          _parse_num(total_str),
        "uncertain_fields": sorted(uncertain),
        "_meta": {
            "route":    result.route,
            "warnings": result.warnings,
            "source":   uploaded.name,
        },
    }
    json_str = json.dumps(output, indent=2, default=str)

    st.download_button(
        label="⬇️  Download JSON",
        data=json_str,
        file_name=f"{Path(uploaded.name).stem}_extracted.json",
        mime="application/json",
        use_container_width=True,
        type="primary",
    )
