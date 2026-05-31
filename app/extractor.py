import json
import logging
import os
from pathlib import Path

import pdfplumber
from dotenv import load_dotenv
from openai import OpenAI

from app.schemas.invoice import InvoiceData

load_dotenv()

log = logging.getLogger(__name__)

_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Minimum characters (after strip) for extracted text to be considered usable
_TEXT_MIN_CHARS = 100
# Minimum ratio of printable characters — filters garbled/binary bleed-through
_PRINTABLE_RATIO_MIN = 0.80

_SYSTEM_PROMPT = """\
You are a document data extraction assistant. Extract invoice data from the provided text and return ONLY valid JSON — no markdown, no explanation, no code fences.

The JSON must match this schema exactly:
{
  "vendor_name": string | null,
  "vendor_address": string | null,
  "invoice_number": string | null,
  "invoice_date": "YYYY-MM-DD" | null,
  "due_date": "YYYY-MM-DD" | null,
  "line_items": [
    {
      "description": string | null,
      "quantity": number | null,
      "unit_price": number | null,
      "total": number | null
    }
  ],
  "subtotal": number | null,
  "tax": number | null,
  "total": number | null,
  "currency": string | null,
  "uncertain_fields": [list of field names you are not confident about]
}

Rules:
- Use null for any field not found or not determinable from the text.
- Dates must be in YYYY-MM-DD format.
- Numbers must be plain numeric values (no currency symbols).
- List field names in uncertain_fields whenever you are guessing or the source text is ambiguous.
- Return ONLY the JSON object.\
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_meaningful_text(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < _TEXT_MIN_CHARS:
        return False
    printable = sum(c.isprintable() for c in stripped)
    return (printable / len(stripped)) >= _PRINTABLE_RATIO_MIN


def _extract_pdfplumber(path: Path) -> str:
    with pdfplumber.open(path) as pdf:
        return "\n\n".join(page.extract_text() or "" for page in pdf.pages).strip()


def _parse_response(raw: str) -> InvoiceData:
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return InvoiceData.model_validate(json.loads(raw))


def _extract_with_retry(call_fn) -> InvoiceData:
    """Call call_fn() to get a raw JSON string, parse it; retry once on failure."""
    raw = call_fn()
    try:
        return _parse_response(raw)
    except (json.JSONDecodeError, ValueError):
        raw = call_fn()
        try:
            return _parse_response(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(
                f"OpenAI returned malformed JSON after two attempts.\nLast response:\n{raw}"
            ) from exc


# ---------------------------------------------------------------------------
# OpenAI callers
# ---------------------------------------------------------------------------

def _call_openai_text(text: str) -> str:
    response = _client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
    )
    return response.choices[0].message.content.strip()


def _call_openai_vision(images) -> str:
    from app.ocr import image_to_base64
    content = [
        {
            "type": "text",
            "text": "Extract invoice data from this document image. Return ONLY valid JSON as specified.",
        }
    ]
    for img in images:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_to_base64(img)}",
                    "detail": "high",
                },
            }
        )
    response = _client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_invoice(text: str) -> InvoiceData:
    """Extract invoice data from a pre-extracted text string."""
    return _extract_with_retry(lambda: _call_openai_text(text))


def extract_invoice_from_pdf(path: Path) -> InvoiceData:
    """
    Routing layer:
      1. pdfplumber  → text long enough and clean  → GPT-4o text
      2. pdfplumber  → too short / garbled          → GPT-4o vision
      3. vision fails                                → Tesseract → GPT-4o text
    """
    path = Path(path)

    # Step 1: attempt digital text extraction
    text = _extract_pdfplumber(path)

    if _is_meaningful_text(text):
        log.info("[%s] Route → text   (%d chars via pdfplumber → GPT-4o text)", path.name, len(text))
        return _extract_with_retry(lambda: _call_openai_text(text))

    # Step 2: scanned / image-based PDF — use vision API
    log.info(
        "[%s] Route → vision  (pdfplumber yielded %d chars — too short or garbled)",
        path.name,
        len(text),
    )
    from app.ocr import pdf_to_images, ocr_images
    images = pdf_to_images(path)

    try:
        return _extract_with_retry(lambda: _call_openai_vision(images))
    except Exception as vision_exc:
        # Step 3: Tesseract last-resort fallback
        log.warning(
            "[%s] Route → OCR fallback  (vision failed: %s)", path.name, vision_exc
        )
        ocr_text = ocr_images(images)
        if not _is_meaningful_text(ocr_text):
            raise ValueError(
                f"Could not extract readable text from '{path.name}' via any method."
            ) from vision_exc
        log.info(
            "[%s] OCR yielded %d chars — sending to GPT-4o text", path.name, len(ocr_text)
        )
        return _extract_with_retry(lambda: _call_openai_text(ocr_text))
