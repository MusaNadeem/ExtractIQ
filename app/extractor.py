import json
import logging
import os
from io import BytesIO
from pathlib import Path

import pdfplumber
from dotenv import load_dotenv
from google import genai
from google.genai import types as _genai_types
from PIL import Image as _PILImage

from app.schemas.invoice import InvoiceData
from app.validator import ValidationResult, validate_invoice

load_dotenv()

log = logging.getLogger(__name__)

_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
_MODEL = "gemini-2.0-flash"

_TEXT_MIN_CHARS = 100
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

_VISION_PROMPT = (
    "Extract invoice data from this document image. "
    "Return ONLY valid JSON as specified."
)


# ---------------------------------------------------------------------------
# LLM interface — single call-through function
# ---------------------------------------------------------------------------

def call_llm(prompt: str, image=None) -> str:
    """
    Send a prompt to Gemini and return the raw text response.

    image: PIL.Image.Image, raw bytes, or None.
    For multi-page documents, stitch pages before calling (see _stitch_images).
    """
    if image is None:
        contents = prompt
    else:
        if isinstance(image, bytes):
            image = _PILImage.open(BytesIO(image))
        contents = [prompt, image]

    response = _client.models.generate_content(
        model=_MODEL,
        contents=contents,
        config=_genai_types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
            temperature=0.0,
        ),
    )
    text = response.text
    if not text:
        raise ValueError("Gemini returned an empty response")
    return text.strip()


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


def _stitch_images(images: list) -> _PILImage.Image:
    """Stack multiple page images vertically into one for vision input."""
    if len(images) == 1:
        return images[0]
    total_h = sum(img.height for img in images)
    max_w = max(img.width for img in images)
    canvas = _PILImage.new("RGB", (max_w, total_h), "white")
    y = 0
    for img in images:
        canvas.paste(img, (0, y))
        y += img.height
    return canvas


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
                f"Gemini returned malformed JSON after two attempts.\nLast response:\n{raw}"
            ) from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_invoice(text: str) -> ValidationResult:
    """Extract invoice data from a pre-extracted text string."""
    result = validate_invoice(_extract_with_retry(lambda: call_llm(text)))
    result.route = "text"
    return result


def extract_invoice_from_image(image) -> ValidationResult:
    """Extract invoice data from a single PIL Image."""
    log.info("Route → vision  (direct image upload)")
    result = validate_invoice(_extract_with_retry(lambda: call_llm(_VISION_PROMPT, image)))
    result.route = "vision"
    return result


def extract_invoice_from_pdf(path: Path) -> ValidationResult:
    """
    Routing layer:
      1. pdfplumber  → text long enough and clean  → Gemini text
      2. pdfplumber  → too short / garbled          → Gemini vision
      3. vision fails                                → Tesseract → Gemini text

    Every path returns a ValidationResult with warnings attached.
    """
    path = Path(path)

    # Step 1: attempt digital text extraction
    text = _extract_pdfplumber(path)

    if _is_meaningful_text(text):
        log.info("[%s] Route → text   (%d chars via pdfplumber → Gemini)", path.name, len(text))
        result = validate_invoice(_extract_with_retry(lambda: call_llm(text)))
        result.route = "text"
        return result

    # Step 2: scanned / image-based PDF — use vision
    log.info(
        "[%s] Route → vision  (pdfplumber yielded %d chars — too short or garbled)",
        path.name,
        len(text),
    )
    from app.ocr import pdf_to_images, ocr_images
    images = pdf_to_images(path)

    try:
        stitched = _stitch_images(images)
        result = validate_invoice(_extract_with_retry(lambda: call_llm(_VISION_PROMPT, stitched)))
        result.route = "vision"
        return result
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
            "[%s] OCR yielded %d chars — sending to Gemini text", path.name, len(ocr_text)
        )
        result = validate_invoice(_extract_with_retry(lambda: call_llm(ocr_text)))
        result.route = "ocr"
        return result
