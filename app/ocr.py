import logging
from io import BytesIO
from pathlib import Path

import pytesseract
from pdf2image import convert_from_path
from PIL import Image

log = logging.getLogger(__name__)


def pdf_to_images(path: Path, dpi: int = 200) -> list[Image.Image]:
    log.debug("Converting '%s' to images at %d DPI", path.name, dpi)
    return convert_from_path(str(path), dpi=dpi)


def pil_to_jpeg_bytes(image: Image.Image, quality: int = 85) -> bytes:
    """Convert a PIL Image to JPEG bytes for Gemini multimodal input."""
    buf = BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def ocr_images(images: list[Image.Image]) -> str:
    log.info("OCR fallback: running Tesseract on %d page(s)", len(images))
    return "\n\n".join(pytesseract.image_to_string(img) for img in images)
