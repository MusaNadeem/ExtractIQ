"""
Unit tests — no API key or real PDF required.

Covers:
  1. _is_meaningful_text  thresholds
  2. Text-PDF route  (pdfplumber returns rich text → Gemini text path)
  3. Vision route    (pdfplumber returns empty     → Gemini vision path)
  4. OCR fallback    (vision raises               → Tesseract path)
  5. validate_invoice — date, numeric, line-item, and total checks
  6. End-to-end extraction chain via mocked call_llm
"""

import json
import os
import shutil
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image as _PILImage, ImageDraw

os.environ.setdefault("GEMINI_API_KEY", "dummy-key-for-tests")

from app.extractor import _is_meaningful_text, extract_invoice, extract_invoice_from_pdf
from app.schemas.invoice import InvoiceData, LineItem
from app.validator import ValidationResult, validate_invoice


RICH_TEXT = (
    "INVOICE #INV-2024-001\n"
    "Vendor: Acme Corporation, 123 Main St, Springfield IL\n"
    "Date: 2024-01-15    Due: 2024-02-15\n"
    "Widget A   x2   $50.00   $100.00\n"
    "Widget B   x1   $75.00    $75.00\n"
    "Subtotal $175.00  Tax $17.50  Total $192.50\n"
) * 3


# ---------------------------------------------------------------------------
# _is_meaningful_text
# ---------------------------------------------------------------------------

class TestIsMeaningfulText(unittest.TestCase):
    def test_rich_text_is_meaningful(self):
        self.assertTrue(_is_meaningful_text(RICH_TEXT))

    def test_empty_string(self):
        self.assertFalse(_is_meaningful_text(""))

    def test_whitespace_only(self):
        self.assertFalse(_is_meaningful_text("   \n\n  "))

    def test_below_length_threshold(self):
        self.assertFalse(_is_meaningful_text("Invoice #001"))

    def test_garbled_binary(self):
        self.assertFalse(_is_meaningful_text("\x00\x01\x02\x03\x04\x05" * 50))

    def test_exactly_at_threshold(self):
        self.assertTrue(_is_meaningful_text("A" * 100))


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

class TestRouting(unittest.TestCase):
    _INVOICE = InvoiceData()   # bare instance passes validation cleanly

    @patch("app.extractor._extract_with_retry", return_value=InvoiceData())
    @patch("app.extractor._extract_pdfplumber", return_value=RICH_TEXT)
    def test_text_route_for_rich_pdf(self, _plumber, _retry):
        with self.assertLogs("app.extractor", level="INFO") as cm:
            result = extract_invoice_from_pdf(Path("invoice.pdf"))
        self.assertIsInstance(result, ValidationResult)
        self.assertTrue(any("Route → text" in l for l in cm.output))

    @patch("app.ocr.pdf_to_images", return_value=[MagicMock()])
    @patch("app.extractor._extract_with_retry", return_value=InvoiceData())
    @patch("app.extractor._extract_pdfplumber", return_value="")
    def test_vision_route_for_empty_pdf(self, _plumber, _retry, _images):
        with self.assertLogs("app.extractor", level="INFO") as cm:
            result = extract_invoice_from_pdf(Path("scanned.pdf"))
        self.assertIsInstance(result, ValidationResult)
        self.assertTrue(any("Route → vision" in l for l in cm.output))

    @patch("app.ocr.ocr_images", return_value=RICH_TEXT)
    @patch("app.ocr.pdf_to_images", return_value=[MagicMock()])
    @patch("app.extractor._extract_with_retry")
    @patch("app.extractor._extract_pdfplumber", return_value="")
    def test_ocr_fallback_when_vision_fails(self, _plumber, mock_retry, _images, _ocr):
        mock_retry.side_effect = [Exception("vision API error"), InvoiceData()]
        with self.assertLogs("app.extractor", level="WARNING") as cm:
            result = extract_invoice_from_pdf(Path("scanned.pdf"))
        self.assertIsInstance(result, ValidationResult)
        self.assertTrue(any("OCR fallback" in l for l in cm.output))
        self.assertEqual(mock_retry.call_count, 2)


# ---------------------------------------------------------------------------
# validate_invoice
# ---------------------------------------------------------------------------

class TestValidation(unittest.TestCase):

    def _make(self, **kwargs) -> InvoiceData:
        return InvoiceData(**kwargs)

    def test_clean_data_no_warnings(self):
        data = self._make(
            invoice_date=date(2024, 1, 15),
            due_date=date(2024, 2, 15),
            subtotal=175.0,
            tax=17.50,
            total=192.50,
            line_items=[
                LineItem(quantity=2, unit_price=50.0, total=100.0),
                LineItem(quantity=1, unit_price=75.0, total=75.0),
            ],
        )
        result = validate_invoice(data)
        self.assertEqual(result.warnings, [])
        self.assertEqual(result.data.uncertain_fields, [])

    def test_due_date_before_invoice_date(self):
        data = self._make(
            invoice_date=date(2024, 3, 1),
            due_date=date(2024, 1, 1),
        )
        result = validate_invoice(data)
        self.assertTrue(any("earlier than invoice_date" in w for w in result.warnings))
        self.assertIn("due_date", result.data.uncertain_fields)

    def test_year_out_of_range(self):
        data = self._make(invoice_date=date(1850, 1, 1))
        result = validate_invoice(data)
        self.assertTrue(any("year 1850" in w for w in result.warnings))
        self.assertIn("invoice_date", result.data.uncertain_fields)

    def test_negative_total(self):
        data = self._make(subtotal=-10.0, total=-10.0)
        result = validate_invoice(data)
        self.assertTrue(any("negative" in w for w in result.warnings))
        self.assertIn("subtotal", result.data.uncertain_fields)

    def test_line_item_qty_price_mismatch(self):
        data = self._make(
            line_items=[LineItem(quantity=2, unit_price=50.0, total=90.0)]
        )
        result = validate_invoice(data)
        self.assertTrue(any("line_items[0]" in w for w in result.warnings))
        self.assertIn("line_items", result.data.uncertain_fields)

    def test_line_items_sum_mismatch_subtotal(self):
        data = self._make(
            line_items=[LineItem(total=100.0), LineItem(total=75.0)],
            subtotal=200.0,   # should be 175.0
        )
        result = validate_invoice(data)
        self.assertTrue(any("sum" in w and "subtotal" in w for w in result.warnings))
        self.assertIn("subtotal", result.data.uncertain_fields)

    def test_total_mismatch_subtotal_plus_tax(self):
        data = self._make(subtotal=175.0, tax=17.50, total=200.0)  # should be 192.50
        result = validate_invoice(data)
        self.assertTrue(any("total" in w and "subtotal + tax" in w for w in result.warnings))
        self.assertIn("total", result.data.uncertain_fields)

    def test_existing_uncertain_fields_preserved(self):
        data = self._make(uncertain_fields=["vendor_name"], total=-5.0)
        result = validate_invoice(data)
        self.assertIn("vendor_name", result.data.uncertain_fields)
        self.assertIn("total", result.data.uncertain_fields)

    def test_within_tolerance_no_false_positive(self):
        # $0.01 rounding — should not warn
        data = self._make(subtotal=175.0, tax=17.50, total=192.49)
        result = validate_invoice(data)
        total_warnings = [w for w in result.warnings if "total" in w and "subtotal" in w]
        self.assertEqual(total_warnings, [])


# ---------------------------------------------------------------------------
# End-to-end extraction chain (call_llm mocked — no real API call)
# ---------------------------------------------------------------------------

_VALID_INVOICE_JSON = json.dumps({
    "vendor_name": "Acme Corp",
    "vendor_address": "123 Main St, Springfield IL",
    "invoice_number": "INV-2024-001",
    "invoice_date": "2024-01-15",
    "due_date": "2024-02-15",
    "line_items": [
        {"description": "Widget A", "quantity": 2.0, "unit_price": 50.0, "total": 100.0},
        {"description": "Widget B", "quantity": 1.0, "unit_price": 75.0, "total": 75.0},
    ],
    "subtotal": 175.0,
    "tax": 17.50,
    "total": 192.50,
    "currency": "USD",
    "uncertain_fields": [],
})

_INVOICE_TEXT = (
    "INVOICE #INV-2024-001\n"
    "Vendor: Acme Corp, 123 Main St, Springfield IL\n"
    "Date: 2024-01-15   Due: 2024-02-15\n"
    "Widget A  x2  $50.00  $100.00\n"
    "Widget B  x1  $75.00   $75.00\n"
    "Subtotal $175.00  Tax $17.50  Total $192.50  USD\n"
) * 3


class TestExtractionChain(unittest.TestCase):
    """
    Exercises the full parse → validate chain using a mocked call_llm.
    Confirms the Gemini swap didn't break JSON parsing, retry, or validation.
    """

    @patch("app.extractor.call_llm", return_value=_VALID_INVOICE_JSON)
    def test_returns_valid_invoice_data(self, mock_llm):
        result = extract_invoice(_INVOICE_TEXT)

        self.assertIsInstance(result, ValidationResult)
        self.assertEqual(result.route, "text")
        self.assertEqual(result.data.vendor_name, "Acme Corp")
        self.assertEqual(result.data.invoice_number, "INV-2024-001")
        self.assertEqual(len(result.data.line_items), 2)
        self.assertAlmostEqual(result.data.total, 192.50)
        self.assertEqual(result.data.currency, "USD")
        self.assertEqual(result.warnings, [])
        mock_llm.assert_called_once_with(_INVOICE_TEXT)

    @patch("app.extractor.call_llm")
    def test_retries_once_on_malformed_json(self, mock_llm):
        mock_llm.side_effect = ["not valid json {{", _VALID_INVOICE_JSON]

        result = extract_invoice(_INVOICE_TEXT)

        self.assertIsInstance(result, ValidationResult)
        self.assertEqual(mock_llm.call_count, 2)

    @patch("app.extractor.call_llm", return_value="still not json")
    def test_raises_after_two_bad_responses(self, mock_llm):
        with self.assertRaises(ValueError) as ctx:
            extract_invoice(_INVOICE_TEXT)

        self.assertIn("malformed JSON", str(ctx.exception))
        self.assertEqual(mock_llm.call_count, 2)

    @patch("app.extractor.call_llm")
    def test_strips_accidental_code_fence(self, mock_llm):
        fenced = f"```json\n{_VALID_INVOICE_JSON}\n```"
        mock_llm.return_value = fenced

        result = extract_invoice(_INVOICE_TEXT)

        self.assertEqual(result.data.vendor_name, "Acme Corp")

    @patch("app.extractor.call_llm", return_value=_VALID_INVOICE_JSON)
    def test_call_llm_receives_text_not_system_prompt(self, mock_llm):
        extract_invoice(_INVOICE_TEXT)

        call_args = mock_llm.call_args
        # image argument should be absent / None
        self.assertIsNone(call_args.kwargs.get("image"))
        # prompt should be the document text, not the system prompt
        prompt_arg = call_args.args[0] if call_args.args else call_args.kwargs["prompt"]
        self.assertEqual(prompt_arg, _INVOICE_TEXT)


# ---------------------------------------------------------------------------
# Vision path — real image-based PDF, mocked call_llm
# ---------------------------------------------------------------------------

def _make_image_pdf(path: Path) -> None:
    """Create a PDF whose content is a rasterised image (no extractable text)."""
    img = _PILImage.new("RGB", (800, 1000), "white")
    draw = ImageDraw.Draw(img)
    lines = [
        "INVOICE #INV-SCAN-001",
        "Vendor: Scanned Corp, 99 Paper Lane",
        "Date: 2024-03-10     Due: 2024-04-10",
        "",
        "Item A   x1   $200.00   $200.00",
        "Item B   x3    $15.00    $45.00",
        "",
        "Subtotal $245.00   Tax $24.50   Total $269.50",
        "Currency: USD",
    ]
    for i, line in enumerate(lines):
        draw.text((60, 80 + i * 40), line, fill="black")
    img.save(str(path), format="PDF")


class TestVisionPath(unittest.TestCase):
    """
    Uses a Pillow-generated image PDF (pdfplumber extracts zero text) to
    confirm the vision route fires and returns structured InvoiceData.
    No real Gemini API calls are made.
    """

    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.mkdtemp()
        cls._pdf_path = Path(cls._tmpdir) / "scanned_invoice.pdf"
        _make_image_pdf(cls._pdf_path)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._tmpdir, ignore_errors=True)

    @patch("app.extractor.call_llm", return_value=_VALID_INVOICE_JSON)
    def test_scanned_pdf_takes_vision_route(self, mock_llm):
        with self.assertLogs("app.extractor", level="INFO") as cm:
            result = extract_invoice_from_pdf(self._pdf_path)

        route_logs = [l for l in cm.output if "Route" in l]
        self.assertTrue(any("vision" in l for l in route_logs),
                        f"Expected vision route, got: {route_logs}")
        self.assertEqual(result.route, "vision")
        self.assertIsInstance(result, ValidationResult)

    @patch("app.extractor.call_llm", return_value=_VALID_INVOICE_JSON)
    def test_call_llm_receives_pil_image(self, mock_llm):
        extract_invoice_from_pdf(self._pdf_path)

        args, kwargs = mock_llm.call_args
        image_arg = kwargs.get("image") if kwargs else (args[1] if len(args) > 1 else None)
        self.assertIsNotNone(image_arg, "call_llm should receive an image argument")
        self.assertIsInstance(image_arg, _PILImage.Image,
                              f"Expected PIL Image, got {type(image_arg)}")

    @patch("app.extractor.call_llm", return_value=_VALID_INVOICE_JSON)
    def test_returns_valid_invoice_data_from_vision(self, mock_llm):
        result = extract_invoice_from_pdf(self._pdf_path)

        self.assertEqual(result.data.vendor_name, "Acme Corp")
        self.assertAlmostEqual(result.data.total, 192.50)
        self.assertEqual(len(result.data.line_items), 2)

    def test_pil_image_converted_to_jpeg_part(self):
        """call_llm internally wraps PIL images in a typed JPEG Part."""
        from google.genai import types
        from app.extractor import call_llm

        captured = {}

        def fake_generate(model, contents, config):
            captured["contents"] = contents

            class FakeResponse:
                text = _VALID_INVOICE_JSON
            return FakeResponse()

        with patch("app.extractor._client") as mock_client:
            mock_client.models.generate_content.side_effect = fake_generate
            img = _PILImage.new("RGB", (100, 100), "white")
            call_llm("extract this", image=img)

        parts = captured["contents"]
        self.assertIsInstance(parts, list)
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0], "extract this")
        image_part = parts[1]
        self.assertIsInstance(image_part, types.Part)
        self.assertEqual(image_part.inline_data.mime_type, "image/jpeg")


if __name__ == "__main__":
    unittest.main(verbosity=2)
